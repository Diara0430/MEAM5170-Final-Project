"""对照原闭指、中段限角限速、中段限角限速+触球柔顺；手臂固定采用原MPC制动。

805为已观察开发集，806为参数冻结后的新验证集。保留原保持/实际关节合格标准。
只改接触后手指命令，同一来球逐条校验首触与已应用手臂计划一致。
"""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.joint_aware_closure import ClosureConfig,JointAwareClosure
from experiments.buffer_ablation import BufferRecord,write_rows
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene

FINGER_VARIANTS=(("original",None),("middle_limited","limited"),("middle_preshape","preshape"),("middle_compliant","compliant"))
MIDDLE=(2,6,10,14)


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


class FingerRecord(BufferRecord):
    def __init__(self,model,holder):
        super().__init__(model);self.holder=holder;self.detail=[]
        self.qadr=[model.joint(f"hand/{i}").qposadr[0] for i in MIDDLE]
        self.dofs=[model.joint(f"hand/{i}").dofadr[0] for i in MIDDLE]
        self.middle_bodies={model.jnt_bodyid[model.joint(f"hand/{i}").id] for i in MIDDLE}
        self.affected=set()
        for geom,body in enumerate(model.geom_bodyid):
            while body:
                if body in self.middle_bodies:self.affected.add(geom);break
                body=model.body_parentid[body]

    def observe(self,model,data,t,first,close,velocity,reference):
        super().observe(model,data,t,first,close,velocity,reference)
        ms=1000.*(t-first)
        if ms>=300.-1e-7:return
        load=power=positive_power=0.
        for i,contact in enumerate(data.contact):
            if self.closure.projectile not in (contact.geom1,contact.geom2) or contact.efc_address<0:continue
            second=contact.geom2==self.closure.projectile
            other=contact.geom1 if second else contact.geom2
            if other not in self.affected:continue
            wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
            force=(1. if second else -1.)*contact.frame.reshape(3,3).T@wrench[:3]
            jac=np.empty((3,model.nv));mujoco.mj_jac(model,data,jac,None,contact.pos,model.geom_bodyid[other])
            moving_boundary_power=float(force@(jac@velocity))
            load+=max(0.,wrench[0]);power+=moving_boundary_power;positive_power+=max(0.,moving_boundary_power)
        control=self.holder[0] if self.holder else None
        fraction=0. if close is None else float(np.clip((t-close)/.14,0,1))
        base_v=np.array([1.3]*3+[1.1])*6.*fraction*(1.-fraction)/.14
        base_q=np.array([1.3]*3+[1.1])*fraction*fraction*(3.-2.*fraction)
        shape_v=base_v if control is None else control.shape_velocity[list(MIDDLE)]
        shape_q=base_q if control is None else control.shape[list(MIDDLE)]
        self.detail.append(dict(contact_ms=round(ms,6),middle_load_N=load,
            moving_contact_power_W=power,positive_moving_contact_power_W=positive_power,
            **{f"q_{j}_rad":float(data.qpos[a]) for j,a in zip(MIDDLE,self.qadr)},
            **{f"v_{j}_rad_s":float(data.qvel[a]) for j,a in zip(MIDDLE,self.dofs)},
            **{f"shape_q_{j}_rad":float(q) for j,q in zip(MIDDLE,shape_q)},
            **{f"shape_v_{j}_rad_s":float(v) for j,v in zip(MIDDLE,shape_v)}))

    def summary(self):
        rows=self.detail;late=[r for r in rows if 60<=r["contact_ms"]<160]
        return dict(**super().summary(),
            middle_actual_peak_speed=max([0.]+[abs(r[f"v_{j}_rad_s"]) for r in rows for j in MIDDLE]),
            middle_actual_peak_angle=max([0.]+[r[f"q_{j}_rad"] for r in rows for j in MIDDLE[:3]]),
            middle_shape_peak_speed=max([0.]+[abs(r[f"shape_v_{j}_rad_s"]) for r in rows for j in MIDDLE]),
            middle_shape_peak_angle=max([0.]+[r[f"shape_q_{j}_rad"] for r in rows for j in MIDDLE[:3]]),
            middle_peak_load_60_160=max([0.]+[r["middle_load_N"] for r in late]),
            middle_positive_moving_contact_work_60_160_J=sum(r["positive_moving_contact_power_W"]*.002 for r in late))


def summarize(rows):
    summary=[]
    for seed in sorted(set(r["seed"] for r in rows)):
        for name,_ in FINGER_VARIANTS:
            group=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            summary.append(dict(seed=seed,strategy=name,n=len(group),held=sum(r["held"] for r in group),
                qualified=sum(r["qualified"] for r in group),support_formed=sum(r["support_formed"] for r in group),
                median_early_rebound=median(r["rebound_0_60_m_s"] for r in group),
                median_later_rebound=median(r["rebound_60_160_m_s"] for r in group),
                median_middle_actual_peak_speed=median(r["middle_actual_peak_speed"] for r in group),
                median_middle_actual_peak_angle=median(r["middle_actual_peak_angle"] for r in group),
                median_middle_later_peak_load_N=median(r["middle_peak_load_60_160"] for r in group),
                median_middle_positive_work_J=median(r["middle_positive_moving_contact_work_60_160_J"] for r in group)))
    return summary


def plot(path,rows,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds=sorted(set(r["seed"] for r in rows));colors=["#274b77","#2b9b8f","#ce7b35"]
    labels=["Original","Middle\nlimited","Middle\npreshape","Middle\ncompliant"]
    colors=["#274b77","#2b9b8f","#8565ac","#ce7b35"]
    fig,axes=plt.subplots(len(seeds),4,figsize=(15,4*len(seeds)),squeeze=False)
    for i,seed in enumerate(seeds):
        group=[r for r in summary if r["seed"]==seed]
        axes[i,0].bar(range(4),[r["held"] for r in group],color=colors)
        axes[i,0].set_ylim(0,max(r["n"] for r in group)+1);axes[i,0].set_title("Held for 2 s")
        axes[i,0].set_ylabel(f"Seed {seed}: trials")
        for col,key,title,unit in ((1,"middle_actual_peak_speed","Middle-joint peak speed","rad/s"),
            (2,"rebound_60_160_m_s","Outward velocity, 60-160 ms","m/s"),
            (3,"middle_positive_moving_contact_work_60_160_J","Positive moving-contact work*","J")):
            for j,(name,_) in enumerate(FINGER_VARIANTS):
                values=[r[key] for r in rows if r["seed"]==seed and r["strategy"]==name]
                axes[i,col].scatter(j+np.linspace(-.10,.10,len(values)),values,color=colors[j],alpha=.7,s=22)
                axes[i,col].plot([j-.2,j+.2],[median(values)]*2,"k-",lw=2)
            axes[i,col].set_title(title);axes[i,col].set_ylabel(unit)
        for ax in axes[i]:ax.set_xticks(range(4),labels);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Joint-aware closure: limited motion, preshape and contact compliance",fontsize=16)
    fig.text(.5,.01,"*Work from moving middle/distal contact boundaries, 60-160 ms; not net object energy. Unchanged hold/actual-arm-limit criteria.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(path/"finger_closure_en.svg");fig.savefig(path/"finger_closure_en.png",dpi=150);plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[805,806])
    parser.add_argument("--cases",type=int,nargs="+",default=list(range(12)))
    parser.add_argument("--angle",type=float,default=1.30);parser.add_argument("--speed",type=float,default=8.5)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    config=ClosureConfig(middle_angle=args.angle,middle_speed=args.speed)
    historical=ROOT/"results/contact_buffer_validation"
    old_calls=read(historical/"solver_times.csv");old_trials=read(historical/"trials.csv")
    model=build_scene("ball");data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    rows=[];steps=[];details=[];calls_all=[]
    for seed in args.seeds:
        for case,kind,p,v in scenarios(center,model.opt.gravity,seed=seed):
            if kind!="near" or case not in args.cases:continue
            matched=[r for r in old_calls if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="brake_140"]
            schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in matched} if matched else None
            baseline=None;preshape_baseline=None
            for name,mode in FINGER_VARIANTS:
                holder=[]
                def factory(m,h):
                    control=JointAwareClosure(m,h,mode,config);holder.append(control);return control
                record=FingerRecord(model,holder)
                result,calls,_=run_case(model,p,v,case,kind,seed,"mpc_match",step_observer=record.observe,
                    solver_schedule=schedule,hand_controller_factory=None if mode is None else factory,
                    allow_schedule_extension=True)
                if baseline is None:
                    baseline=result
                    if schedule is None:schedule={round(r["time_s"]/model.opt.timestep):bool(r["overrun"]) for r in calls}
                    prior=next((r for r in old_trials if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="brake_140"),None)
                    if prior:
                        for key in ("held","first_contact_s","close_s","drift_mm","peak_normal_first100ms_N"):
                            assert np.isclose(float(result[key]),float(prior[key]),atol=1e-5,rtol=1e-8),(seed,case,key)
                if mode=="preshape":preshape_baseline=result
                # 预成形是接触前的有意干预，只与相同预成形的柔顺组核对首触一致。
                compare=preshape_baseline if mode=="compliant" else baseline
                if mode!="preshape":
                    for key in ("first_contact_s","first_normal_relative_speed_m_s","plans_applied"):
                        assert np.isclose(result[key],compare[key],atol=1e-9,rtol=1e-9),(seed,case,name,key)
                tags=dict(seed=seed,scenario=case,strategy=name)
                result.update(record.summary());result.update(strategy=name,**asdict(config));rows.append(result)
                steps.extend(dict(**tags,**r) for r in record.rows);details.extend(dict(**tags,**r) for r in record.detail)
                calls_all.extend(dict(**r,strategy=name) for r in calls)
            print(seed,case,[(r["strategy"],r["held"],round(r["middle_actual_peak_speed"],2),round(r["rebound_60_160_m_s"],3)) for r in rows[-4:]],flush=True)
            write_rows(args.output_dir/"trials.csv",rows)
    summary=summarize(rows)
    for name,content in (("summary.csv",summary),("time_series.csv",steps),("finger_states.csv",details),("solver_times.csv",calls_all)):
        write_rows(args.output_dir/name,content)
    plot(args.output_dir,rows,summary)
    for r in summary:print(r,flush=True)


if __name__=="__main__":main()
