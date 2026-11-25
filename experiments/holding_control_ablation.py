"""保持阶段三组配对对照：中段柔顺基线、仅摩擦风险反馈、力/力矩保持QP。

806用于已知案例开发检查；807为新验证来球。所有组前300ms捕获状态须一致。
不修改摩擦、碰撞材料、质量、执行器能力或原保持判定。
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from statistics import median
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.friction_hold import FrictionHoldClosure,HoldConfig
from controllers.hand import LeapHand
from controllers.joint_aware_closure import JointAwareClosure
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from experiments.mpc_contact_diagnosis import ContactRecord
from scene_setup.scene import build_scene

HOLD_VARIANTS=(("middle_compliant",None),("normal_hold","normal"),("wrench_hold","wrench"))


def summarize(rows):
    summary=[]
    for seed in sorted(set(r["seed"] for r in rows)):
        for name,_ in HOLD_VARIANTS:
            group=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            summary.append(dict(seed=seed,strategy=name,n=len(group),held=sum(r["held"] for r in group),
                qualified=sum(r["qualified"] for r in group),median_drift_mm=median(r["drift_mm"] for r in group),
                median_hold_slip_mm_s=median(r["hold_median_max_slip_mm_s"] for r in group),
                median_near_slide_load_fraction=median(r["hold_near_slide_load_fraction"] for r in group),
                qp_calls=sum(r["hold_qp_calls"] for r in group),qp_failed=sum(r["hold_qp_failed"] for r in group),
                qp_over_10ms=sum(r["hold_qp_over_10ms"] for r in group)))
    return summary


def plot(path,rows,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds=sorted(set(r["seed"] for r in rows));colors=["#274b77","#2b9b8f","#ce7b35"]
    fig,axes=plt.subplots(len(seeds),3,figsize=(12,4*len(seeds)),squeeze=False)
    labels=["Compliant\nbaseline","Friction-risk\nonly","Wrench\nhold QP"]
    for i,seed in enumerate(seeds):
        group=[r for r in summary if r["seed"]==seed]
        axes[i,0].bar(range(3),[r["held"] for r in group],color=colors)
        axes[i,0].set_ylim(0,max(r["n"] for r in group)+1);axes[i,0].set_ylabel(f"Seed {seed}: held trials")
        axes[i,0].set_title("Held continuously for 2 s")
        for col,key,title in ((1,"drift_mm","Maximum hold drift (mm)"),(2,"hold_median_max_slip_mm_s","Median contact slip speed (mm/s)")):
            for j,(name,_) in enumerate(HOLD_VARIANTS):
                values=[r[key] for r in rows if r["seed"]==seed and r["strategy"]==name]
                axes[i,col].scatter(j+np.linspace(-.1,.1,len(values)),values,color=colors[j],s=22,alpha=.7)
                axes[i,col].plot([j-.2,j+.2],[median(values)]*2,"k-",lw=2)
            axes[i,col].set_title(title)
        axes[i,1].axhline(25.,color="gray",ls=":",lw=1)
        for ax in axes[i]:ax.set_xticks(range(3),labels);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Capture-to-hold control: friction risk and relative-motion feedback",fontsize=14)
    fig.text(.5,.01,"Same capture state at 300 ms. Contact forces act through original torque-limited finger motors. Hold/arm-limit criteria unchanged.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(path/"holding_control_en.svg");fig.savefig(path/"holding_control_en.png",dpi=150);plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[806,807]);parser.add_argument("--cases",type=int,nargs="+",default=list(range(12)))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    prior=ROOT/"results/finger_closure_validation";old_calls=read(prior/"solver_times.csv");old_rows=read(prior/"trials.csv")
    model=build_scene("ball");data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    hand=LeapHand(model);dofs=[model.joint(f"hand/{i}").dofadr[0] for i in range(16)]
    rows=[];steps=[];points=[];control_steps=[];all_calls=[]
    for seed in args.seeds:
        for case,kind,p,v in scenarios(center,model.opt.gravity,seed=seed):
            if kind!="near" or case not in args.cases:continue
            selected=[r for r in old_calls if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="middle_compliant"]
            schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in selected} if selected else None
            capture_baseline=None
            for name,mode in HOLD_VARIANTS:
                holder=[];captured=[];torques=[];ctrl_errors=[]
                def factory(m,h):
                    control=JointAwareClosure(m,h,"compliant") if mode is None else FrictionHoldClosure(m,h,mode)
                    holder.append(control);return control
                def frame(m,d,info):
                    first=info["first_contact_s"]
                    if first is not None and abs(d.time-first-.3)<m.opt.timestep/4.:
                        captured.append(np.r_[d.qpos,d.qvel,d.ctrl].copy())
                    torques.append(float(max(abs(d.qfrc_actuator[dofs]))))
                    ctrl_errors.append(float(max(np.r_[d.ctrl[hand.ids]-hand.limits[:,1],hand.limits[:,0]-d.ctrl[hand.ids]])))
                record=ContactRecord(model,case,name,seed,True,point_window_ms=None)
                result,calls,_=run_case(model,p,v,case,kind,seed,"mpc_match",solver_schedule=schedule,
                    hand_controller_factory=factory,step_observer=record.observe,frame_observer=frame)
                assert len(captured)==1,(seed,case,name,"capture checkpoint")
                if capture_baseline is None:
                    capture_baseline=captured[0]
                    if schedule is None:schedule={round(r["time_s"]/model.opt.timestep):bool(r["overrun"]) for r in calls}
                    original=next((r for r in old_rows if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="middle_compliant"),None)
                    if original:
                        for key in ("held","drift_mm","first_contact_s","peak_normal_first100ms_N"):
                            assert np.isclose(result[key],float(original[key]),atol=1e-5,rtol=1e-8),(seed,case,key)
                else:assert np.allclose(captured[0],capture_baseline,atol=1e-12,rtol=0),(seed,case,name,"capture changed")
                assert max(torques)<=.950001 and max(ctrl_errors)<=1e-9
                late=[r for r in record.rows if r["contact_ms"]>=300.-1e-8]
                total=sum(sum(r[p+"_N"] for p in ("palm","finger1","finger2","finger3","thumb")) for r in late)
                control=holder[0];history=[] if mode is None else control.rows
                result.update(strategy=name,hold_median_max_slip_mm_s=median(r["max_slip_mm_s"] for r in late),
                    hold_near_slide_load_fraction=sum(r["near_sliding_load_N"] for r in late)/max(total,1e-12),
                    hold_qp_calls=sum(r["qp_called"] for r in history),hold_qp_failed=sum(r["qp_failed"] for r in history),
                    hold_qp_over_10ms=sum(r["wall_ms"]>10. for r in history),
                    actual_hand_peak_torque=max(torques),capture_checkpoint_match=1,
                    capture_hash=hashlib.sha256(captured[0].tobytes()).hexdigest(),**asdict(HoldConfig()))
                rows.append(result)
                tags=dict(seed=seed,scenario=case,strategy=name)
                control_steps.extend(dict(**tags,**r) for r in history)
                steps.extend(record.rows)
                points.extend(p for p in record.points if round(p["contact_ms"])%10==0)
                all_calls.extend(dict(**r,strategy=name) for r in calls)
            print(seed,case,[(r["strategy"],r["held"],round(r["drift_mm"],1),r["hold_qp_failed"]) for r in rows[-3:]],flush=True)
            write_rows(args.output_dir/"trials.csv",rows)
    summary=summarize(rows)
    for name,content in (("summary.csv",summary),("time_series.csv",steps),("contact_points_10ms.csv",points),
                         ("holding_control.csv",control_steps),("solver_times.csv",all_calls)):
        if content:write_rows(args.output_dir/name,content)
    plot(args.output_dir,rows,summary)
    for row in summary:print(row,flush=True)


if __name__=="__main__":main()
