"""缓冲×闭指时长的配对对照；默认803开发检查、805独立检查轨迹集。

同一轨迹先运行原始控制器，再复用其规划超时决策，隔离计时抖动。
4组只改变接触后的动作：原关节制动/25 mm导纳，140/220 ms闭指。
保持原有两秒连续手指接触、无落地、保持漂移<=25 mm及实际关节合格标准。
"""
import argparse
import csv
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.contact_admittance import ContactAdmittance
from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene

VARIANTS=(("brake_140",False,.14),("admittance_140",True,.14),
          ("brake_220",False,.22),("admittance_220",True,.22))


def write_rows(path,rows):
    # 本实验目录由 exist_ok=False 创建；检查点仅更新本次运行自己的 CSV。
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


class BufferRecord:
    def __init__(self,model):
        self.closure=ContactClosure(model,LeapHand(model))
        self.rows=[];self.origin=self.normal=self.direction=None

    def observe(self,model,data,t,first,close,velocity,reference):
        ms=1000.*(t-first)
        if ms>=300.-1e-7:return
        palm=data.body("hand/palm_lower")
        f=model.joint("flight").dofadr[0]
        if self.origin is None:
            self.origin=palm.xpos.copy();self.normal=-palm.xmat.reshape(3,3)[:,2].copy()
            self.direction=velocity[f:f+3]/np.linalg.norm(velocity[f:f+3])
        self.closure.observe(model,data)
        self.rows.append(dict(contact_ms=round(ms,6),normal_velocity_m_s=float(velocity[f:f+3]@self.normal),
            opposed=int(self.closure.holding),retreat_mm=1000.*float((palm.xpos-self.origin)@self.direction)))

    def summary(self):
        def peak(lo,hi):
            return max([0.]+[r["normal_velocity_m_s"] for r in self.rows if lo<=r["contact_ms"]<hi])
        return dict(rebound_0_60_m_s=peak(0,60),rebound_60_160_m_s=peak(60,160),
            rebound_0_300_m_s=peak(0,300),support_formed=int(any(r["opposed"] for r in self.rows)),
            support_first300_ms=sum(r["opposed"] for r in self.rows)*2.,
            first_support_ms=next((r["contact_ms"] for r in self.rows if r["opposed"]),None),
            actual_retreat_first300_mm=max([0.]+[r["retreat_mm"] for r in self.rows]))


def summarize(rows):
    result=[]
    for seed in sorted(set(r["seed"] for r in rows)):
        for name,_,_ in VARIANTS:
            group=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            if not group:continue
            result.append(dict(seed=seed,strategy=name,n=len(group),held=sum(r["held"] for r in group),
                qualified=sum(r["qualified"] for r in group),support_formed=sum(r["support_formed"] for r in group),
                median_rebound_0_60=median(r["rebound_0_60_m_s"] for r in group),
                median_rebound_60_160=median(r["rebound_60_160_m_s"] for r in group),
                median_peak_normal_N=median(r["peak_normal_first100ms_N"] for r in group),
                median_retreat_mm=median(r["actual_retreat_first300_mm"] for r in group),
                buffer_qp_failed=sum(r["buffer_qp_failed"] for r in group),
                buffer_over_2ms=sum(r["buffer_over_2ms"] for r in group),
                buffer_reference_limit_steps=sum(r["buffer_reference_limit_steps"] for r in group)))
    return result


def plot(path,rows,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds=sorted(set(r["seed"] for r in rows));colors=["#274b77","#2b9b8f","#ce7b35","#9b5195"]
    fig,axes=plt.subplots(len(seeds),3,figsize=(13,4*len(seeds)),squeeze=False)
    labels=[v[0].replace("_","\n") for v in VARIANTS]
    for i,seed in enumerate(seeds):
        subset=[r for r in summary if r["seed"]==seed]
        axes[i,0].bar(range(4),[r["held"] for r in subset],color=colors)
        axes[i,0].plot(range(4),[r["support_formed"] for r in subset],"ko--",label="Opposed support <=300 ms")
        axes[i,0].set_ylim(0,max(r["n"] for r in subset)+1);axes[i,0].set_ylabel(f"Seed {seed}: trials")
        axes[i,0].set_title("Held for 2 s (bars) / support (dots)");axes[i,0].legend(fontsize=8)
        for col,key,title in ((1,"rebound_0_60_m_s","Early outward velocity: 0-60 ms"),
                              (2,"rebound_60_160_m_s","Later outward velocity: 60-160 ms")):
            for j,(name,_,_) in enumerate(VARIANTS):
                values=[r[key] for r in rows if r["seed"]==seed and r["strategy"]==name]
                axes[i,col].scatter(j+np.linspace(-.12,.12,len(values)),values,color=colors[j],s=20,alpha=.65)
                axes[i,col].plot([j-.23,j+.23],[median(values)]*2,color="black",lw=2)
            axes[i,col].set_title(title);axes[i,col].set_ylabel("Peak outward normal velocity (m/s)")
        for ax in axes[i]:
            ax.set_xticks(range(4),labels);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Paired post-contact buffering and closure-duration experiment",fontsize=15)
    fig.text(.5,.01,"Same pre-contact MPC and timeout decisions within each launch. 25 mm virtual stroke; unchanged hold/limit criteria.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(path/"buffer_ablation_en.svg");fig.savefig(path/"buffer_ablation_en.png",dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[803,805])
    parser.add_argument("--count",type=int,default=12)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    model=build_scene("ball");data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    rows=[];times=[];controls=[];calls_all=[]
    for seed in args.seeds:
        for scenario,kind,p,v in scenarios(center,model.opt.gravity,seed=seed):
            if kind!="near" or scenario>=args.count:continue
            schedule=None;baseline=None
            for name,buffered,duration in VARIANTS:
                holder=[]
                def factory(m,d,t,velocity):
                    control=ContactAdmittance(m,d,t,velocity);holder.append(control);return control
                record=BufferRecord(model)
                outcome,calls,_=run_case(model,p,v,scenario,kind,seed,"mpc_match",record.observe,schedule,
                    buffer_factory=factory if buffered else None,closure_duration=duration)
                if schedule is None:
                    schedule={round(c["time_s"]/model.opt.timestep):bool(c["overrun"]) for c in calls}
                    baseline=outcome
                # 改动仅在接触后开始，首触时间、来球相对速度和应用规划次数必须一致。
                for key in ("first_contact_s","first_normal_relative_speed_m_s","plans_applied"):
                    if outcome[key] is None and baseline[key] is None:continue
                    if not np.isclose(outcome[key],baseline[key],atol=1e-9,rtol=1e-9):
                        raise ValueError(f"Pre-contact mismatch: {seed}/{scenario}/{name}/{key}")
                tags=dict(seed=seed,scenario=scenario,strategy=name)
                control_rows=holder[0].rows if holder else []
                outcome.update(record.summary());outcome.update(strategy=name,closure_ms=duration*1000.,
                    buffer_qp_failed=sum(r["qp_failed"] for r in control_rows),
                    buffer_over_2ms=sum(r["wall_ms"]>2. for r in control_rows),
                    buffer_reference_limit_steps=sum(r["reference_speed"]>1.00001 or r["reference_accel"]>4.00001 or
                                                     r["reference_margin"]< -1e-6 for r in control_rows))
                rows.append(outcome)
                times.extend(dict(**tags,**r) for r in record.rows)
                controls.extend(dict(**tags,**r) for r in control_rows)
                calls_all.extend(dict(**c,strategy=name) for c in calls)
            print(seed,scenario,[(r["strategy"],r["held"],round(r["rebound_60_160_m_s"],3)) for r in rows[-4:]],flush=True)
            # 每条轨迹后保存，长实验中断时保留已完成数据。
            write_rows(args.output_dir/"trials.csv",rows)
    summary=summarize(rows)
    for filename,content in (("summary.csv",summary),("time_series.csv",times),("buffer_control.csv",controls),("solver_times.csv",calls_all)):
        if content:write_rows(args.output_dir/filename,content)
    plot(args.output_dir,rows,summary)
    for r in summary:print(r,flush=True)


if __name__=="__main__":main()
