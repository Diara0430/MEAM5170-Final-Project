"""固定执行延迟下，比较独立制动、实测状态缓冲和模型预测补偿。

810开发检查，811为固定设计后的新验证；固定2/4 ms仅施加于七轴机械臂命令。
球体和仿真时间在排队期间照常演化；手指仍每2 ms更新。无机器学习。
"""
import argparse
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from controllers.measured_contact_reference import MeasuredContactReference
from controllers.scheduled_arm_tracker import ScheduledArmTracker
from controllers.sphere_cage import SphereCageClosure
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene

DELAY_VARIANTS=(("reference_now",False,0,False),("reference_2ms",False,1,False),
    ("measured_now",True,0,False),("measured_2ms",True,1,False),
    ("predictive_2ms",True,1,True),("predictive_4ms",True,2,True),
    ("reference_predictive_2ms",False,1,True),("reference_predictive_4ms",False,2,True))

DELAY_REPLAY_NAMES=("reference_now","reference_2ms","reference_predictive_2ms","reference_predictive_4ms")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[810,811])
    parser.add_argument("--cases",type=int,nargs="+",default=list(range(12)))
    parser.add_argument("--variants",nargs="+",choices=[r[0] for r in DELAY_VARIANTS])
    parser.add_argument("--baseline-dir",type=Path,default=ROOT/"results/arm_filter_validation")
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    previous=args.baseline_dir
    old_trials=read(previous/"trials.csv");old_calls=read(previous/"solver_times.csv")
    m=build_scene("ball");rest=initial(m);palm=rest.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    closure=ContactClosure(m,LeapHand(m));hand_dof=[m.joint(f"hand/{i}").dofadr[0] for i in range(16)]
    trials=[];series=[];commands=[];references=[];call_rows=[]
    for seed in args.seeds:
        for case,kind,p,v in scenarios(center,m.opt.gravity,seed=seed):
            if kind!="near" or case not in args.cases:continue
            historical=[r for r in old_calls if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"] in ("headroom_filter","reference_now")]
            schedule={round(float(r["time_s"])/m.opt.timestep):bool(int(r["overrun"])) for r in historical} if historical else None
            checkpoints={}
            for name,buffered,delay,predictive in DELAY_VARIANTS:
                if args.variants and name not in args.variants:continue
                trackers=[];buffers=[];frames=[];last=[];first_state=[]
                def factory(model):
                    control=ScheduledArmTracker(model,delay,predictive);trackers.append(control);return control
                def buffer_factory(model,data,now,velocity):
                    ref=MeasuredContactReference(model,data,now,velocity);buffers.append(ref);return ref
                def frame(model,data,info):
                    tr=trackers[0];velocity=data.qvel[tr.dofs].copy()
                    if not last:last.append(velocity);return
                    accel=(velocity-last[0])/model.opt.timestep;last[0]=velocity
                    first=info["first_contact_s"];start=float(data.time)-model.opt.timestep
                    if first is not None and not first_state:first_state.append(np.r_[data.qpos,data.qvel,data.ctrl].copy())
                    loads=np.zeros(5);force=np.zeros(3);self_depth=0.
                    for i,c in enumerate(data.contact):
                        a,b=closure.group.get(c.geom1),closure.group.get(c.geom2)
                        if a is not None and b is not None and 3 in (a,b):self_depth=max(self_depth,-float(c.dist))
                        if closure.projectile not in (c.geom1,c.geom2) or c.efc_address<0:continue
                        second=c.geom2==closure.projectile;part=closure.group.get(c.geom1 if second else c.geom2)
                        if part is None:continue
                        wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
                        loads[part+1]+=max(0.,wrench[0]);force+=(1. if second else -1.)*c.frame.reshape(3,3).T@wrench[:3]
                    log=tr.rows[-1];ref_log=buffers[0].rows[-1] if buffers and buffers[0].rows else None
                    wall=log["wall_ms"]+(0. if ref_log is None else ref_log["wall_ms"])
                    expected=tr.applied_acceleration_prediction
                    # 对齐的是实际交付指令的预测，不是刚生成但仍在队列中的新预测。
                    frames.append(dict(seed=seed,scenario=case,strategy=name,time_s=float(data.time),
                        contact_ms=-1. if first is None else 1000.*(start-first),precontact=int(first is None),
                        actual_peak_accel=float(max(abs(accel))),actual_peak_speed=float(max(abs(velocity))),
                        applied_age_steps=log["applied_age_steps"],applied_generation_step=log["applied_generation_step"],
                        applied_prediction_error=None if expected is None else float(max(abs(accel-expected))),
                        summed_normal_N=float(loads.sum()),net_force_N=float(np.linalg.norm(force)),thumb_N=float(loads[4]),
                        thumb_self_overlap_mm=1000.*max(0.,self_depth),hand_peak_torque_Nm=float(max(abs(data.qfrc_actuator[hand_dof]))),
                        drift_mm=info["drift_mm"],finger_contact=int(info["finger_contact"]),controller_wall_ms=wall,
                        exceeds_injected_delay=int(wall>max(1,delay)*model.opt.timestep*1000.),
                        measured_retreat_mm=0. if ref_log is None else ref_log["measured_retreat_mm"],
                        **{f"accel_{i+1}":float(accel[i]) for i in range(7)}))
                result,calls,_=run_case(m,p,v,case,kind,seed,"mpc_match",solver_schedule=schedule,allow_schedule_extension=True,
                    tracker_factory=factory,mpc_acceleration_limit=3.,hand_controller_factory=lambda m,h:SphereCageClosure(m,h),
                    buffer_factory=buffer_factory if buffered else None,frame_observer=frame)
                if name=="reference_now":
                    if schedule is None:schedule={round(r["time_s"]/m.opt.timestep):bool(int(r["overrun"])) for r in calls}
                    old=next((r for r in old_trials if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"] in ("headroom_filter","reference_now")),None)
                    if old:
                        for key in ("held","qualified","first_contact_s","drift_mm","peak_accel","peak_normal_first100ms_N"):
                            assert np.isclose(result[key],float(old[key]),atol=1e-5,rtol=1e-8),(seed,case,key)
                if first_state:checkpoints[name]=first_state[0]
                for left,right in (("reference_now","measured_now"),("reference_2ms","measured_2ms")):
                    if name==right and left in checkpoints:
                        assert right in checkpoints and np.allclose(checkpoints[left],checkpoints[right],atol=1e-12,rtol=0),(seed,case,"precontact mismatch")
                # 延迟真实性：每个已交付命令的生成时间必须早于当前时刻相应步数。
                assert all(r["applied_age_steps"] in (-1,delay) for r in frames)
                assert all(r["applied_generation_step"]==-1 if round((r["time_s"]-m.opt.timestep)/m.opt.timestep)<delay else
                    r["applied_generation_step"]==round((r["time_s"]-m.opt.timestep)/m.opt.timestep)-delay for r in frames)
                assert max(r["hand_peak_torque_Nm"] for r in frames)<=.950001
                early=[r for r in frames if 0<=r["contact_ms"]<300.]
                prediction_errors=[r["applied_prediction_error"] for r in frames if r["applied_prediction_error"] is not None]
                result.update(strategy=name,delay_ms=delay*m.opt.timestep*1000.,predictive=int(predictive),measured_reference=int(buffered),
                    peak_precontact_accel=max([0.]+[r["actual_peak_accel"] for r in frames if r["precontact"]]),
                    peak_postcontact_accel=max([0.]+[r["actual_peak_accel"] for r in frames if not r["precontact"]]),
                    peak_normal_first300_N=max([0.]+[r["summed_normal_N"] for r in early]),
                    peak_net_force_first300_N=max([0.]+[r["net_force_N"] for r in early]),
                    max_thumb_self_overlap_mm=max(r["thumb_self_overlap_mm"] for r in frames),
                    max_hand_torque_Nm=max(r["hand_peak_torque_Nm"] for r in frames),
                    max_retreat_mm=max([0.]+[r["measured_retreat_mm"] for r in frames]),
                    peak_applied_prediction_error=max(prediction_errors,default=0.),
                    median_applied_prediction_error=median(prediction_errors) if prediction_errors else 0.,
                    controller_p95_ms=float(np.percentile([r["controller_wall_ms"] for r in frames],95)),
                    runtime_over_injected_delay=sum(r["exceeds_injected_delay"] for r in frames),
                    steps=len(frames),qp_failures=sum(r["qp_failed"] for r in trackers[0].rows),
                    fixed_arm_delay_charged=int(delay>0),wall_time_latency_charged=0,realtime_certified=0)
                trials.append(result);series.extend(frames);tags=dict(seed=seed,scenario=case,strategy=name)
                commands.extend(dict(**tags,**r) for r in trackers[0].rows)
                if buffers:references.extend(dict(**tags,**r) for r in buffers[0].rows)
                call_rows.extend(dict(**r,strategy=name) for r in calls)
                print(seed,case,name,"held",result["held"],"qual",result["qualified"],"a",round(result["peak_accel"],3),
                    "drift",round(result["drift_mm"],1),"load",round(result["peak_normal_first100ms_N"],1),flush=True)
                write_rows(args.output_dir/"trials.csv",trials)
    summary=[]
    for seed in args.seeds:
        for name,_,_,_ in DELAY_VARIANTS:
            if args.variants and name not in args.variants:continue
            group=[r for r in trials if r["seed"]==seed and r["strategy"]==name]
            summary.append(dict(seed=seed,strategy=name,n=len(group),held=sum(r["held"] for r in group),qualified=sum(r["qualified"] for r in group),
                accel_within4=sum(r["peak_accel"]<=4 for r in group),median_peak_accel=median(r["peak_accel"] for r in group),
                max_peak_accel=max(r["peak_accel"] for r in group),median_peak_normal_N=median(r["peak_normal_first100ms_N"] for r in group),
                median_drift_mm=median(r["drift_mm"] for r in group),runtime_over_injected_delay=sum(r["runtime_over_injected_delay"] for r in group),
                steps=sum(r["steps"] for r in group)))
    for file,content in (("summary.csv",summary),("time_series.csv",series),("command_pipeline.csv",commands),
                         ("contact_reference.csv",references),("solver_times.csv",call_rows)):
        if content:write_rows(args.output_dir/file,content)
    for r in summary:print(r,flush=True)


if __name__=="__main__":main()
