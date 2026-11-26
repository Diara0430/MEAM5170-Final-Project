"""实际加速度滤波×导纳缓冲对照。809开发，810保留为本轮新验证。

各组保持相同球体、手指协调控制、材料和原始成功判据。
滤波使用当前已知模型的一步预测，不把计算超时隐藏成实时通过。
"""
import argparse
from pathlib import Path
from statistics import median
import sys
import time

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.arm_reference_tracker import ArmReferenceTracker
from controllers.contact_admittance import ContactAdmittance
from controllers.contact_closure import ContactClosure
from controllers.discrete_arm_filter import DiscreteArmFilter
from controllers.hand import LeapHand
from controllers.sphere_cage import SphereCageClosure
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene

ARM_VARIANTS=(("coordinated",False,False,4.),("acceleration_filter",True,False,4.),
              ("filter_admittance",True,True,4.),("headroom_filter",True,False,3.))


class AuditedArmTracker(ArmReferenceTracker):
    """仅加日志，名义控制命令与历史基线完全一致。"""
    def __init__(self,model):super().__init__(model);self.rows=[]
    def command(self,data,plan,t):
        start=time.perf_counter();super().command(data,plan,t)
        self.rows.append(dict(time_s=t,wall_ms=1000.*(time.perf_counter()-start)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=[809,810])
    parser.add_argument("--cases",nargs="+",type=int,default=list(range(12)))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    prior_dir=ROOT/"results/sphere_cage_clearance_validation"
    prior=read(prior_dir/"trials.csv");previous_calls=read(prior_dir/"solver_times.csv")
    model=build_scene("ball");rest=initial(model);palm=rest.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    support=ContactClosure(model,LeapHand(model));hand_dof=[model.joint(f"hand/{i}").dofadr[0] for i in range(16)]
    rows=[];series=[];control=[];call_rows=[];buffer_rows=[]
    for seed in args.seeds:
        for case,kind,p,v in scenarios(center,model.opt.gravity,seed=seed):
            if kind!="near" or case not in args.cases:continue
            old_calls=[r for r in previous_calls if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="thumb_preshape"]
            schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in old_calls} if old_calls else None
            filtered_contact_checkpoint=None
            for name,filtered,buffered,planning_limit in ARM_VARIANTS:
                holders=[];buffers=[];frames=[];last_velocity=[];contact_checkpoint=[]
                def factory(m):
                    tr=DiscreteArmFilter(m) if filtered else AuditedArmTracker(m)
                    holders.append(tr);return tr
                def buffer_factory(m,d,t,velocity):
                    ref=ContactAdmittance(m,d,t,velocity);buffers.append(ref);return ref
                def frame(m,d,info):
                    tr=holders[0];velocity=d.qvel[tr.dofs].copy()
                    if not last_velocity:last_velocity.append(velocity);return
                    accel=(velocity-last_velocity[0])/m.opt.timestep;last_velocity[0]=velocity
                    first=info["first_contact_s"];start=float(d.time)-m.opt.timestep
                    if first is not None and not contact_checkpoint:contact_checkpoint.append(np.r_[d.qpos,d.qvel,d.ctrl].copy())
                    loads=np.zeros(5);net=np.zeros(3);depth=0.
                    for i,c in enumerate(d.contact):
                        a,b=support.group.get(c.geom1),support.group.get(c.geom2)
                        if a is not None and b is not None and 3 in (a,b):depth=max(depth,-float(c.dist))
                        if support.projectile not in (c.geom1,c.geom2) or c.efc_address<0:continue
                        second=c.geom2==support.projectile;other=c.geom1 if second else c.geom2;part=support.group.get(other)
                        if part is None:continue
                        force=np.empty(6);mujoco.mj_contactForce(m,d,i,force)
                        loads[part+1]+=max(0.,force[0]);net+=(1. if second else -1.)*c.frame.reshape(3,3).T@force[:3]
                    log=tr.rows[-1]
                    predict=np.array([log[f"predicted_accel_{i+1}"] for i in range(7)]) if filtered else accel
                    frames.append(dict(seed=seed,scenario=case,strategy=name,time_s=float(d.time),
                        contact_ms=-1. if first is None else 1000.*(start-first),precontact=int(first is None),
                        peak_actual_accel=float(max(abs(accel))),peak_actual_speed=float(max(abs(velocity))),
                        predicted_actual_error=float(max(abs(predict-accel))) if filtered else 0.,
                        summed_normal_N=float(loads.sum()),net_contact_force_N=float(np.linalg.norm(net)),thumb_N=float(loads[4]),
                        thumb_self_overlap_mm=1000.*max(0.,depth),hand_peak_torque_Nm=float(max(abs(d.qfrc_actuator[hand_dof]))),
                        controller_wall_ms=log["wall_ms"],finger_contact=int(info["finger_contact"]),
                        drift_mm=info["drift_mm"],**{f"accel_{i+1}":float(accel[i]) for i in range(7)}))
                result,calls,_=run_case(model,p,v,case,kind,seed,"mpc_match",solver_schedule=schedule,
                    allow_schedule_extension=True,hand_controller_factory=lambda m,h:SphereCageClosure(m,h),
                    tracker_factory=factory,buffer_factory=buffer_factory if buffered else None,frame_observer=frame,
                    mpc_acceleration_limit=planning_limit)
                if name=="acceleration_filter" and contact_checkpoint:filtered_contact_checkpoint=contact_checkpoint[0]
                if name=="filter_admittance" and filtered_contact_checkpoint is not None:
                    assert contact_checkpoint and np.allclose(contact_checkpoint[0],filtered_contact_checkpoint,atol=1e-12,rtol=0),(seed,case,"precontact buffer mismatch")
                if name=="coordinated":
                    if schedule is None:schedule={round(r["time_s"]/model.opt.timestep):bool(r["overrun"]) for r in calls}
                    old=next((r for r in prior if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="thumb_preshape"),None)
                    if old:
                        for key in ("held","qualified","drift_mm","first_contact_s","peak_accel","peak_normal_first100ms_N"):
                            assert np.isclose(result[key],float(old[key]),atol=1e-5,rtol=1e-8),(seed,case,key)
                tr=holders[0];tags=dict(seed=seed,scenario=case,strategy=name)
                early=[r for r in frames if 0<=r["contact_ms"]<300.]
                assert max(r["hand_peak_torque_Nm"] for r in frames)<=.950001
                assert max(r["predicted_actual_error"] for r in frames)<1e-7
                result.update(strategy=name,planning_acceleration_limit=planning_limit,
                    peak_accel_precontact=max([0.]+[r["peak_actual_accel"] for r in frames if r["precontact"]]),
                    peak_accel_postcontact=max([0.]+[r["peak_actual_accel"] for r in frames if not r["precontact"]]),
                    peak_summed_normal_first300_N=max([0.]+[r["summed_normal_N"] for r in early]),
                    peak_net_force_first300_N=max([0.]+[r["net_contact_force_N"] for r in early]),
                    peak_thumb_first300_N=max([0.]+[r["thumb_N"] for r in early]),
                    max_thumb_self_overlap_mm=max(r["thumb_self_overlap_mm"] for r in frames),
                    max_hand_torque_Nm=max(r["hand_peak_torque_Nm"] for r in frames),
                    max_prediction_error=max(r["predicted_actual_error"] for r in frames),
                    filter_steps=len(tr.rows) if filtered else 0,filter_overruns=sum(r.get("overrun",0) for r in tr.rows),
                    filter_failures=sum(r.get("qp_failed",0) for r in tr.rows),
                    filter_remaining_violations=sum(r.get("predicted_violation",0)>1e-4 for r in tr.rows),
                    controller_p50_ms=float(np.percentile([r["wall_ms"] for r in tr.rows],50)),
                    controller_p95_ms=float(np.percentile([r["wall_ms"] for r in tr.rows],95)),
                    controller_max_ms=max(r["wall_ms"] for r in tr.rows),
                    computation_delay_charged=0,realtime_certified=0)
                rows.append(result);series.extend(frames)
                if filtered:control.extend(dict(**tags,**r) for r in tr.rows)
                call_rows.extend(dict(**r,strategy=name) for r in calls)
                if buffers:buffer_rows.extend(dict(**tags,**r) for r in buffers[0].rows)
                print(seed,case,name,"held",result["held"],"qualified",result["qualified"],"acc",round(result["peak_accel"],3),
                    "drift",round(result["drift_mm"],1),"normal",round(result["peak_normal_first100ms_N"],1),"overruns",result["filter_overruns"],flush=True)
                write_rows(args.output_dir/"trials.csv",rows)
    summary=[]
    for seed in args.seeds:
        for name,_,_,_ in ARM_VARIANTS:
            group=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            summary.append(dict(seed=seed,strategy=name,n=len(group),held=sum(r["held"] for r in group),
                qualified=sum(r["qualified"] for r in group),accel_within4=sum(r["peak_accel"]<=4. for r in group),
                median_peak_accel=median(r["peak_accel"] for r in group),max_peak_accel=max(r["peak_accel"] for r in group),
                median_peak_normal_first100_N=median(r["peak_normal_first100ms_N"] for r in group),
                median_drift_mm=median(r["drift_mm"] for r in group),filter_overruns=sum(r["filter_overruns"] for r in group),
                filter_steps=sum(r["filter_steps"] for r in group),filter_failures=sum(r["filter_failures"] for r in group)))
    for file,content in (("summary.csv",summary),("time_series.csv",series),("filter_control.csv",control),
                         ("solver_times.csv",call_rows),("buffer_control.csv",buffer_rows)):
        if content:write_rows(args.output_dir/file,content)
    for row in summary:print(row,flush=True)


if __name__=="__main__":main()
