"""延迟预测加入手指反馈/首触事件与加速度余量的消融对照。811开发，812新检查。"""
import argparse
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.coupled_hand_forecast import CoupledHandForecast
from controllers.sphere_cage import SphereCageClosure
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene

HAND_FORECAST_VARIANTS=(("frozen_2ms",1,False,3.8),("frozen_margin_2ms",1,False,3.5),
    ("coupled_2ms",1,True,3.8),("coupled_margin_2ms",1,True,3.5),
    ("frozen_4ms",2,False,3.8),("coupled_margin_4ms",2,True,3.5))
HAND_REPLAY_NAMES=("frozen_2ms","coupled_2ms","coupled_margin_2ms","coupled_margin_4ms")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[811,812]);parser.add_argument("--cases",type=int,nargs="+",default=list(range(12)))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    previous=ROOT/"results/contact_delay_final";old_trials=read(previous/"trials.csv");old_calls=read(previous/"solver_times.csv")
    model=build_scene("ball");data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    hand_dof=[model.joint(f"hand/{i}").dofadr[0] for i in range(16)]
    trials=[];controls=[];calls_all=[];summary=[];hand_plans=[]
    for seed in args.seeds:
        seed_series=[]
        for case,kind,p,v in scenarios(center,model.opt.gravity,seed=seed):
            if kind!="near" or case not in args.cases:continue
            recorded=[r for r in old_calls if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="reference_now"]
            schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in recorded} if recorded else None
            for name,delay,feedback,limit in HAND_FORECAST_VARIANTS:
                trackers=[];hands=[];frames=[];previous_velocity=[]
                def factory(m):
                    tr=CoupledHandForecast(m,delay,feedback,limit);trackers.append(tr);return tr
                def hand_factory(m,h):
                    hand=SphereCageClosure(m,h);hands.append(hand);return hand
                def frame(m,d,info):
                    tr=trackers[0];velocity=d.qvel[tr.dofs].copy()
                    if not previous_velocity:previous_velocity.append(velocity);return
                    acceleration=(velocity-previous_velocity[0])/m.opt.timestep;previous_velocity[0]=velocity
                    first=info["first_contact_s"];age=-1. if first is None else 1000.*(d.time-m.opt.timestep-first)
                    applied=tr.applied;expected=tr.applied_acceleration_prediction;log=tr.rows[-1]
                    load=0.;net=np.zeros(3)
                    for i,c in enumerate(d.contact):
                        if tr.projectile not in (c.geom1,c.geom2) or c.efc_address<0:continue
                        second=c.geom2==tr.projectile;other=c.geom1 if second else c.geom2
                        if other not in tr.hand_geoms:continue
                        wrench=np.empty(6);mujoco.mj_contactForce(m,d,i,wrench)
                        load+=max(0.,wrench[0]);net+=(1. if second else -1.)*c.frame.reshape(3,3).T@wrench[:3]
                    frames.append(dict(seed=seed,scenario=case,strategy=name,time_s=float(d.time),contact_ms=age,
                        actual_peak_accel=float(max(abs(acceleration))),actual_peak_speed=float(max(abs(velocity))),
                        prediction_error=None if expected is None else float(max(abs(acceleration-expected))),
                        hand_target_error=log["applied_hand_target_error_rad"],hand_q_error=log["applied_hand_q_error_rad"],
                        hand_v_error=log["applied_hand_v_error_rad_s"],
                        contact_mode_match=None if applied is None else int(applied["contact_signature"]==tr.contact_signature(d)),
                        applied_age_steps=log["applied_age_steps"],applied_generation_step=log["applied_generation_step"],
                        snapshot_unchanged=log["snapshot_unchanged"],drift_mm=info["drift_mm"],
                        summed_normal_N=load,net_force_N=float(np.linalg.norm(net)),hand_peak_torque_Nm=float(max(abs(d.qfrc_actuator[hand_dof]))),
                        wall_ms=log["wall_ms"],over_budget=int(log["wall_ms"]>delay*m.opt.timestep*1000.),
                        **{f"accel_{i+1}":float(acceleration[i]) for i in range(7)}))
                result,calls,_=run_case(model,p,v,case,kind,seed,"mpc_match",solver_schedule=schedule,allow_schedule_extension=True,
                    tracker_factory=factory,mpc_acceleration_limit=3.,hand_controller_factory=hand_factory,frame_observer=frame)
                old=None
                if name=="frozen_2ms":
                    if schedule is None:schedule={round(r["time_s"]/model.opt.timestep):bool(int(r["overrun"])) for r in calls}
                    old=next((r for r in old_trials if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="reference_predictive_2ms"),None)
                    if old:
                        # 旧数据未记录晚期拇指规划的超时决策，保持阶段只报告差异，不能冒充逐步复现。
                        for key in ("first_contact_s","peak_normal_first100ms_N"):
                            assert np.isclose(result[key],float(old[key]),atol=1e-5,rtol=1e-8),(seed,case,key)
                assert all(r["snapshot_unchanged"] for r in frames)
                assert all(r["applied_age_steps"] in (-1,delay) for r in frames)
                assert max(r["hand_peak_torque_Nm"] for r in frames)<=.950001
                valid=[r for r in frames if r["prediction_error"] is not None];early=[r for r in valid if 0<=r["contact_ms"]<300.]
                result.update(strategy=name,delay_ms=2*delay,hand_feedback=int(feedback),filter_limit=limit,steps=len(frames),
                    historical_held=None if old is None else int(old["held"]),historical_drift_mm=None if old is None else float(old["drift_mm"]),
                    historical_full_match=None if old is None else int(all(np.isclose(result[k],float(old[k]),atol=1e-5,rtol=1e-8) for k in ("held","qualified","drift_mm","peak_accel"))),
                    p95_prediction_error=float(np.percentile([r["prediction_error"] for r in valid],95)),
                    peak_prediction_error=max(r["prediction_error"] for r in valid),
                    p95_capture_prediction_error=float(np.percentile([r["prediction_error"] for r in early],95)) if early else 0.,
                    p95_hand_target_error=float(np.percentile([r["hand_target_error"] for r in valid],95)),
                    peak_hand_target_error=max(r["hand_target_error"] for r in valid),
                    capture_mode_match=float(np.mean([r["contact_mode_match"] for r in early])) if early else 0.,
                    peak_precontact_accel=max([0.]+[r["actual_peak_accel"] for r in frames if r["contact_ms"]<0]),
                    peak_postcontact_accel=max([0.]+[r["actual_peak_accel"] for r in frames if r["contact_ms"]>=0]),
                    p95_wall_ms=float(np.percentile([r["wall_ms"] for r in frames],95)),over_budget=sum(r["over_budget"] for r in frames),
                    qp_failures=sum(r["qp_failed"] for r in trackers[0].rows),
                    snapshot_checks=len(frames),fixed_arm_delay_charged=1,wall_clock_overruns_charged=0,realtime_certified=0)
                trials.append(result);seed_series.extend(frames);tags=dict(seed=seed,scenario=case,strategy=name)
                controls.extend(dict(**tags,**r) for r in trackers[0].rows);calls_all.extend(dict(**r,strategy=name) for r in calls)
                hand_plans.extend(dict(**tags,**r) for r in hands[0].rows if r["plan_called"])
                write_rows(args.output_dir/"trials.csv",trials)
                print(seed,case,name,"held",result["held"],"qual",result["qualified"],"a",round(result["peak_accel"],3),
                    "drift",round(result["drift_mm"],1),"e95",round(result["p95_prediction_error"],5),flush=True)
        write_rows(args.output_dir/f"time_series_{seed}.csv",seed_series)
        for name,_,_,_ in HAND_FORECAST_VARIANTS:
            g=[r for r in trials if r["seed"]==seed and r["strategy"]==name]
            summary.append(dict(seed=seed,strategy=name,n=len(g),held=sum(r["held"] for r in g),qualified=sum(r["qualified"] for r in g),
                accel_within4=sum(r["peak_accel"]<=4 for r in g),median_peak_accel=median(r["peak_accel"] for r in g),
                max_peak_accel=max(r["peak_accel"] for r in g),median_drift_mm=median(r["drift_mm"] for r in g),
                median_error_p95=median(r["p95_prediction_error"] for r in g),median_capture_error_p95=median(r["p95_capture_prediction_error"] for r in g),
                median_capture_mode_match=median(r["capture_mode_match"] for r in g),over_budget=sum(r["over_budget"] for r in g),steps=sum(r["steps"] for r in g)))
    for file,content in (("summary.csv",summary),("command_pipeline.csv",controls),("solver_times.csv",calls_all),("hand_planner.csv",hand_plans)):
        if content:write_rows(args.output_dir/file,content)
    for r in summary:print(r,flush=True)


if __name__=="__main__":main()
