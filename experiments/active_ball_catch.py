"""移动机械臂的真实接球实验，规划延迟期间球体继续自由运动。

固定手掌对照，以及预测闭指/接触触发/二者取早的主动接取。
不增加倾腕，计划接取时刻后留300ms过渡，再要求原姿态稳定保持2秒。
"""
import argparse
import copy
from pathlib import Path
from statistics import median
import sys
import time

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.arm_reference_tracker import ArmReferenceTracker
from controllers.hand import LeapHand
from controllers.intercept_planner import InterceptPlanner,InterceptPlan
from experiments.contact_impulse import write_rows
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.static_grasp import contacts_with
from scene_setup.scene import build_scene


def hold_plan(model,data):
    q=np.array([data.joint(f"joint{i}").qpos[0] for i in range(1,8)])
    c=np.zeros((6,7));c[0]=q
    palm=data.body("hand/palm_lower")
    return InterceptPlan(0.,1.,palm.xpos.copy(),palm.xmat.reshape(3,3).copy(),q,c)


def launch(model,p,v):
    data=initial(model)
    data.joint("flight").qpos[:3]=p
    data.joint("flight").qvel[:3]=v
    mujoco.mj_forward(model,data)
    return data


def prepare(model,p,v,budget=.200):
    data=launch(model,p,v)
    planner=InterceptPlanner(model,data)
    start=time.perf_counter()
    # 从预留计算时间结束后的球体预测状态开始规划，而不是事后压缩旧轨迹。
    predicted_p=p+v*budget+.5*model.opt.gravity*budget**2
    predicted_v=v+model.opt.gravity*budget
    plan=planner.plan(predicted_p,predicted_v,now=budget,angles=(0.,),warm_start=True)
    wall=time.perf_counter()-start
    info=dict(planning_wall_s=wall,charged_delay_s=0.,start_state_error_rad=None,
              object_prediction_error_mm=None,planning_contact_s=None,planned_deadline_s=None,plan_reason="no_candidate")
    if plan is None:
        return data,None,info
    info["planned_deadline_s"]=plan.intercept_time
    # 预留30ms接口开销；球体运动推进到预算结束，参考轨迹也在此时开始。
    dt=model.opt.timestep
    delay=np.ceil(budget/dt)*dt
    info["charged_delay_s"]=delay
    if wall+.030>budget:
        info["plan_reason"]="planning_budget_exceeded"
        return data,None,info
    if delay>=plan.intercept_time-.040:
        info["plan_reason"]="planning_too_late"
        return data,None,info
    tracker=ArmReferenceTracker(model)
    stationary=hold_plan(model,data)
    projectile=model.geom("projectile").id
    hand_geoms={i for i,b in enumerate(model.geom_bodyid) if (model.body(b).name or "").startswith("hand/")}
    contact_during_planning=False
    for k in range(round(delay/dt)):
        tracker.command(data,stationary,k*dt)
        mujoco.mj_step(model,data)
        contact_during_planning |= bool(contacts_with(data,projectile)&hand_geoms)
        if contact_during_planning and info["planning_contact_s"] is None:
            info["planning_contact_s"]=k*dt
    if contact_during_planning:
        info["plan_reason"]="contact_during_planning"
        return data,None,info
    # 未来起点基于机械臂等待期间保持静止的预测；偏离过大就拒绝执行。
    q0,v0,_=plan.reference(delay)
    info["start_state_error_rad"]=float(max(abs(data.qpos[planner.qadr]-q0)))
    info["object_prediction_error_mm"]=1000*float(np.linalg.norm(data.joint("flight").qpos[:3]-predicted_p))
    if info["start_state_error_rad"]>.001 or max(abs(data.qvel[planner.dofs]-v0))>.01:
        info["plan_reason"]="waiting_state_changed"
        return data,None,info
    info["plan_reason"]="accepted"
    return data,plan,info


def run_trial(model,p,v,scenario,kind,mode,prepared):
    prepared_data,plan,info=prepared
    static=mode=="static_predict"
    info=dict(info)
    if static:
        data=launch(model,p,v)
        reference=hold_plan(model,data)
        center=data.body("hand/palm_lower").xpos+data.body("hand/palm_lower").xmat.reshape(3,3)@np.array([0.,-.055,-.08])
        roots=np.roots([.5*model.opt.gravity[2],v[2],p[2]-center[2]])
        crossing=[float(r.real) for r in roots if abs(r.imag)<1e-9 and r.real>0 and v[2]+model.opt.gravity[2]*r.real<0]
        deadline=min(crossing) if crossing else .5
        info.update(planning_wall_s=0.,charged_delay_s=0.,start_state_error_rad=0.,object_prediction_error_mm=0.,
                    planning_contact_s=None,planned_deadline_s=deadline,plan_reason="static_baseline")
    else:
        if plan is None:
            return dict(scenario=scenario,kind=kind,mode=mode,**info,accepted=0,contacted=int(info["planning_contact_s"] is not None),held_success=0,qualified_success=0,
                first_contact_s=info["planning_contact_s"],close_s=None,max_drift_mm=None,peak_precontact_accel=None,peak_postcontact_accel=None,
                peak_speed=None,max_force_fraction=None,joint_margin=None,deadline_position_mm=None),[]
        data=copy.copy(prepared_data)
        reference=plan
        deadline=plan.intercept_time
    tracker=ArmReferenceTracker(model)
    hand=LeapHand(model)
    checker=InterceptPlanner(model,data)
    projectile=model.geom("projectile").id
    floor=model.geom("floor").id
    palm=model.body("hand/palm_lower").id
    hand_geoms={i for i,b in enumerate(model.geom_bodyid) if (model.body(b).name or "").startswith("hand/")}
    fingers={i for i in hand_geoms if model.geom_bodyid[i]!=palm}
    first_contact=close=None
    grounded=False
    continuous=True
    origin=None
    drift=pre_accel=post_accel=peak_speed=force_fraction=0.
    margin=float("inf")
    endpoint=None
    trace=[]
    dt=model.opt.timestep
    hold_start=deadline+.3
    while data.time<hold_start+2.-dt/2:
        t=float(data.time)
        if close is None:
            predicted=t>=deadline-.040-1e-12
            touched=first_contact is not None
            trigger=predicted if mode in ("static_predict","active_predict") else touched if mode=="active_contact" else predicted or touched
            if trigger:
                close=t
        if close is not None:
            hand.command(data,hand.open_pose,hand.closed,(t-close)/.140)
        tracker.command(data,reference,t)
        previous=data.qvel[tracker.dofs].copy()
        mujoco.mj_step(model,data)
        touching=contacts_with(data,projectile)
        touched=bool(touching&hand_geoms)
        if first_contact is None and touched:
            first_contact=t
        grounded |= floor in touching
        accel=float(max(abs((data.qvel[tracker.dofs]-previous)/dt)))
        if first_contact is None:
            pre_accel=max(pre_accel,accel)
        else:
            post_accel=max(post_accel,accel)
        peak_speed=max(peak_speed,float(max(abs(data.qvel[tracker.dofs]))))
        force_fraction=max(force_fraction,float(max(abs(data.actuator_force[tracker.ids])/
            np.max(abs(model.actuator_forcerange[tracker.ids]),axis=1))))
        q=data.qpos[tracker.qadr]
        margin=min(margin,float(min(np.r_[q-checker.lower,checker.upper-q])))
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        palm_state=data.body(palm)
        rotation=palm_state.xmat.reshape(3,3)
        relative=rotation.T@(data.body("projectile").xpos-palm_state.xpos)
        if t+dt>=hold_start-1e-10:
            if origin is None:
                origin=relative.copy()
            drift=max(drift,float(np.linalg.norm(relative-origin)))
            continuous &= bool(touching&fingers)
        if endpoint is None and t+dt>=deadline-1e-10:
            endpoint=1000*float(np.linalg.norm(palm_state.xpos+rotation@checker.local_capture-reference.point)) if not static else None
        if round((t+dt)/dt)%5==0:
            trace.append(dict(scenario=scenario,mode=mode,time_s=t+dt,finger_contact=int(bool(touching&fingers)),
                floor=int(floor in touching),relative_x_mm=1000*relative[0],relative_y_mm=1000*relative[1],
                relative_z_mm=1000*relative[2],hold_drift_mm=1000*drift))
    held=first_contact is not None and not grounded and continuous and drift<=.025
    qualified=held and peak_speed<=1. and max(pre_accel,post_accel)<=4. and margin>=0 and force_fraction<=1.000001
    return dict(scenario=scenario,kind=kind,mode=mode,**info,accepted=1,contacted=int(first_contact is not None),
        held_success=int(held),qualified_success=int(qualified),first_contact_s=first_contact,close_s=close,
        max_drift_mm=1000*drift,peak_precontact_accel=pre_accel,peak_postcontact_accel=post_accel,
        peak_speed=peak_speed,max_force_fraction=force_fraction,joint_margin=margin,deadline_position_mm=endpoint),trace


def plot(path,summary):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="340">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="Arial" fill="#18364a">',
         '<text x="25" y="35" font-size="22">Active ball catching with planning latency</text>',
         '<text x="25" y="62" font-size="13">Nearby trajectories, same 12 releases. Active planning reserves 200 ms before arm motion.</text>']
    columns=((25,"Strategy"),(300,"Executed"),(440,"Contacted"),(580,"Held for 2 s"),(760,"Held + motion limits"))
    for x,label in columns:svg.append(f'<text x="{x}" y="109" font-size="14" font-weight="bold">{label}</text>')
    for i,r in enumerate(s for s in summary if s["kind"]=="near"):
        values=(r["mode"],f'{r["accepted"]}/{r["n"]}',f'{r["contacted"]}/{r["n"]}',
                f'{r["held"]}/{r["n"]}',f'{r["qualified"]}/{r["n"]}')
        for (x,_),v in zip(columns,values):svg.append(f'<text x="{x}" y="{148+34*i}" font-size="14">{v}</text>')
    svg.append('<text x="25" y="303" font-size="13">Motion limits include collision acceleration. Rejected plans remain in the denominator.</text></g></svg>')
    path.write_text("\n".join(svg),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--planning-budget",type=float,default=.200)
    args=parser.parse_args()
    if not np.isfinite(args.planning_budget) or args.planning_budget<=.030:
        parser.error("Planning budget must exceed 30 ms")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    model=build_scene("ball")
    data=initial(model)
    palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    rows,traces=[],[]
    modes=("static_predict","active_predict","active_contact","active_hybrid")
    for scenario,kind,p,v in scenarios(center,model.opt.gravity):
        prepared=prepare(model,p,v,args.planning_budget)
        for mode in modes:
            row,trace=run_trial(model,p,v,scenario,kind,mode,prepared)
            rows.append(row);traces.extend(trace)
        print(scenario,kind,prepared[2]["plan_reason"],flush=True)
    summary=[]
    for kind in ("near","far","urgent"):
        for mode in modes:
            group=[r for r in rows if r["kind"]==kind and r["mode"]==mode]
            accepted=[r for r in group if r["accepted"]]
            summary.append(dict(kind=kind,mode=mode,n=len(group),accepted=len(accepted),
                contacted=sum(r["contacted"] for r in group),held=sum(r["held_success"] for r in group),
                qualified=sum(r["qualified_success"] for r in group),
                median_planning_ms=1000*median(r["planning_wall_s"] for r in group)))
    write_rows(args.output_dir/"trials.csv",rows)
    write_rows(args.output_dir/"summary.csv",summary)
    write_rows(args.output_dir/"time_series.csv",traces)
    plot(args.output_dir/"active_ball_catch_en.svg",summary)
    for r in summary:print(r)
