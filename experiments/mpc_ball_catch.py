"""真实球体的滚动时域接取对照。20ms求解/通信等待期间继续执行旧参考。

固定掌面预测闭指、固定掌面接触闭指、MPC零速度目标、MPC部分速度匹配。
种子800为历史基准，801为本轮独立验证；接触后限幅制动并保持两秒。
"""
import argparse
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.ball_observer import crossing_durations
from controllers.arm_reference_tracker import ArmReferenceTracker
from controllers.catching_mpc import CatchingMPC,AccelerationReference,braking_reference
from controllers.catch_approach import EntryTrigger
from controllers.hand import LeapHand
from experiments.active_ball_catch import launch,hold_plan
from experiments.contact_impulse import write_rows
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.static_grasp import contacts_with
from scene_setup.scene import build_scene


def run_case(model,p,v,scenario,kind,seed,mode,step_observer=None,solver_schedule=None,
             buffer_factory=None,closure_duration=.14,frame_observer=None,hand_controller_factory=None,
             allow_schedule_extension=False,tracker_factory=None,mpc_acceleration_limit=4.,follow_through=None,approach=None,
             observer=None, height_offsets=(0.,), velocity_selection=None):
    if not np.isfinite(closure_duration) or closure_duration<=0:
        raise ValueError("Closure duration must be positive")
    if buffer_factory is not None and follow_through is not None:
        raise ValueError("Choose one post-contact reference generator")
    if approach is not None and mode != "mpc_match":
        raise ValueError("Approach policies require active velocity-matched MPC")
    if observer is not None and (mode == "static_predict" or (approach and approach.entry)):
        raise ValueError("Observed trials use tactile closure")
    data=launch(model,p,v)
    tracker=ArmReferenceTracker(model) if tracker_factory is None else tracker_factory(model)
    hand=LeapHand(model)
    hand_controller=None if hand_controller_factory is None else hand_controller_factory(model,hand)
    active=mode.startswith("mpc")
    mpc=CatchingMPC(model,data,.2 if mode=="mpc_match" else 0.,acceleration_limit=mpc_acceleration_limit,follow_through=follow_through,approach=approach,
                    velocity_selection=velocity_selection)
    entry = EntryTrigger(model, mpc.checker) if approach is not None and approach.entry else None
    last_applied = None
    close_reason = None
    measured_fraction = actual_tilt = contact_ball_speed = None
    reference=hold_plan(model,data)
    home_palm = data.body("hand/palm_lower")
    home_height = (home_palm.xpos + home_palm.xmat.reshape(3, 3) @ mpc.checker.local_capture)[2]
    nominal = None
    if observer is None:
        roots = np.roots([.5 * model.opt.gravity[2], v[2], p[2] - home_height])
        times = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0
                 and v[2] + model.opt.gravity[2] * r.real < 0]
        nominal = min(times) if times else .5
    # Observed trials use a fixed free-flight timeout, independent of launch state.
    flight_deadline = nominal + 2.3 if observer is None else 2.5
    robot_view = mujoco.MjData(model) if observer is not None else None
    first_applied = first_application_time = None
    projectile=model.geom("projectile").id;floor=model.geom("floor").id
    palm=model.body("hand/palm_lower").id
    hand_geoms={i for i,b in enumerate(model.geom_bodyid) if (model.body(b).name or "").startswith("hand/")}
    fingers={i for i in hand_geoms if model.geom_bodyid[i]!=palm}
    pending=None
    accepted=not active
    braking=False
    buffering_started=False
    first_contact=close=origin=None
    normal_speed=None
    grounded=False;continuous=True
    drift=peak_speed=peak_accel=peak_force=force_fraction=0.
    impulse=0.
    margin=float("inf")
    overrun=applied=0
    solves=[];trace=[]
    dt=model.opt.timestep
    tick=round(mpc.period/dt)
    if frame_observer is not None:
        frame_observer(model,data,dict(first_contact_s=None,close_s=None,drift_mm=0.,
            floor=False,finger_contact=False,plans_applied=0))
    stop_fallbacks = 0
    def stop_reference(t):
        nonlocal stop_fallbacks
        if follow_through is not None:
            q = data.qpos[tracker.qadr].copy()
            v = data.qvel[tracker.dofs].copy()
            if follow_through.fits(q, v, mpc.checker.lower, mpc.checker.upper):
                return follow_through.reference_from_state(t, q, v)
            stop_fallbacks += 1
        return braking_reference(model, data, t)

    while data.time < (first_contact + 2.3 if first_contact is not None else flight_deadline) - dt/2:
        t=float(data.time)
        camera_tick = round(t/dt) % tick == 0
        control_data = data
        if observer is not None and first_contact is None:
            if camera_tick:
                estimate = observer.sample(t, data.joint("flight").qpos[:3])
                if estimate is not None:
                    crossing = estimate.descending_crossing(home_height)
                    if crossing is not None:
                        nominal = crossing
            # Neither the pre-contact tracker nor its one-step dynamics probe may
            # anticipate impact by inspecting the simulator projectile.
            mujoco.mj_copyData(robot_view, model, data)
            robot_view.joint("flight").qpos[:] = [4., 4., 4., 1., 0., 0., 0.]
            robot_view.joint("flight").qvel[:] = 0.
            mujoco.mj_forward(model, robot_view)
            control_data = robot_view
        if active:
            if first_contact is not None and (not braking or (buffer_factory is not None and not buffering_started)):
                reference=(stop_reference(t) if buffer_factory is None else
                           buffer_factory(model,data,t,velocity))
                pending=None;braking=True
                buffering_started=buffer_factory is not None
            if not braking and isinstance(reference,AccelerationReference) and t>=reference.intercept_time-1e-10:
                reference=stop_reference(t);pending=None;braking=True
            if pending is not None and t>=pending.observation_time-1e-10:
                reference=pending;pending=None;accepted=True;applied+=1
                last_applied = reference
                if first_applied is None:
                    first_applied, first_application_time = reference, t
            if not braking and first_contact is None and camera_tick and nominal is not None and t < nominal + .25:
                if observer is None:
                    position, velocity = data.joint("flight").qpos[:3], data.joint("flight").qvel[:3]
                    durations = None
                else:
                    position, velocity = observer.estimate.at(t)
                    durations = crossing_durations(observer.estimate, home_height, height_offsets, t, mpc.period)
                plan,info=mpc.solve(control_data,position,velocity,t,reference,durations=durations)
                measured_overrun=info["wall_s"]>mpc.period
                # 诊断回放可使用已记录的决策，隔离记录开销对控制时序的影响。
                step=round(t/dt)
                extending=solver_schedule is not None and step not in solver_schedule
                if extending and not allow_schedule_extension:raise ValueError("Recorded solver schedule ended before contact")
                missed=measured_overrun if solver_schedule is None or extending else solver_schedule[step]
                overrun+=int(missed)
                solves.append(dict(seed=seed,scenario=scenario,kind=kind,mode=mode,time_s=t,
                    wall_ms=1000*info["wall_s"],feasible=info["feasible_candidates"],overrun=int(missed),
                    measured_overrun=int(measured_overrun),decision_source="measured_extension" if extending else "measured" if solver_schedule is None else "recorded"))
                if not missed and plan is not None:
                    pending=plan
        if close is None:
            trigger=t>=nominal-.04 if mode=="static_predict" else first_contact is not None
            reason = None
            if entry is not None and accepted:
                reason = entry.decision(model, data, t, first_contact)
                trigger = reason is not None
            if trigger and accepted:
                close=t
                close_reason = reason or ("predicted_time" if mode == "static_predict" else "contact")
        if close is not None:
            if hand_controller is None:
                hand.command(data,hand.open_pose,hand.closed,(t-close)/closure_duration)
            else:
                hand_controller.command(model,data,t-close)
        elif hand_controller is not None and nominal is not None:
            hand_controller.prepare(model,control_data,t,nominal)
        if hasattr(reference,"update"):
            reference.update(model,data,t)
        tracker.command(control_data,reference,t)
        if control_data is not data:
            data.ctrl[:] = control_data.ctrl
        velocity=data.qvel.copy()
        mujoco.mj_step(model,data)
        touching=contacts_with(data,projectile)
        onset=first_contact is None and bool(touching&hand_geoms)
        if onset:
            first_contact=t
            if approach is not None:
                palm_state = data.body(palm)
                rotation = palm_state.xmat.reshape(3, 3)
                center = palm_state.xpos + rotation @ mpc.checker.local_capture
                jac = np.empty((3, model.nv))
                mujoco.mj_jac(model, data, jac, None, center, palm)
                flight_dof = model.joint("flight").dofadr[0]
                ball_velocity = velocity[flight_dof:flight_dof + 3]
                contact_ball_speed = float(np.linalg.norm(ball_velocity))
                if contact_ball_speed > 1e-9:
                    measured_fraction = float((jac @ velocity) @ ball_velocity / contact_ball_speed**2)
                actual_tilt = float(np.rad2deg(np.arccos(np.clip(rotation[:, 2] @ mpc.orientation[:, 2], -1., 1.))))
        if hand_controller is not None and (observer is None or first_contact is not None):
            # 已解出的真实接触力只影响下一物理步，避免使用未来接触信息。
            hand_controller.observe(model,data,velocity=velocity)
        if first_contact is not None and t<first_contact+.1-1e-10:
            normal_load=0.
            for i,c in enumerate(data.contact):
                if projectile not in (c.geom1,c.geom2):continue
                other=c.geom2 if c.geom1==projectile else c.geom1
                if other not in hand_geoms:continue
                if c.efc_address>=0:
                    wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
                    normal_load+=max(0.,wrench[0])
                if onset and c.dist<=0:
                    jo,jh=np.empty((3,model.nv)),np.empty((3,model.nv))
                    mujoco.mj_jac(model,data,jo,None,c.pos,model.body("projectile").id)
                    mujoco.mj_jac(model,data,jh,None,c.pos,model.geom_bodyid[other])
                    speed=abs(float(c.frame[:3]@((jo-jh)@velocity)))
                    normal_speed=max(normal_speed or 0.,speed)
            peak_force=max(peak_force,normal_load);impulse+=normal_load*dt
        grounded |= floor in touching
        peak_speed=max(peak_speed,float(max(abs(data.qvel[tracker.dofs]))))
        peak_accel=max(peak_accel,float(max(abs((data.qvel[tracker.dofs]-velocity[tracker.dofs])/dt))))
        force_fraction=max(force_fraction,float(max(abs(data.actuator_force[tracker.ids])/
            np.max(abs(model.actuator_forcerange[tracker.ids]),axis=1))))
        q=data.qpos[tracker.qadr]
        margin=min(margin,float(min(np.r_[q-mpc.checker.lower,mpc.checker.upper-q])))
        if step_observer is not None and first_contact is not None:
            step_observer(model,data,t,first_contact,close,velocity,reference)
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        palm_state=data.body(palm)
        relative=palm_state.xmat.reshape(3,3).T@(data.body("projectile").xpos-palm_state.xpos)
        anchor = first_contact if first_contact is not None else nominal
        hold_start = anchor + .3 if anchor is not None else float("inf")
        if t+dt>=hold_start-1e-10:
            if origin is None:origin=relative.copy()
            drift=max(drift,float(np.linalg.norm(relative-origin)))
            continuous &= bool(touching&fingers)
        if round((t+dt)/dt)%5==0:
            trace.append(dict(seed=seed,scenario=scenario,mode=mode,time_s=t+dt,accepted=int(accepted),
                finger_contact=int(bool(touching&fingers)),floor=int(floor in touching),
                object_palm_x_mm=1000*relative[0],object_palm_y_mm=1000*relative[1],object_palm_z_mm=1000*relative[2],drift_mm=1000*drift))
        # 完整状态记录在物理步和运动学更新后执行，观察器不参与控制。
        if frame_observer is not None:
            frame_observer(model,data,dict(first_contact_s=first_contact,close_s=close,drift_mm=1000*drift,
                floor=bool(floor in touching),finger_contact=bool(touching&fingers),plans_applied=applied,
                applied_reference=last_applied))
        if grounded:break
    physically_held = first_contact is not None and not grounded and continuous and drift <= .025
    held = accepted and physically_held
    qualified=held and peak_speed<=1. and peak_accel<=4. and margin>=0 and force_fraction<=1.000001
    diagnostics = {}
    if approach is not None:
        diagnostics = dict(close_reason=close_reason,
            close_minus_contact_ms=None if close is None or first_contact is None else 1000*(close-first_contact),
            last_target_fraction=None if last_applied is None else last_applied.match_fraction,
            last_predicted_speed=None if last_applied is None else last_applied.incoming_speed,
            measured_fraction=measured_fraction, actual_tilt_deg=actual_tilt, contact_ball_speed=contact_ball_speed)
    if observer is not None:
        diagnostics.update(physical_hold=int(physically_held), first_application_s=first_application_time,
            first_selected_s=None if first_applied is None else first_applied.intercept_time,
            first_planned_height_offset_m=None if first_applied is None else first_applied.point[2] - home_height)
    if velocity_selection is not None:
        diagnostics.update(
            predicted_match=None if last_applied is None else last_applied.predicted_match,
            predicted_normal_speed=None if last_applied is None else last_applied.predicted_normal_speed,
            planned_stop_margin_rad=None if last_applied is None else last_applied.stop_margin)
    return dict(**diagnostics,seed=seed,scenario=scenario,kind=kind,mode=mode,accepted=int(accepted),contacted=int(first_contact is not None),
        held=int(held),qualified=int(qualified),first_contact_s=first_contact,close_s=close,
        first_normal_relative_speed_m_s=normal_speed,peak_normal_first100ms_N=peak_force,normal_impulse_first100ms_Ns=impulse,
        drift_mm=1000*drift,peak_speed=peak_speed,peak_accel=peak_accel,force_fraction=force_fraction,
        joint_margin_rad=margin,stop_fallbacks=stop_fallbacks,qp_calls=len(solves),qp_overruns=overrun,plans_applied=applied),solves,trace


def summarize(rows,solves):
    summary=[]
    for seed in sorted(set(r["seed"] for r in rows)):
        for kind in ("near","far","urgent"):
            for mode in ("static_predict","static_contact","mpc_stop","mpc_match"):
                group=[r for r in rows if (r["seed"],r["kind"],r["mode"])==(seed,kind,mode)]
                speed=[r["first_normal_relative_speed_m_s"] for r in group if r["first_normal_relative_speed_m_s"] is not None]
                force=[r["peak_normal_first100ms_N"] for r in group if r["contacted"]]
                calls=[r["wall_ms"] for r in solves if (r["seed"],r["kind"],r["mode"])==(seed,kind,mode)]
                summary.append(dict(seed=seed,kind=kind,mode=mode,n=len(group),accepted=sum(r["accepted"] for r in group),
                    contacted=sum(r["contacted"] for r in group),held=sum(r["held"] for r in group),qualified=sum(r["qualified"] for r in group),
                    median_normal_speed=median(speed) if speed else None,median_peak_normal_N=median(force) if force else None,
                    median_qp_ms=median(calls) if calls else None,qp_overruns=sum(r["qp_overruns"] for r in group)))
    return summary


def plot(path,summary):
    seed=max(int(r["seed"]) for r in summary)
    near=[r for r in summary if r["kind"]=="near" and int(r["seed"])==seed]
    height=170+40*len(near)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="Arial" fill="#18364a">',
         f'<text x="25" y="35" font-size="22">Course-based QP-MPC: ball trajectory seed {seed}</text>',
         '<text x="25" y="62" font-size="13">20 ms computation delay. Contact-triggered closure for both MPC variants. Matching target: 20% of ball velocity.</text>']
    columns=((25,"Strategy"),(230,"Applied"),(350,"Held 2 s"),(480,"Qualified"),(600,"Normal speed (m/s)"),(820,"Peak force (N)"),(980,"QP (ms)"))
    for x,label in columns:svg.append(f'<text x="{x}" y="105" font-size="13" font-weight="bold">{label}</text>')
    for i,r in enumerate(near):
        speed="-" if r["median_normal_speed"] is None else f'{r["median_normal_speed"]:.2f}'
        force="-" if r["median_peak_normal_N"] is None else f'{r["median_peak_normal_N"]:.1f}'
        solve="-" if r["median_qp_ms"] is None else f'{r["median_qp_ms"]:.2f}'
        values=(r["mode"],f'{r["accepted"]}/{r["n"]}',f'{r["held"]}/{r["n"]}',f'{r["qualified"]}/{r["n"]}',speed,force,solve)
        for (x,_),value in zip(columns,values):svg.append(f'<text x="{x}" y="{144+40*i}" font-size="14">{value}</text>')
    svg.append(f'<text x="25" y="{height-20}" font-size="13">Qualified includes actual joint limits and collision acceleration. Force values are medians of first-100-ms normal-load peaks.</text></g></svg>')
    path.write_text("\n".join(svg),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[800,801])
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    model=build_scene("ball");data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    rows,solves,traces=[],[],[]
    for seed in args.seeds:
        for scenario,kind,p,v in scenarios(center,model.opt.gravity,seed=seed):
            for mode in ("static_predict","static_contact","mpc_stop","mpc_match"):
                row,calls,trace=run_case(model,p,v,scenario,kind,seed,mode)
                rows.append(row);solves.extend(calls);traces.extend(trace)
            print(seed,scenario,[(r["mode"],r["held"]) for r in rows[-4:]],flush=True)
    summary=summarize(rows,solves)
    write_rows(args.output_dir/"trials.csv",rows)
    write_rows(args.output_dir/"solver_times.csv",solves)
    write_rows(args.output_dir/"time_series.csv",traces)
    write_rows(args.output_dir/"summary.csv",summary)
    plot(args.output_dir/"catching_mpc_en.svg",summary)
    for r in summary:
        if r["kind"]=="near":print(r)
