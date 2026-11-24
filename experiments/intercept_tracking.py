"""对比候选筛选耗时，并实际仿真机械臂的接取参考轨迹跟踪。

球体暂放远处，隔离机械臂跟踪误差。该实验尚不闭指接球。
所有补偿经受力矩限制的位置执行器施加。
"""
import argparse
from pathlib import Path
from statistics import median
import sys
import time

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.arm_reference_tracker import ArmReferenceTracker
from controllers.intercept_planner import InterceptPlanner
from experiments.contact_impulse import write_rows
from experiments.intercept_candidates import scenarios
from scene_setup.scene import build_scene


def initial(model):
    data=mujoco.MjData(model)
    ids=[model.actuator(f"act{i}").id for i in range(1,8)]
    dofs=[model.joint(f"joint{i}").dofadr[0] for i in range(1,8)]
    home=np.array([0.,.4,0.,2.,0.,-1.54,0.])
    for i,q in enumerate(home,1):
        data.joint(f"joint{i}").qpos[:]=q
    data.joint("flight").qpos[:3]=[3.,3.,3.]
    data.ctrl[ids]=home
    mujoco.mj_forward(model,data)
    data.ctrl[ids]=home+data.qfrc_bias[dofs]/model.actuator_gainprm[ids,0]
    mujoco.mj_forward(model,data)
    return data


def track(model,plan,scenario,mode):
    data=initial(model)
    controller=ArmReferenceTracker(model,mode)
    planner=InterceptPlanner(model,data)
    dt=model.opt.timestep
    peak_v=peak_a=force_fraction=0.
    margin=float("inf")
    arrival=None
    deadline=None
    trace=[]
    for k in range(round((plan.intercept_time+.1)/dt)):
        t=k*dt
        controller.command(data,plan,t)
        previous_velocity=data.qvel[controller.dofs].copy()
        mujoco.mj_step(model,data)
        peak_v=max(peak_v,float(max(abs(data.qvel[controller.dofs]))))
        # 隐式积分下，用实际积分速度的增量评价离散运动加速度。
        actual_acceleration=(data.qvel[controller.dofs]-previous_velocity)/dt
        peak_a=max(peak_a,float(max(abs(actual_acceleration))))
        force_fraction=max(force_fraction,float(max(abs(data.actuator_force[controller.ids])/
            np.max(abs(model.actuator_forcerange[controller.ids]),axis=1))))
        q=data.qpos[controller.qadr]
        margin=min(margin,float(min(np.r_[q-planner.lower,planner.upper-q])))
        # 更新步末运动学，误差与积分后的 qpos 对齐。
        mujoco.mj_kinematics(model,data)
        mujoco.mj_comPos(model,data)
        palm=data.body("hand/palm_lower")
        rot=palm.xmat.reshape(3,3)
        capture=palm.xpos+rot@planner.local_capture
        error_mm=1000*float(np.linalg.norm(capture-plan.point))
        error_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(plan.palm_rotation.T@rot)-1)/2,-1,1))))
        t_end=(k+1)*dt
        if arrival is None and error_mm<=5. and error_deg<=3.:
            arrival=t_end
        row=dict(scenario=scenario,mode=mode,time_s=t_end,position_error_mm=error_mm,orientation_error_deg=error_deg,
                 **{f"q{i+1}":q[i] for i in range(7)})
        trace.append(row)
        if abs(t_end-plan.intercept_time)<dt/2:
            deadline=row.copy()
    if deadline is None:
        raise ValueError("Missing deadline sample")
    return dict(scenario=scenario,mode=mode,intercept_time_s=plan.intercept_time,
        first_within_tolerance_s=arrival,deadline_position_error_mm=deadline["position_error_mm"],
        deadline_orientation_error_deg=deadline["orientation_error_deg"],
        reached_at_deadline=int(deadline["position_error_mm"]<=5 and deadline["orientation_error_deg"]<=3),
        peak_actual_speed_rad_s=peak_v,peak_actual_accel_rad_s2=peak_a,
        max_actuator_force_fraction=force_fraction,ctrl_clipped_steps=controller.clipped_steps,
        joint_margin_rad=margin,
        within_motion_limits=int(peak_v<=1. and peak_a<=4. and margin>=0.)),trace


def plot(path,rows):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="450">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="Arial" fill="#18364a">',
         '<text x="25" y="35" font-size="22">Arm tracking error at interception deadline</text>',
         '<text x="25" y="61" font-size="13">Blue: PD + gravity. Orange: inverse dynamics + velocity feedforward. Ball held away from robot.</text>',
         '<text x="25" y="83" font-size="13">Dashed line: 5 mm position tolerance. Both controllers use the same reference for each scenario.</text>']
    cases=sorted(set(r["scenario"] for r in rows))
    limit=max(6.,max(r["deadline_position_error_mm"] for r in rows)*1.15)
    x0,y0,w,h=70,120,880,250
    for value in (0.,5.,limit):
        y=y0+h*(1-value/limit)
        svg.append(f'<path d="M{x0} {y}h{w}" stroke="#b5c3ca" stroke-dasharray="4 3"/><text x="{x0-8}" y="{y+4}" text-anchor="end" font-size="12">{value:.1f}</text>')
    for i,case in enumerate(cases):
        x=x0+(i+.5)*w/len(cases)
        for j,(mode,color) in enumerate((("pd_gravity","#28648d"),("inverse","#bd542c"))):
            r=next(r for r in rows if r["scenario"]==case and r["mode"]==mode)
            height=h*r["deadline_position_error_mm"]/limit
            svg.append(f'<rect x="{x-17+18*j}" y="{y0+h-height}" width="15" height="{height}" fill="{color}"/>')
        svg.append(f'<text x="{x}" y="{y0+h+20}" font-size="12" text-anchor="middle">{case}</text>')
    svg.append('<text x="450" y="425" font-size="14">Scenario ID. Vertical axis: capture-center error (mm).</text></g></svg>')
    path.write_text("\n".join(svg),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    model=build_scene("ball")
    data=initial(model)
    planner=InterceptPlanner(model,data)
    center=data.body("hand/palm_lower").xpos+data.body("hand/palm_lower").xmat.reshape(3,3)@planner.local_capture
    timings,rows,traces=[],[],[]
    for scenario,kind,p,v in scenarios(center,model.opt.gravity):
        plans={}
        for mode,angles,warm in (("all_angles",(0.,30.,45.,60.,90.),False),("sphere_prior",(0.,),True)):
            start=time.perf_counter()
            plans[mode]=planner.plan(p,v,angles=angles,warm_start=warm)
            timings.append(dict(scenario=scenario,kind=kind,mode=mode,wall_s=time.perf_counter()-start,
                selected=int(plans[mode] is not None),candidate_count=len(planner.candidates)))
        plan=plans["sphere_prior"]
        if plan is not None:
            for mode in ("pd_gravity","inverse"):
                row,trace=track(model,plan,scenario,mode)
                rows.append(row);traces.extend(trace)
        print(f'Scenario {scenario}: reference={plan is not None}',flush=True)
    write_rows(args.output_dir/"planning_times.csv",timings)
    write_rows(args.output_dir/"tracking_trials.csv",rows)
    write_rows(args.output_dir/"tracking_time_series.csv",traces)
    summary=[]
    for mode in ("pd_gravity","inverse"):
        group=[r for r in rows if r["mode"]==mode]
        summary.append(dict(mode=mode,n=len(group),reached=sum(r["reached_at_deadline"] for r in group),
            within_motion_limits=sum(r["within_motion_limits"] for r in group),
            median_position_error_mm=median(r["deadline_position_error_mm"] for r in group),
            median_orientation_error_deg=median(r["deadline_orientation_error_deg"] for r in group),
            max_force_fraction=max(r["max_actuator_force_fraction"] for r in group)))
    write_rows(args.output_dir/"summary.csv",summary)
    plot(args.output_dir/"arm_tracking_en.svg",rows)
    for r in summary:
        print(r)
    for mode in ("all_angles","sphere_prior"):
        group=[r["wall_s"] for r in timings if r["mode"]==mode]
        print(mode,"median planning ms",1000*median(group),"worst ms",1000*max(group))
