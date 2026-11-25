"""球体主动接取的目标筛选实验，保存方案、逐候选拒绝原因和参考轨迹。

python experiments/intercept_candidates.py --output-dir results/intercept_candidates
生成的五次轨迹为后续控制器参考，尚未执行闭环接取。
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.intercept_planner import InterceptPlanner
from experiments.contact_impulse import write_rows
from scene_setup.scene import build_scene


def scenarios(center,gravity,seed=800):
    rng=np.random.default_rng(seed)
    for i in range(20):
        kind="near" if i<12 else "far" if i<16 else "urgent"
        delay=rng.uniform(.45,.65) if kind!="urgent" else .08
        target=center+rng.uniform([-.04,-.04,-.02],[.04,.04,.02])
        if kind=="far":
            target[0]+=2.
        start=target+rng.uniform([-.15,-.15,.15],[.15,.15,.30])
        if kind=="urgent":
            start=target+np.array([.02,-.02,.04])
        velocity=(target-start-.5*gravity*delay**2)/delay
        yield i,kind,start,velocity


def plot(path,rows):
    height=150+28*len(rows)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="Arial" fill="#18364a">',
         '<text x="25" y="36" font-size="22">Interception candidate screening</text>',
         '<text x="25" y="62" font-size="13">Ballistic sphere prediction. Joint limits: 1 rad/s and 4 rad/s^2. Sampled open-hand path collision check.</text>',
         '<text x="25" y="84" font-size="13">A selected reference still requires dynamic tracking and grasp validation. No selection applies to this candidate set.</text>']
    columns=((25,"Scenario"),(150,"Type"),(270,"Reference"),(430,"Time (s)"),(560,"Palm (deg)"),(700,"Peak speed"),(850,"Peak accel."))
    for x,label in columns:
        svg.append(f'<text x="{x}" y="117" font-size="13" font-weight="bold">{label}</text>')
    for i,r in enumerate(rows):
        y=143+28*i
        values=(str(r["scenario"]),r["kind"],"Selected" if r["selected"] else "No feasible candidate",
                "-" if r["time_s"] is None else f'{r["time_s"]:.2f}',
                "-" if r["angle_deg"] is None else f'{r["angle_deg"]:g}',
                "-" if r["peak_speed"] is None else f'{r["peak_speed"]:.2f}',
                "-" if r["peak_accel"] is None else f'{r["peak_accel"]:.2f}')
        for (x,_),value in zip(columns,values):
            svg.append(f'<text x="{x}" y="{y}" font-size="13">{value}</text>')
    svg.append('</g></svg>')
    path.write_text("\n".join(svg),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    model=build_scene("ball")
    data=mujoco.MjData(model)
    home=[0.,.4,0.,2.,0.,-1.54,0.]
    dofs=[]
    for i,q in enumerate(home,1):
        data.joint(f"joint{i}").qpos[:]=q
        data.actuator(f"act{i}").ctrl[:]=q
        dofs.append(model.joint(f"joint{i}").dofadr[0])
    data.joint("flight").qpos[:3]=[3.,3.,1.]
    mujoco.mj_forward(model,data)
    data.qfrc_applied[dofs]=data.qfrc_bias[dofs]
    mujoco.mj_forward(model,data)
    planner=InterceptPlanner(model,data)
    center=data.body("hand/palm_lower").xpos+data.body("hand/palm_lower").xmat.reshape(3,3)@planner.local_capture
    original=data.qpos.copy()
    rows,candidates,trajectory=[],[],[]
    for scenario,kind,p,v in scenarios(center,model.opt.gravity):
        start_time=time.perf_counter()
        plan=planner.plan(p,v)
        duration=time.perf_counter()-start_time
        assert np.array_equal(data.qpos,original),"Planner modified live state"
        feasible=[r for r in planner.candidates if r["reason"]=="feasible_reference"]
        selected=min(feasible,key=lambda r:r["score"]) if feasible else None
        rows.append(dict(scenario=scenario,kind=kind,selected=int(plan is not None),
            feasible_count=len(feasible),planning_wall_s=duration,
            initial_x=p[0],initial_y=p[1],initial_z=p[2],vx=v[0],vy=v[1],vz=v[2],
            time_s=selected["delay_s"] if selected else None,
            angle_deg=selected["angle_deg"] if selected else None,
            x_m=selected["x_m"] if selected else None,y_m=selected["y_m"] if selected else None,z_m=selected["z_m"] if selected else None,
            peak_speed=selected["peak_joint_speed_rad_s"] if selected else None,
            peak_accel=selected["peak_joint_accel_rad_s2"] if selected else None,
            **{f"q{i}":selected[f"q{i}"] if selected else None for i in range(1,8)}))
        candidates.extend(dict(scenario=scenario,kind=kind,**r) for r in planner.candidates)
        if plan is not None:
            q0,v0,a0=plan.reference(0.)
            qe,ve,ae=plan.reference(plan.intercept_time)
            assert np.allclose(q0,data.qpos[planner.qadr]) and np.allclose(v0,data.qvel[planner.dofs])
            assert np.allclose(qe,plan.q_goal) and np.linalg.norm(ve)<1e-8 and np.linalg.norm(ae)<1e-8
            for t in np.linspace(0,plan.intercept_time,41):
                q,velocity,accel=plan.reference(t)
                trajectory.append(dict(scenario=scenario,time_s=t,
                    **{f"q{i+1}":q[i] for i in range(7)},
                    **{f"v{i+1}":velocity[i] for i in range(7)},
                    **{f"a{i+1}":accel[i] for i in range(7)}))
        print(f'{scenario:02} {kind}: {len(feasible)} feasible candidates ({duration:.2f}s)',flush=True)
    write_rows(args.output_dir/"scenarios.csv",rows)
    write_rows(args.output_dir/"candidates.csv",candidates)
    if trajectory:
        write_rows(args.output_dir/"reference_trajectories.csv",trajectory)
    counts=Counter(r["reason"] for r in candidates)
    write_rows(args.output_dir/"rejection_summary.csv",[dict(reason=k,count=v) for k,v in sorted(counts.items())])
    plot(args.output_dir/"intercept_candidates_en.svg",rows)
    for kind in ("near","far","urgent"):
        group=[r for r in rows if r["kind"]==kind]
        print(kind,sum(r["selected"] for r in group),"/",len(group))
