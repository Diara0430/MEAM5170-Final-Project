"""固定世界系来物轨迹，对照接球前掌面旋转 0/45/90 度。

先比较低速球体，再比较低速长方体。固定手指目标、无缓冲、原姿态保持两秒。
不可达或跟踪超差直接记为不可行，不运行接取。
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial,gust_for_seed


def trial(job):
    seed,shape,angle,speed=job
    timing=TimingTrial(gust_for_seed(seed),False)
    reason=""
    launch={}
    contact={}

    def observe(model,data,t,tilt,hold):
        if launch:
            return
        normal=-data.body("hand/palm_lower").xmat.reshape(3,3)[:,2]
        velocity=data.joint("flight").qvel[:3].copy()
        launch.update(palm_up_angle_deg=float(np.rad2deg(np.arccos(np.clip(normal[2],-1,1)))),
                      launch_vx=float(velocity[0]),launch_vy=float(velocity[1]),launch_vz=float(velocity[2]))

    def observe_contact(model,data,elapsed,closure,velocity):
        if contact:
            return
        dof=model.joint("flight").dofadr[0]
        incoming=velocity[dof:dof+3]
        normal=-data.body("hand/palm_lower").xmat.reshape(3,3)[:,2]
        contact["first_contact_incidence_deg"]=float(np.rad2deg(np.arccos(np.clip(-normal@incoming/max(np.linalg.norm(incoming),1e-12),-1,1))))
    try:
        with redirect_stdout(StringIO()):
            run("early",.04,.19,True,shape=shape,pitch_deg=40. if shape=="box" else 0.,
                palm_x_mm=0.,arrival_speed=speed,timing_trial=timing,
                evaluation="hold",hold_duration_s=2.,pretilt_deg=angle,frame_observer=observe,
                contact_observer=observe_contact)
    except ValueError as exc:
        if not str(exc).startswith(("Requested palm pose", "Pre-catch pose tracking")):
            raise
        reason=str(exc)
        timing.outcome=dict(passed=0,failure_stage="pose_infeasible",close_s=None,
                            first_contact_s=None,max_drift_mm=None)
    return dict(seed=seed,shape=shape,speed=speed,pretilt_deg=angle,**timing.outcome,pose_error=reason,
                **{key:launch.get(key) for key in ("palm_up_angle_deg","launch_vx","launch_vy","launch_vz")},
                first_contact_incidence_deg=contact.get("first_contact_incidence_deg"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--shapes",nargs="+",choices=("ball","box"),default=["ball","box"])
    parser.add_argument("--speeds",nargs="+",choices=("slow","medium","fast"),default=["slow"])
    parser.add_argument("--angles",nargs="+",type=float,default=[0.,45.,90.])
    parser.add_argument("--seed-start",type=int,default=568)
    parser.add_argument("--seed-count",type=int,default=24)
    args=parser.parse_args()
    if args.seed_count<1 or not np.isfinite(args.angles).all() or any(abs(a)>90 for a in args.angles):
        parser.error("Positive seed count and finite angles within +/-90 degrees required")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    jobs=[(seed,shape,angle,speed) for shape in args.shapes for seed in range(args.seed_start,args.seed_start+args.seed_count)
          for angle in args.angles for speed in args.speeds]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(trial,jobs))
    groups=defaultdict(list)
    for r in rows:
        groups[r["shape"],r["speed"],r["pretilt_deg"]].append(r)
    summary=[]
    for (shape,speed,angle),trials in groups.items():
        summary.append(dict(shape=shape,speed=speed,pretilt_deg=angle,n=len(trials),
            infeasible=sum(r["failure_stage"]=="pose_infeasible" for r in trials),
            contacted=sum(r["first_contact_s"] is not None for r in trials),
            succeeded=sum(r["passed"] for r in trials)))
    write_rows(args.output_dir/"trials.csv",rows)
    write_rows(args.output_dir/"summary.csv",summary)
    for r in summary:
        print(r)
