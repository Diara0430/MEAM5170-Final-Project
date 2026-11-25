"""完整保持阶段的摩擦/滑动诊断，先复盘806/3与806/11；不改变控制。"""
import argparse
from pathlib import Path
from statistics import median
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.joint_aware_closure import JointAwareClosure
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from experiments.mpc_contact_diagnosis import ContactRecord
from scene_setup.scene import build_scene


def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=False)
    prior=ROOT/"results/finger_closure_validation"
    outcomes=read(prior/"trials.csv");timing=read(prior/"solver_times.csv")
    model=build_scene("ball");data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    summary=[];steps=[];points=[]
    for case,kind,p,v in scenarios(center,model.opt.gravity,seed=806):
        if case not in args.cases:continue
        for name in ("original","middle_compliant"):
            original=next(r for r in outcomes if r["seed"]=="806" and int(r["scenario"])==case and r["strategy"]==name)
            schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in timing
                      if r["seed"]=="806" and int(r["scenario"])==case and r["strategy"]==name}
            record=ContactRecord(model,case,name,806,True,point_window_ms=None)
            result,_,_=run_case(model,p,v,case,kind,806,"mpc_match",solver_schedule=schedule,step_observer=record.observe,
                hand_controller_factory=(lambda m,h:JointAwareClosure(m,h,"compliant")) if name=="middle_compliant" else None)
            for key in ("held","qualified","drift_mm","first_contact_s","peak_normal_first100ms_N"):
                assert np.isclose(result[key],float(original[key]),atol=1e-5,rtol=1e-8),(case,name,key)
            late=[r for r in record.rows if r["contact_ms"]>=300.-1e-8]
            origin=np.array([late[0][f"object_palm_{a}_mm"] for a in "xyz"])
            for row in record.rows:
                row["hold_motion_mm"]=0. if row["contact_ms"]<300.-1e-8 else float(np.linalg.norm(np.array([row[f"object_palm_{a}_mm"] for a in "xyz"])-origin))
            total=sum(sum(r[p+"_N"] for p in ("palm","finger1","finger2","finger3","thumb")) for r in late)
            crossing=next((r["contact_ms"] for r in late if r["hold_motion_mm"]>25.),None)
            selected=[pt for pt in record.points if pt["contact_ms"]>=300.]
            summary.append(dict(seed=806,scenario=case,strategy=name,held=result["held"],drift_mm=result["drift_mm"],
                first_hold_motion_over25_ms=crossing,hold_duration_ms=len(late)*2.,
                near_sliding_load_fraction=sum(r["near_sliding_load_N"] for r in late)/max(total,1e-12),
                rolling_proxy_load_fraction=sum(r["rolling_proxy_load_N"] for r in late)/max(total,1e-12),
                median_relative_angular_speed_rad_s=median(r["relative_angular_speed_rad_s"] for r in late),
                median_hold_max_slip_mm_s=median(r["max_slip_mm_s"] for r in late),
                max_hold_slip_mm_s=max(r["max_slip_mm_s"] for r in late),
                opposed_fraction=sum(r["opposed"] for r in late)/max(1,len(late)),
                finger_contact_loss_ms=sum(not r["finger_contact"] for r in late)*2.,
                max_force_balance_error_N=max(r["hand_force_balance_error_N"] for r in record.rows[:-1]),
                mu1_min=min((p["mu1"] for p in selected),default=None),mu1_max=max((p["mu1"] for p in selected),default=None)))
            steps.extend(record.rows);points.extend(record.points)
            print(summary[-1],flush=True)
    write_rows(args.output_dir/"summary.csv",summary);write_rows(args.output_dir/"time_series.csv",steps)
    write_rows(args.output_dir/"contact_points.csv",points)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--cases",type=int,nargs="+",default=[3,11])
    run(parser.parse_args())
