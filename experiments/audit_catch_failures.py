"""回放96次已记录试验，区分保持失败、早期支撑缺失及接触前/后的实际加速度超限。"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.contact_admittance import ContactAdmittance
from experiments.buffer_ablation import VARIANTS,write_rows
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results/contact_buffer_validation")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    trials=read(args.results_dir/"trials.csv");timing=read(args.results_dir/"solver_times.csv")
    model=build_scene("ball");rest=initial(model);palm=rest.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    dofs=[model.joint(f"joint{i}").dofadr[0] for i in range(1,8)]
    launches={(seed,case):(kind,p,v) for seed in sorted(set(int(r["seed"]) for r in trials))
              for case,kind,p,v in scenarios(center,model.opt.gravity,seed=seed)}
    rows=[]
    for prior in trials:
        seed,case,name=int(prior["seed"]),int(prior["scenario"]),prior["strategy"]
        kind,p,v=launches[seed,case]
        schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in timing
                  if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]==name}
        recorded=[]
        def frame(m,d,info):recorded.append((float(d.time),d.qvel[dofs].copy()))
        _,buffered,duration=next(x for x in VARIANTS if x[0]==name)
        result,_,_=run_case(model,p,v,case,kind,seed,"mpc_match",solver_schedule=schedule,
                           buffer_factory=ContactAdmittance if buffered else None,closure_duration=duration,frame_observer=frame)
        for key in ("held","qualified","first_contact_s","drift_mm","peak_normal_first100ms_N","peak_accel"):
            assert np.isclose(float(result[key]),float(prior[key]),atol=1e-5,rtol=1e-8),(seed,case,name,key)
        times=np.array([r[0] for r in recorded]);vel=np.array([r[1] for r in recorded])
        acc=np.max(abs(np.diff(vel,axis=0)/np.diff(times)[:,None]),axis=1);times=times[1:]
        first=result["first_contact_s"]
        pre=times<=first+1e-10;post=~pre
        peak=int(np.argmax(acc))
        category="held" if result["held"] else "no_opposed_support_first300ms" if not int(prior["support_formed"]) else "early_support_but_hold_failed"
        rows.append(dict(seed=seed,scenario=case,strategy=name,held=result["held"],qualified=result["qualified"],
            failure_category=category,first_contact_s=first,peak_accel_precontact=float(max(acc[pre],default=0.)),
            peak_accel_postcontact=float(max(acc[post],default=0.)),peak_accel_time_s=float(times[peak]),
            global_peak_phase="before_contact" if pre[peak] else "contact_or_after",
            historical_match=1))
        if len(rows)%4==0:print(f"Verified launch {seed}/{case} ({len(rows)} trials)",flush=True)
    write_rows(args.output_dir/"failure_and_acceleration_audit.csv",rows)
    groups=[]
    for seed in sorted(set(r["seed"] for r in rows)):
        for name,_,_ in VARIANTS:
            group=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            groups.append(dict(seed=seed,strategy=name,n=len(group),held=sum(r["held"] for r in group),
                fail_without_early_support=sum(r["failure_category"]=="no_opposed_support_first300ms" for r in group),
                fail_after_early_support=sum(r["failure_category"]=="early_support_but_hold_failed" for r in group),
                precontact_accel_over4=sum(r["peak_accel_precontact"]>4 for r in group),
                postcontact_accel_over4=sum(r["peak_accel_postcontact"]>4 for r in group),
                global_peak_before_contact=sum(r["global_peak_phase"]=="before_contact" for r in group)))
    write_rows(args.output_dir/"summary.csv",groups)
    print("TOTAL",dict(n=len(rows),held=sum(r["held"] for r in rows),
        pre_over4=sum(r["peak_accel_precontact"]>4 for r in rows),
        post_over4=sum(r["peak_accel_postcontact"]>4 for r in rows),
        global_peak_before=sum(r["global_peak_phase"]=="before_contact" for r in rows)),flush=True)


if __name__=="__main__":main()
