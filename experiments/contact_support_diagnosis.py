"""重放中速盒子新增的七组保持失败，记录几何接触与实际承载的区别。"""
import argparse
import csv
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.catch_stability import ArmBuffer
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed

PARTS = ("palm", "finger1", "finger2", "finger3", "thumb")


def longest_gap(rows, field, dt):
    longest = streak = 0.
    for r in rows:
        streak = streak+dt if not r[field] else 0.
        longest = max(longest, streak)
    return round(longest, 3)


def replay(seed, mode, prior):
    timing = TimingTrial(gust_for_seed(seed), False)
    trace = []

    def observe(model, data, elapsed, closure, velocity):
        loads = np.zeros(5)
        geometry = np.zeros(5, dtype=int)
        max_slip = 0.
        center = data.body("projectile")
        palm = data.body("hand/palm_lower")
        relative = palm.xmat.reshape(3,3).T@(center.xpos-palm.xpos)
        for i, c in enumerate(data.contact):
            if closure.projectile not in (c.geom1,c.geom2):
                continue
            other = c.geom2 if c.geom1 == closure.projectile else c.geom1
            group = closure.group.get(other)
            if group is None:
                continue
            geometry[group+1] |= int(c.dist <= 0)
            if c.efc_address < 0:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(model,data,i,wrench)
            if wrench[0] < .02:
                continue
            loads[group+1] += wrench[0]
            if group >= 0:
                jo,jh = np.empty((3,model.nv)),np.empty((3,model.nv))
                mujoco.mj_jac(model,data,jo,None,c.pos,center.id)
                mujoco.mj_jac(model,data,jh,None,c.pos,model.geom_bodyid[other])
                max_slip = max(max_slip,float(np.linalg.norm(c.frame.reshape(3,3)[1:]@((jo-jh)@velocity))))
        trace.append(dict(seed=seed,mode=mode,elapsed_s=elapsed,
            finger_contact=int(any(geometry[1:])),finger_loaded=int(any(loads[1:])),
            palm_loaded=int(loads[0]>0),opposed=int(closure.holding),max_finger_slip_mm_s=1000*max_slip,
            box_palm_x_mm=1000*relative[0],box_palm_y_mm=1000*relative[1],box_palm_z_mm=1000*relative[2],
            **{p+"_N":float(loads[i]) for i,p in enumerate(PARTS)},
            **{p+"_contact":int(geometry[i]) for i,p in enumerate(PARTS)}))

    with redirect_stdout(StringIO()):
        run("early",.04,.19,True,shape="box",pitch_deg=40.,palm_x_mm=0.,arrival_speed="medium",
            timing_trial=timing,arm_buffer=ArmBuffer() if mode=="buffer" else None,
            evaluation="hold",hold_duration_s=2.,contact_observer=observe)
    for key in ("passed","failure_stage","first_contact_s","max_drift_mm"):
        if str(timing.outcome[key]) != prior[key]:
            raise ValueError(f"Replay differs: {seed}, {mode}, {key}")
    hold=[]
    for r in trace:
        r["release_s"]=round(timing.outcome["first_contact_s"]+r["elapsed_s"],6)
        if r["release_s"] >= .9-1e-9:
            hold.append(r)
    dt_ms=2.
    assert len(hold)==1000
    summary=dict(seed=seed,mode=mode,**timing.outcome,
        longest_finger_contact_gap_ms=longest_gap(hold,"finger_contact",dt_ms),
        total_finger_contact_gap_ms=dt_ms*sum(not r["finger_contact"] for r in hold),
        longest_finger_load_gap_ms=longest_gap(hold,"finger_loaded",dt_ms),
        palm_only_loaded_ms=dt_ms*sum(r["palm_loaded"] and not r["finger_loaded"] for r in hold),
        opposed_ms=dt_ms*sum(r["opposed"] for r in hold),
        **{p+"_loaded_ms":dt_ms*sum(r[p+"_N"]>0 for r in hold) for p in PARTS})
    return summary,trace


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    with (root/"results/buffer_generalization/trials.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=[r for r in csv.DictReader(f) if (r["shape"],r["speed"],r["evaluation"])==("box","medium","hold")]
    pairs={int(r["seed"]):{} for r in rows}
    for r in rows:
        pairs[int(r["seed"])][r["mode"]]=r
    args.output_dir.mkdir(parents=True,exist_ok=False)
    summary,trace=[],[]
    for seed,pair in sorted(pairs.items()):
        if pair["fixed"]["passed"]!="1" or pair["buffer"]["passed"]!="0":
            continue
        for mode in ("fixed","buffer"):
            s,t=replay(seed,mode,pair[mode]);summary.append(s);trace.extend(t)
    write_rows(args.output_dir/"summary.csv",summary)
    write_rows(args.output_dir/"time_series.csv",trace)
    for r in summary:
        print({k:r[k] for k in ("seed","mode","longest_finger_contact_gap_ms","longest_finger_load_gap_ms","palm_only_loaded_ms","opposed_ms")})
