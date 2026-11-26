"""Measure when a camera-only prediction becomes accurate on a common catch plane."""
import argparse
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from controllers.lock_timing import agreement, crossing, uncertainty, RULES
from experiments.active_ball_catch import launch
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.lock_timing import StudyObserver, ADAPTIVE
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


def evaluate(model, height, seed, case, p, v, profile):
    data = launch(model,p,v)
    sensor = StudyObserver(model.opt.gravity,np.random.SeedSequence([seed,case,93]),
                           profile=profile,seed=seed,case=case)
    dt = model.opt.timestep
    stride = round(.020/dt)
    history = deque(maxlen=5)
    rows = []
    previous_t = None
    for step in range(round(1.6/dt)):
        t = float(data.time)
        true_pos = data.joint("flight").qpos[:3].copy()
        if previous_t is not None and previous_p[2]>=height>true_pos[2]:
            fraction = (height-previous_p[2])/(true_pos[2]-previous_p[2])
            truth_xy = previous_p[:2]+fraction*(true_pos[:2]-previous_p[:2])
            truth_s = previous_t+fraction*(t-previous_t)
            break
        if step%stride == 0:
            estimate = sensor.sample(t,true_pos)
            if estimate is not None:
                history.append(estimate)
                predicted = crossing(estimate,height)
                if predicted is not None and predicted[0]>t:
                    stamp, point, incoming = predicted
                    radius,width = uncertainty(sensor,stamp,incoming)
                    row = dict(seed=seed,scenario=case,profile=profile,frame_s=t,
                               target_s=stamp,x=point[0],y=point[1],radius_mm=1000*radius,width_ms=1000*width)
                    for name in ADAPTIVE:
                        rule = RULES[name]
                        spread = agreement(history,height,rule.frames,.020)
                        row[name] = int(spread is not None and spread[0]<=rule.spread_m
                                        and spread[1]<=rule.spread_s and
                                        (not rule.confidence or (radius<=.012 and width<=.004)))
                    rows.append(row)
        previous_t, previous_p = t,true_pos
        mujoco.mj_step(model,data)
    else:
        raise ValueError("No true descending crossing in the flight window")
    for row in rows:
        row.update(xy_error_mm=1000*float(np.linalg.norm(truth_xy-[row["x"],row["y"]])),
                   time_error_ms=1000*abs(row["target_s"]-truth_s),lead_s=truth_s-row["frame_s"])
    return rows


def metrics(group):
    xy = np.array([r["xy_error_mm"] for r in group])
    times = np.array([r["time_error_ms"] for r in group])
    return dict(n=len(group),xy_median_mm=np.median(xy),xy_p90_mm=np.percentile(xy,90),
                time_median_ms=np.median(times),time_p90_ms=np.percentile(times,90),
                both_within=sum((xy<=10.) & (times<=4.)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",required=True,choices=("normal","noisy","corrected"))
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model=build_scene("ball")
    # The evaluator needs the undisturbed flight to the plane, even if a physical
    # catch would have happened earlier. Nothing here commands the catching arm.
    ball = model.geom("projectile").id
    model.geom_contype[ball]=0
    model.geom_conaffinity[ball]=0
    center=capture_center(initial(model))
    rows, first = [], []
    for seed in range(941,946):
        for case,p,v,*_ in throws(center,model.opt.gravity,seed):
            flight=evaluate(model,center[2],seed,case,p,v,args.profile)
            rows.extend(flight)
            for name in ADAPTIVE:
                first_ok=next((r for r in flight if r[name] and r["lead_s"]>=.12),None)
                if first_ok is not None:
                    first.append(dict(rule=name,**first_ok))
    summary=[]
    for t in (.22,.26,.32,.38,.44,.50,.56,.62,.68,.74,.80):
        group=[r for r in rows if abs(r["frame_s"]-t)<1e-7]
        if len(group)!=40:
            raise ValueError("Missing paired plane forecast")
        summary.append(dict(profile=args.profile,frame_s=t,**metrics(group)))
    save_csv(args.output_dir/"all_frames.csv",rows)
    save_csv(args.output_dir/"first_gate.csv",first)
    save_csv(args.output_dir/"by_time.csv",summary)


if __name__=="__main__":
    main()
