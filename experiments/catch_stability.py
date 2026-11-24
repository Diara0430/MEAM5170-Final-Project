"""固定手腕/顺势缓冲/防滑闭指/组合的配对实验。统一使用半速倾腕。"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.catch_stability import ArmBuffer, SlipGrip
from controllers.wrist import WristProgress
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed


def trial(job):
    seed, mode, split = job
    buffer = ArmBuffer() if mode in ("buffer", "combined") else None
    grip = SlipGrip() if mode in ("grip", "combined") else None
    wrist = WristProgress("slow")
    timing = TimingTrial(gust_for_seed(seed), False)
    samples = []

    def measure(model, data, elapsed, closure, velocity):
        normal = 0.
        hand_force = np.zeros(3)
        for i, c in enumerate(data.contact):
            if closure.projectile not in (c.geom1, c.geom2) or c.efc_address < 0:
                continue
            other = c.geom1 if c.geom2 == closure.projectile else c.geom2
            if other not in closure.group:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            sign = 1 if c.geom2 == closure.projectile else -1
            hand_force += sign*c.frame.reshape(3, 3).T@wrench[:3]
            normal += max(0., wrench[0])
        if elapsed < .100-1e-10:
            samples.append((normal, hand_force*model.opt.timestep))

    run("early", .04, .19, True, shape="box", pitch_deg=40., palm_x_mm=0.,
        arrival_speed="fast", timing_trial=timing, wrist_control=wrist,
        arm_buffer=buffer, post_close_control=grip, contact_observer=measure)
    return dict(seed=seed, split=split, mode=mode, **timing.outcome,
        peak_normal_first100ms_N=max((r[0] for r in samples), default=0.),
        hand_impulse_first100ms_Ns=float(np.linalg.norm(sum((r[1] for r in samples), np.zeros(3)))),
        planned_buffer_mm=buffer.planned_mm if buffer else 0.,
        actual_buffer_pretilt_mm=buffer.actual_mm if buffer else 0.,
        buffer_infeasible=int(buffer.infeasible) if buffer else 0,
        max_buffer_command_speed_rad_s=buffer.max_command_speed_rad_s if buffer else 0.,
        grip_max_extra_rad=float(max(grip.extra)) if grip else 0.,
        grip_active_steps=grip.active_steps if grip else 0)


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["split"], r["mode"]].append(r)
    summary = []
    for (split, mode), trials in groups.items():
        base = {r["seed"]:r for r in groups[split, "fixed"]}
        summary.append(dict(split=split, mode=mode, n=len(trials), caught=sum(r["passed"] for r in trials),
            gains=sum(r["passed"] and not base[r["seed"]]["passed"] for r in trials),
            losses=sum(not r["passed"] and base[r["seed"]]["passed"] for r in trials),
            median_drift_mm=median(r["max_drift_mm"] for r in trials),
            median_peak_normal_N=median(r["peak_normal_first100ms_N"] for r in trials),
            median_impulse_Ns=median(r["hand_impulse_first100ms_Ns"] for r in trials),
            median_buffer_mm=median(r["actual_buffer_pretilt_mm"] for r in trials),
            infeasible=sum(r["buffer_infeasible"] for r in trials),
            grip_triggered=sum(r["grip_active_steps"]>0 for r in trials)))
    return summary


def plot(path, summary):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="940" height="440">',
           '<rect width="940" height="440" fill="white"/>',
           '<g font-family="Arial" fill="#18364a">',
           '<text x="25" y="35" font-size="22">Velocity-based recoil and grip feedback</text>',
           '<text x="25" y="62" font-size="13">40-degree fast box. All four controllers use the same 2-second wrist tilt and 0.5-second hold.</text>']
    for panel, split in enumerate(("selection", "validation")):
        x = 30+460*panel
        svg.append(f'<text x="{x}" y="103" font-size="18">{split.title()}</text>')
        for i, r in enumerate(s for s in summary if s["split"] == split):
            y = 145+64*i
            caught, n = int(r["caught"]), int(r["n"])
            width = 180*caught/n
            svg.append(f'<text x="{x}" y="{y}" font-size="14">{r["mode"]}</text>')
            svg.append(f'<rect x="{x+85}" y="{y-16}" width="{width}" height="19" fill="#30718b"/>')
            svg.append(f'<text x="{x+285}" y="{y}" font-size="14">{caught}/{n}</text>')
            svg.append(f'<text x="{x+85}" y="{y+23}" font-size="12">Median drift {float(r["median_drift_mm"]):.2f} mm | Gains {r["gains"]}, losses {r["losses"]}</text>')
    svg.append('<text x="30" y="425" font-size="13">Fixed = no recoil, fixed finger targets. Buffer = recoil only. Grip = finger feedback only.</text></g></svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = [(seed, mode, split) for split, seeds in
            (("selection", range(480,496)), ("validation", range(496,544)))
            for seed in seeds for mode in ("fixed", "buffer", "grip", "combined")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(trial, jobs))
    summary = summarize(rows)
    write_rows(args.output_dir/"trials.csv", rows)
    write_rows(args.output_dir/"summary.csv", summary)
    plot(args.output_dir/"catch_stability_en.svg", summary)
    for row in summary:
        print(row)
