"""Fixed space-time speed comparisons, parked control and rolling control on rotating boxes."""
import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from controllers.deadline_arm_filter import DeadlineArmFilter
from controllers.fixed_intercept import FixedIntercept, CalibrationMPC, FixedPointMPC, PreparationFinished
from controllers.sphere_cage import SphereCageClosure
from experiments.contact_preview import trial
from experiments.fixed_speed_sweep import LABELS
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from experiments.observed_long_arc import capture_center, throws
from experiments.shape_catch import BoxObserver, box_launch, simulate
from experiments.velocity_selection import ImpactMeasurement
from scene_setup.scene import build_scene


def arm(model, observer, enabled):
    tracker = DeadlineArmFilter(model, observer, enabled, fast=True, refine=True)
    tracker.config = replace(tracker.config, speed=.985)
    return tracker


def run(model, seed, case, p, v, profile):
    sensor = partial(BoxObserver, seed=seed, case=case, profile=profile, long_track=True, use_pose=True)
    motion = box_launch(seed, case)
    common = dict(tracker_factory=arm, observer_factory=sensor, launch_rotation=motion)
    fixture = FixedIntercept()
    try:
        trial(model, seed, case, p, v, True, planner_factory=partial(CalibrationMPC, fixture=fixture,
              planning_speed_limit=.97), **common)
    except PreparationFinished:
        pass
    methods = ["parked", "rolling", *LABELS]
    shift = (seed+case) % len(methods)
    rows = []
    for method in methods[shift:]+methods[:shift]:
        if method == "parked":
            observer = sensor(model.opt.gravity, np.random.SeedSequence([seed, case, 93]))
            impact = ImpactMeasurement(model)
            tracker = arm(model, observer, True)
            def record(m,d,t,onset,close,previous,ref):
                tracker.touched = True
                impact(m,d,t,onset,close,previous,ref)
            result, _, _ = run_case(model, p, v, case, "long_arc", seed, "static_contact",
                observer=observer, launch_rotation=motion, tracker_factory=lambda m: tracker,
                hand_controller_factory=SphereCageClosure, step_observer=record)
            result.update(impact20_peak_N=impact.peak_force if result["contacted"] else None,
                          impulse20_Ns=impact.impulse if result["contacted"] else None)
        elif method == "rolling":
            result = simulate(model, seed, case, p, v, profile, "pose")
        else:
            factory = partial(FixedPointMPC, fixture=fixture, label=method, planning_speed_limit=.97)
            result = trial(model, seed, case, p, v, True, planner_factory=factory, **common)
        ratio = result.get("measured_fraction")
        requested = fixture.target_speed(method) if method in ("low","mid","high") else None
        result.update(shape="box", profile=profile, method=method, fixture_status=fixture.reason,
                      fixed_time=fixture.time, fixed_x=None if fixture.point is None else fixture.point[0],
                      fixed_y=None if fixture.point is None else fixture.point[1],
                      fixed_z=None if fixture.point is None else fixture.point[2],
                      feasible_levels=len(fixture.levels), requested_m_s=requested,
                      measured_m_s=None if ratio is None else ratio*result["contact_ball_speed"])
        rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("normal","noisy"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = build_scene("box")
    center = capture_center(initial(model))
    rows = []
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        rows.extend(run(model, args.seed, case, p, v, args.profile))
    columns = {"close_reason": None, **dict.fromkeys(k for r in rows for k in r)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir/"trials.csv", [{**columns, **r} for r in rows])


if __name__ == "__main__":
    main()
