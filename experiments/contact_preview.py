"""Isolate the effect of using the fitted ball state in the low-level torque filter."""
import argparse
from pathlib import Path

import numpy as np

from controllers.ball_observer import BallObserver
from controllers.catching_mpc import CatchingMPC
from controllers.catch_approach import CatchApproach
from controllers.contact_preview import ContactPreviewFilter
from controllers.follow_through import FollowThrough
from controllers.sphere_cage import SphereCageClosure
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from experiments.observed_long_arc import capture_center, throws, STRATEGIES
from experiments.velocity_selection import ImpactMeasurement
from scene_setup.scene import build_scene


def trial(model, seed, case, p, v, preview, tracker_factory=ContactPreviewFilter,
          planner_factory=CatchingMPC, record_frame=None):
    observer = BallObserver(model.opt.gravity, np.random.SeedSequence([seed, case, 93]))
    tracker = tracker_factory(model, observer, preview)
    impact = ImpactMeasurement(model)
    peak_20 = 0.
    capture_error = None
    precursor_ms = None
    previous_arm_v = None
    peak_flight = 0.
    def flight(model, data, info):
        nonlocal previous_arm_v, peak_flight
        state = data.qvel[tracker.dofs].copy()
        if previous_arm_v is not None and info["first_contact_s"] is None:
            peak_flight = max(peak_flight, float(np.max(np.abs(state-previous_arm_v))) / model.opt.timestep)
        previous_arm_v = state
        if record_frame is not None:
            record_frame(model, data, info, observer, tracker)

    def touch(model, data, t, contact, close, previous, ref):
        nonlocal peak_20, capture_error, precursor_ms
        if not tracker.touched:
            estimate = observer.estimate
            if estimate is not None:
                predicted, _ = estimate.at(t + model.opt.timestep)
                # This post-step residual is diagnostic; the controller receives only the tactile flag.
                capture_error = 1000*float(np.linalg.norm(predicted-data.joint("flight").qpos[:3]))
            if tracker.last_near_t is not None and abs(tracker.last_near_t - t) < 1e-9:
                precursor_ms = tracker.near_times_ms[-1]
        if not tracker.touched and hasattr(tracker, "deadline_misses"):
            tracker.onset_missed = int(tracker.last_missed)
        tracker.touched = True
        impact(model, data, t, contact, close, previous, ref)
        if t < contact + .020 - 1e-9:
            accel = (data.qvel[tracker.dofs]-previous[tracker.dofs]) / model.opt.timestep
            peak_20 = max(peak_20, float(np.max(np.abs(accel))))

    result, _, _ = run_case(model, p, v, case, "long_arc", seed, "mpc_match",
        tracker_factory=lambda m: tracker, hand_controller_factory=SphereCageClosure,
        mpc_acceleration_limit=3., follow_through=FollowThrough(), approach=CatchApproach(fraction=.30),
        observer=observer, height_offsets=STRATEGIES["seven_heights"], step_observer=touch, frame_observer=flight,
        planner_factory=planner_factory)
    times = tracker.near_times_ms
    result.update(policy="projected_contact" if preview else "robot_only",
                  peak_contact20_accel=peak_20 if tracker.touched else None,
                  impulse20_Ns=impact.impulse if tracker.touched else None,
                  impact20_peak_N=impact.peak_force if tracker.touched else None,
                  near_steps=tracker.near_steps, onset_in_preview_region=int(precursor_ms is not None),
                  peak_flight_accel=peak_flight,
                  near_overruns=sum(t>1000*model.opt.timestep for t in times),
                  near_p95_ms=float(np.percentile(times, 95)) if times else None,
                  last_near_step_ms=precursor_ms, onset_poststep_error_mm=capture_error)
    if hasattr(tracker, "deadline_misses"):
        result.update(deadline_misses=tracker.deadline_misses, command_count=tracker.command_count,
                      onset_deadline_missed=tracker.onset_missed)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case, p, v, duration, distance, offset in throws(center, model.opt.gravity, args.seed):
        for enabled in ((False, True) if case % 2 == 0 else (True, False)):
            row = trial(model, args.seed, case, p, v, enabled)
            rows.append(row)
            print(args.seed, case, enabled, row["held"], row["qualified"], flush=True)
    save_csv(args.output_dir/"trials.csv", rows)
