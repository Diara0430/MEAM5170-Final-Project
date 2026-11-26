"""Paired long throws with delayed position observations and active interception."""
import argparse
import json
from pathlib import Path

import numpy as np

from controllers.ball_observer import BallObserver
from controllers.catch_approach import CatchApproach
from controllers.discrete_arm_filter import DiscreteArmFilter
from controllers.follow_through import FollowThrough
from controllers.sphere_cage import SphereCageClosure
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene


STRATEGIES = {"parked": (), "single_height": (0.,), "select_height": (.065, 0., -.065),
              "seven_heights": tuple(np.linspace(.065, -.065, 7))}
OFFSET = np.array([0., -.055, -.08])


def throws(center, gravity, seed):
    rng = np.random.default_rng(seed)
    for case in range(8):
        high_arc = case % 2 == 1
        wide_target = case % 4 >= 2
        duration = rng.uniform(1.05, 1.18) if high_arc else rng.uniform(.90, 1.04)
        distance = rng.uniform(.88, 1.15) if high_arc else rng.uniform(.65, .88)
        offset = rng.uniform(.065, .105) if wide_target else rng.uniform(.018, .046)
        azimuth = rng.uniform(-np.pi, np.pi)
        target = center + offset * np.array([np.cos(azimuth), np.sin(azimuth), 0.])
        # Launch from outside the arm, with lateral variation in the incoming direction.
        direction = rng.uniform(-1., 1.) + np.arctan2(center[1], center[0])
        start = target + np.array([distance*np.cos(direction), distance*np.sin(direction),
                                   rng.uniform(.05, .20)])
        velocity = (target - start - .5 * gravity * duration**2) / duration
        yield case, start, velocity, duration, distance, offset


def capture_center(data):
    palm = data.body("hand/palm_lower")
    return palm.xpos + palm.xmat.reshape(3, 3) @ OFFSET


def run_trial(model, center, seed, case, start, velocity, strategy):
    sensor = BallObserver(model.opt.gravity, np.random.SeedSequence([seed, case, 93]))
    path = []
    errors = {}
    first_touch_travel = None
    touch_target_error = touch_lead = None
    closest = np.inf
    preplan_travel = 0.
    active_travel = 0.
    last_free = None
    previous_arm_v = None
    arm_dofs = [model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
    stage_peaks = dict(flight=0., contact_20ms=0., later=0.)
    peak_time = None
    dt = model.opt.timestep
    camera_ticks = round(.020 / dt)

    def record(m, d, info):
        nonlocal first_touch_travel, closest, preplan_travel, active_travel, last_free
        nonlocal previous_arm_v, peak_time, touch_target_error, touch_lead
        now = float(d.time)
        center_now = capture_center(d)
        ball = d.joint("flight").qpos[:3].copy()
        shift = float(np.linalg.norm(center_now - center))
        contact = info["first_contact_s"]
        arm_v = d.qvel[arm_dofs].copy()
        if previous_arm_v is not None:
            acceleration = float(np.max(np.abs(arm_v - previous_arm_v)) / dt)
            stage = "flight" if contact is None else "contact_20ms" if now-dt < contact+.020-1e-9 else "later"
            if acceleration > max(stage_peaks.values()):
                peak_time = now-dt
            stage_peaks[stage] = max(stage_peaks[stage], acceleration)
        previous_arm_v = arm_v
        if contact is None:
            active_travel = max(active_travel, shift)
            if info["plans_applied"] == 0:
                preplan_travel = max(preplan_travel, shift)
            closest = min(closest, float(np.linalg.norm(ball - center_now)))
            last_free = shift
            if sensor.first_estimate is not None:
                for horizon in (.2, .4):
                    target = sensor.first_estimate.time + horizon
                    if abs(target-now) < dt/2:
                        predicted, _ = sensor.first_estimate.at(now)
                        errors[horizon] = float(np.linalg.norm(ball - predicted))
        elif first_touch_travel is None:
            first_touch_travel = last_free
            plan = info.get("applied_reference")
            if plan is not None:
                touch_target_error = 1000 * float(np.linalg.norm(center_now - plan.point))
                touch_lead = 1000 * (plan.intercept_time - contact)
        stride = 5*camera_ticks if contact is not None and now > contact + .12 else camera_ticks
        if round(now/dt) % stride:
            return
        ref = info.get("applied_reference")
        last = sensor.frames[-1] if sensor.frames else None
        fitted = sensor.estimate if contact is None else None
        path.append(dict(time_s=now, ball=ball.tolist(), hand=center_now.tolist(),
                         qpos=d.qpos.round(8).tolist(),
                         forecast=None if fitted is None else [fitted.time, fitted.position.tolist(), fitted.velocity.tolist()],
                         measurement_time=None if last is None else last[0],
                         measurement=None if last is None else last[1].tolist(),
                         intercept=None if ref is None else ref.point.tolist(),
                         intercept_s=None if ref is None else ref.intercept_time))

    active = strategy != "parked"
    row, calls, _ = run_case(model, start, velocity, case, "long_arc", seed,
        "mpc_match" if active else "static_contact",
        tracker_factory=DiscreteArmFilter, hand_controller_factory=SphereCageClosure,
        mpc_acceleration_limit=3., follow_through=FollowThrough() if active else None,
        approach=CatchApproach(fraction=.30) if active else None,
        observer=sensor, height_offsets=STRATEGIES[strategy], frame_observer=record)
    estimate = sensor.first_estimate
    if estimate is None or abs(estimate.time - .16) > dt/2:
        raise AssertionError("First estimate requires eight delivered frames at 50 Hz")
    if row["first_application_s"] is not None and row["first_application_s"] < estimate.time + .02 - 1e-8:
        raise AssertionError("Arm plan applied before measurement and solver delay")
    row.update(strategy=strategy, first_estimate_s=estimate.time, peak_precontact_travel_mm=1000*active_travel,
               palm_to_selected_at_touch_mm=touch_target_error, planned_minus_touch_ms=touch_lead,
               peak_flight_accel=stage_peaks["flight"], peak_contact20_accel=stage_peaks["contact_20ms"],
               peak_later_accel=stage_peaks["later"], peak_accel_time_s=peak_time,
               before_plan_travel_mm=1000*preplan_travel, closest_free_mm=1000*closest,
               precontact_at_touch_mm=None if first_touch_travel is None else 1000*first_touch_travel,
               first_forecast_200ms_mm=1000*errors[.2] if .2 in errors else None,
               first_forecast_400ms_mm=1000*errors[.4] if .4 in errors else None,
               max_feasible=max((r["feasible"] for r in calls), default=0),
               multiple_choice_ticks=sum(r["feasible"] >= 2 for r in calls))
    # Common result schema, including baseline rows which have no approach diagnostics.
    for key in ("close_reason", "close_minus_contact_ms", "last_target_fraction", "last_predicted_speed",
                "measured_fraction", "actual_tilt_deg", "contact_ball_speed"):
        row.setdefault(key, None)
    sample = dict(seed=seed, case=case, strategy=strategy, contact=row["first_contact_s"],
                  held=row["held"], qualified=row["qualified"], first_plan=row["first_application_s"],
                  first_estimate=dict(time=estimate.time, p=estimate.position.tolist(),
                                      v=estimate.velocity.tolist(), g=estimate.gravity.tolist()), path=path)
    return row, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case, start, velocity, duration, distance, offset in throws(center, model.opt.gravity, args.seed):
        for name in STRATEGIES:
            row, sample = run_trial(model, center, args.seed, case, start, velocity, name)
            row.update(home_crossing_s=duration, horizontal_throw_m=distance, home_offset_m=offset)
            rows.append(row)
            if name in ("select_height", "seven_heights"):
                (args.output_dir / f"{name}_case{case}.json").write_text(json.dumps(sample), encoding="utf-8")
            print(args.seed, case, name, row["contacted"], row["held"], row["qualified"], flush=True)
    save_csv(args.output_dir / "trials.csv", rows)


if __name__ == "__main__":
    main()
