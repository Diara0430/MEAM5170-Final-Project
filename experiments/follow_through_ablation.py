"""Paired catch trials with a shared timing schedule and a ball-removed replay."""
import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from controllers.discrete_arm_filter import DiscreteArmFilter
from controllers.follow_through import FollowThrough
from controllers.sphere_cage import SphereCageClosure
from experiments.active_ball_catch import launch
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene


ROOT = Path(__file__).resolve().parents[1]


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


class TraceTracker(DiscreteArmFilter):
    def __init__(self, model):
        super().__init__(model)
        self.last_plan = None

    def command(self, data, plan, t):
        self.switched = self.last_plan is not plan
        self.last_plan = plan
        self.reference_accel = plan.reference(t)[2].copy()
        super().command(data, plan, t)


def run_trial(model, xyz, speed, seed, case, policy, schedule, approach=None):
    tracker = TraceTracker(model)
    flight = model.joint("flight").dofadr[0]
    ball = model.geom("projectile").id
    hand = {i for i, body in enumerate(model.geom_bodyid)
            if (model.body(body).name or "").startswith("hand/")}
    trace = []
    controls = []
    previous = None
    direction = None
    impulse = 0.0
    peak_force = 0.0
    dt = model.opt.timestep

    def frame(m, d, info):
        nonlocal previous, direction, impulse, peak_force
        if previous is None:
            previous = d.qvel.copy()
            return
        start = float(d.time) - dt
        first = info["first_contact_s"]
        acceleration = (d.qvel[tracker.dofs] - previous[tracker.dofs]) / dt
        if first is None:
            controls.append((d.ctrl.copy(), d.qvel[tracker.dofs].copy()))
        if first is not None and start - first < .020 - 1e-9:
            if direction is None:
                incoming = previous[flight:flight + 3]
                direction = incoming / np.linalg.norm(incoming)
            net = np.zeros(3)
            for index, contact in enumerate(d.contact):
                if contact.efc_address < 0:
                    continue
                if contact.geom2 == ball and contact.geom1 in hand:
                    sign = 1.0
                elif contact.geom1 == ball and contact.geom2 in hand:
                    sign = -1.0
                else:
                    continue
                wrench = np.empty(6)
                mujoco.mj_contactForce(m, d, index, wrench)
                net += sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
            # Positive means that the hand removes momentum along the incoming velocity.
            impulse -= float(net @ direction) * dt
            peak_force = max(peak_force, float(np.linalg.norm(net)))
        previous = d.qvel.copy()
        if first is None or start <= first + .15:
            trace.append((start, int(first is None), int(tracker.switched),
                          tracker.reference_accel.copy(), acceleration.copy(),
                          d.actuator_force[tracker.ids].copy()))

    result, calls, _ = run_case(
        model, xyz, speed, case, "near", seed, "mpc_match",
        solver_schedule=schedule, allow_schedule_extension=True,
        tracker_factory=lambda m: tracker,
        hand_controller_factory=lambda m, h: SphereCageClosure(m, h),
        mpc_acceleration_limit=3.0, follow_through=policy, frame_observer=frame, approach=approach,
    )
    # Same actuator commands, including fingers, but no ball-hand contact.
    ghost = launch(model, np.array([4., 4., 10.]), np.zeros(3))
    error = 0.0
    for command, original_velocity in controls:
        ghost.ctrl[:] = command
        mujoco.mj_step(model, ghost)
        error = max(error, float(np.max(np.abs(ghost.qvel[tracker.dofs] - original_velocity))))
    result.update(net_opposing_impulse_20ms_Ns=impulse if direction is not None else None,
                  peak_net_force_20ms_N=peak_force if direction is not None else None,
                  precontact_replay_steps=len(controls), precontact_replay_velocity_error=error)
    return result, calls, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[810, 811])
    parser.add_argument("--cases", nargs="+", type=int, choices=range(12), default=list(range(12)))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedules = {}
    with (ROOT / "results/arm_filter_validation/solver_times.csv").open(encoding="utf-8-sig") as f:
        model = build_scene("ball")
        for row in csv.DictReader(f):
            if row["strategy"] == "headroom_filter":
                key = int(row["seed"]), int(row["scenario"])
                schedules.setdefault(key, {})[round(float(row["time_s"]) / model.opt.timestep)] = bool(int(row["overrun"]))
    rest = initial(model)
    palm = rest.body("hand/palm_lower")
    center = palm.xpos + palm.xmat.reshape(3, 3) @ np.array([0., -.055, -.08])
    results = []
    for seed in args.seeds:
        for case, kind, xyz, speed in scenarios(center, model.opt.gravity, seed=seed):
            if kind != "near" or case not in args.cases:
                continue
            schedule = schedules.get((seed, case))
            for name, policy in (("headroom_filter", None), ("follow_through", FollowThrough())):
                result, calls, trace = run_trial(model, xyz, speed, seed, case, policy, schedule)
                if calls:
                    save_csv(args.output_dir / f"seed{seed}_case{case}_{name}_timing.csv", calls)
                if name == "headroom_filter":
                    schedule = {round(r["time_s"] / model.opt.timestep): bool(r["overrun"]) for r in calls}
                result["strategy"] = name
                results.append(result)
                save_csv(args.output_dir / "trials.csv", results)
                traces = []
                for t, pre, switched, commanded, actual, torque in trace:
                    row = dict(step_start_s=t, precontact=pre, new_reference=switched)
                    for j in range(7):
                        row.update({f"ref_accel_{j+1}": commanded[j], f"actual_accel_{j+1}": actual[j],
                                    f"actuator_Nm_{j+1}": torque[j]})
                    traces.append(row)
                save_csv(args.output_dir / f"seed{seed}_case{case}_{name}.csv", traces)
                print(seed, case, name, result["held"], result["qualified"], flush=True)


if __name__ == "__main__":
    main()
