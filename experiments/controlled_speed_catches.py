"""Held-out velocity-matching study at equal launch-to-crossing time."""
import argparse
from pathlib import Path

import numpy as np

from controllers.catch_approach import CatchApproach
from controllers.follow_through import FollowThrough
from experiments.follow_through_ablation import run_trial, save_csv
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from scene_setup.scene import build_scene


def launch_for_speed(start, velocity, height, gravity, downward_speed, duration=.45):
    """Preserve the original descending XY crossing; prescribe its vertical speed."""
    times = np.roots([.5 * gravity[2], velocity[2], start[2] - height])
    descending = [float(t.real) for t in times if abs(t.imag) < 1e-9 and t.real > 0
                  and velocity[2] + gravity[2] * t.real < 0]
    if len(descending) != 1:
        raise ValueError("Expected a unique descending crossing")
    t = descending[0]
    crossing = start + velocity*t + .5*gravity*t**2
    new_start = start.copy()
    new_start[2] = height + downward_speed * duration + .5*gravity[2]*duration**2
    new_velocity = (crossing - new_start - .5*gravity*duration**2)/duration
    np.testing.assert_allclose(new_start + new_velocity*duration + .5*gravity*duration**2, crossing)
    np.testing.assert_allclose((new_velocity + gravity*duration)[2], -downward_speed)
    return new_start, new_velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vertical-speed", type=float, choices=(2.4, 3.0, 3.6), required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = build_scene("ball")
    rest = initial(model)
    palm = rest.body("hand/palm_lower")
    center = palm.xpos + palm.xmat.reshape(3, 3) @ np.array([0., -.055, -.08])
    policies = [("fixed20", CatchApproach()), ("entry10", CatchApproach(entry=True, fraction=.10)),
                ("entry20", CatchApproach(entry=True)), ("entry30", CatchApproach(entry=True, fraction=.30)),
                ("entry_curve", CatchApproach(entry=True, nonlinear=True))]
    rows = []
    for case, kind, start, velocity in scenarios(center, model.opt.gravity, seed=args.seed):
        if kind != "near":
            continue
        p, v = launch_for_speed(start, velocity, center[2], model.opt.gravity, args.vertical_speed)
        # Rotate order to balance cold-start and timing effects across policies.
        offset = case % len(policies)
        for name, policy in policies[offset:] + policies[:offset]:
            result, calls, _ = run_trial(model, p, v, args.seed, case, FollowThrough(), None, approach=policy)
            if result["precontact_replay_velocity_error"] > 1e-7:
                raise ValueError("Precontact replay velocity mismatch")
            result.update(policy=name, profile=f"v{args.vertical_speed:.1f}", timing="live",
                          measured_qp_overruns=sum(r["measured_overrun"] for r in calls))
            rows.append(result)
        save_csv(args.output_dir / "trials.csv", rows)
    if len(rows) != 60:
        raise ValueError("Expected 12 throws with 5 policies")


if __name__ == "__main__":
    main()
