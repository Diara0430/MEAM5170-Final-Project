"""Compare achievable-velocity interception on matched noisy long throws."""
import argparse
import csv
from pathlib import Path
from statistics import median

import mujoco
import numpy as np

from controllers.velocity_selection import VelocitySelection
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, run_trial, throws
from scene_setup.scene import build_scene


POLICIES = {"reference": None, "rank_velocity": VelocitySelection(),
            "prepare_velocity": VelocitySelection(prepare_early=True)}


class ImpactMeasurement:
    """Integrate the resultant hand-on-ball force against the incoming flight direction."""
    def __init__(self, model):
        self.projectile = model.geom("projectile").id
        self.flight = model.joint("flight").dofadr[0]
        self.hand = {i for i, b in enumerate(model.geom_bodyid)
                     if (model.body(b).name or "").startswith("hand/")}
        self.direction = None
        self.impulse = 0.
        self.peak_force = 0.

    def __call__(self, model, data, t, contact, close, previous_velocity, reference):
        if t >= contact + .020 - 1e-9:
            return
        if self.direction is None:
            v = previous_velocity[self.flight:self.flight+3]
            self.direction = v / np.linalg.norm(v)
        force = np.zeros(3)
        for i, c in enumerate(data.contact):
            if c.efc_address < 0:
                continue
            if c.geom2 == self.projectile and c.geom1 in self.hand:
                sign = 1.
            elif c.geom1 == self.projectile and c.geom2 in self.hand:
                sign = -1.
            else:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            force += sign * (c.frame.reshape(3, 3).T @ wrench[:3])
        self.impulse -= float(force @ self.direction) * model.opt.timestep
        self.peak_force = max(self.peak_force, float(np.linalg.norm(force)))


def select_policy(paths):
    groups = {name: {} for name in POLICIES}
    for path in paths:
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = int(row["seed"]), int(row["scenario"])
                group = groups[row["policy"]]
                if key in group:
                    raise ValueError(f"Duplicate screening throw: {key}")
                group[key] = row
    reference = groups["reference"]
    expected = {(s, i) for s in (821, 822, 823) for i in range(8)}
    if any(set(group) != expected for group in groups.values()):
        raise ValueError("Selection requires all three complete paired screening seeds")
    common = [key for key in reference if all(int(g[key]["held"]) for g in groups.values())]
    if not common:
        raise ValueError("No commonly held throws to compare impacts")

    def rank(name):
        group = groups[name]
        # Prefer complete qualification, then active holds, before comparing
        # contact acceleration and impulse on exactly the same successful throws.
        return (-sum(int(r["qualified"]) for r in group.values()),
                -sum(int(r["held"]) for r in group.values()),
                median(float(group[k]["peak_contact20_accel"])-float(reference[k]["peak_contact20_accel"])
                       for k in common),
                median(float(group[k]["impulse20_Ns"])-float(reference[k]["impulse20_Ns"])
                       for k in common), name)

    return min(("rank_velocity", "prepare_velocity"), key=rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", nargs="+", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = ("reference", select_policy(args.selection)) if args.selection else tuple(POLICIES)
    print("Policies:", policies, flush=True)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case, start, velocity, duration, distance, offset in throws(center, model.opt.gravity, args.seed):
        # Rotate execution order to avoid consistently giving any policy the first run.
        order = policies[case % len(policies):] + policies[:case % len(policies)]
        for policy in order:
            impact = ImpactMeasurement(model)
            row, sample = run_trial(model, center, args.seed, case, start, velocity, "seven_heights",
                                    velocity_selection=POLICIES[policy], impact_observer=impact)
            applied = [f for f in sample["path"] if f["intercept"] is not None]
            last = applied[-1] if applied else None
            row.update(policy=policy, home_crossing_s=duration, horizontal_throw_m=distance,
                       home_offset_m=offset, impact20_peak_N=impact.peak_force if row["contacted"] else None,
                       impulse20_Ns=impact.impulse if row["contacted"] else None,
                       last_height_mm=1000*(last["intercept"][2]-center[2]) if last else None,
                       last_intercept_s=last["intercept_s"] if last else None)
            for key in ("predicted_match", "predicted_normal_speed", "planned_stop_margin_rad"):
                row.setdefault(key, None)
            rows.append(row)
            print(args.seed, case, policy, row["held"], row["qualified"], flush=True)
    save_csv(args.output_dir / "trials.csv", rows)


if __name__ == "__main__":
    main()
