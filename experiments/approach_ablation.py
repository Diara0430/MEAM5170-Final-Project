"""Compare catch timing, palm aim and speed matching on identical throws."""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from controllers.catch_approach import CatchApproach
from controllers.follow_through import FollowThrough
from experiments.follow_through_ablation import run_trial, save_csv
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from scene_setup.scene import build_scene


POLICIES = {"fixed20": CatchApproach(), "fixed10": CatchApproach(fraction=.1),
            "fixed30": CatchApproach(fraction=.3)}
for entry, tilt, nonlinear in itertools.product((False, True), repeat=3):
    flags = {"entry": entry, "tilt": tilt, "nonlinear": nonlinear}
    name = "_".join(key for key, enabled in flags.items() if enabled)
    if name:
        POLICIES[name] = CatchApproach(**flags)


def retime(start, velocity, center_height, gravity, profile):
    """Keep the nominal descending crossing point, changing available flight time."""
    if profile == "nominal":
        return velocity
    roots = np.roots([gravity[2] / 2, velocity[2], start[2] - center_height])
    descending = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0
                  and velocity[2] + gravity[2] * r.real < 0]
    if not descending:
        raise ValueError("Nominal throw must cross the capture plane while descending")
    t = min(descending)
    crossing = start + velocity * t + gravity * t**2 / 2
    duration = {"short": .33, "long": .78}[profile]
    return (crossing - start - gravity * duration**2 / 2) / duration


def rank_candidates(paths):
    groups = {name: {} for name in POLICIES}
    for path in paths:
        with path.open(encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                key = (row["seed"], row["scenario"], row["profile"])
                group = groups[row["policy"]]
                if key in group:
                    raise ValueError(f"Duplicate screening trial {key}")
                group[key] = row
    if not groups["fixed20"] or any(g.keys() != groups["fixed20"].keys() for g in groups.values()):
        raise ValueError("Selection requires all paired screening trials")

    def score(name):
        rows = list(groups[name].values())
        policy = POLICIES[name]
        complexity = sum((policy.entry, policy.tilt, policy.nonlinear, policy.fraction != .20))
        return (-sum(int(r["qualified"]) for r in rows), -sum(int(r["held"]) for r in rows),
                complexity, name)
    return sorted(groups, key=score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profiles", nargs="+", choices=["nominal", "short", "long"], default=["nominal"])
    parser.add_argument("--selection", type=Path, nargs="+", help="Screening CSVs to choose confirmation arms")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.selection:
        ranking = rank_candidates(args.selection)
        winner = ranking[0]
        challenger = next(n for n in ranking if n != "fixed20")
        best_nonlinear = next(n for n in ranking if POLICIES[n].nonlinear)
        names = list(dict.fromkeys(("fixed20", challenger, best_nonlinear)))
        print("Screen winner:", winner, "confirmation:", names, flush=True)
    else:
        names = list(POLICIES)

    model = build_scene("ball")
    dt = model.opt.timestep
    palm = initial(model).body("hand/palm_lower")
    center = palm.xpos + palm.xmat.reshape(3, 3) @ np.array([0., -.055, -.08])
    rows = []
    for case, kind, p, nominal_v in scenarios(center, model.opt.gravity, seed=args.seed):
        if kind != "near":
            continue
        for profile in args.profiles:
            v = retime(p, nominal_v, center[2], model.opt.gravity, profile)
            schedule = None
            for name in names:
                result, calls, _ = run_trial(
                    model, p, v, args.seed, case, FollowThrough(), schedule, approach=POLICIES[name])
                if result["precontact_replay_velocity_error"] > 1e-7:
                    raise ValueError("Precontact replay velocity mismatch")
                # In screening, all common ticks use the baseline timing decision.
                # Confirmation charges every policy its own measured MPC overruns.
                if not args.selection and name == "fixed20":
                    schedule = {round(r["time_s"]/dt): bool(r["overrun"]) for r in calls}
                result.update(policy=name, profile=profile,
                              timing="live" if args.selection else "shared",
                              measured_qp_overruns=sum(r["measured_overrun"] for r in calls))
                rows.append(result)
            save_csv(args.output_dir / "trials.csv", rows)
    for profile in args.profiles:
        for name in names:
            group = [r for r in rows if r["profile"] == profile and r["policy"] == name]
            if len(group) != 12:
                raise ValueError("Expected twelve throws per cell")
            print(profile, name, "held", sum(r["held"] for r in group),
                  "qualified", sum(r["qualified"] for r in group), flush=True)


if __name__ == "__main__":
    main()
