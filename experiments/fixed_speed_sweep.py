"""Paired fixed-intercept speed sweep, prospective rule selection and observation stress test."""
import argparse
import csv
from functools import partial
from itertools import product
from pathlib import Path
from statistics import median

import numpy as np

from controllers.ball_observer import BallObserver
from controllers.catching_mpc import CatchingMPC
from controllers.deadline_arm_filter import DeadlineArmFilter
from controllers.fixed_intercept import CalibrationMPC, FixedIntercept, FixedPointMPC, PreparationFinished
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


LABELS = ("locked30", "low", "mid", "high")
RULES = (*LABELS, "fast_high", "fast_low")
TRACKER = partial(DeadlineArmFilter, fast=True, refine=True)


def route(rule, predicted_speed):
    if rule in LABELS:
        return rule
    if rule not in RULES:
        raise ValueError(rule)
    fast = predicted_speed is not None and predicted_speed >= 5.2
    if rule == "fast_high":
        return "high" if fast else "mid"
    return "low" if fast else "mid"


def select(paths):
    table = {}
    for path in paths:
        with path.open(encoding="utf-8-sig") as stream:
            for r in csv.DictReader(stream):
                if r["policy"] == "free30":
                    continue
                key = (int(r["seed"]), int(r["scenario"]), r["policy"])
                if key in table:
                    raise ValueError(f"Duplicate screening row {key}")
                table[key] = r
    keys = list(product((871, 872, 873), range(8)))
    if set(table) != {(*k, name) for k in keys for name in LABELS}:
        raise ValueError("Incomplete or unexpected screening sample")
    common = [k for k in keys if all(int(table[*k, p]["held"]) for p in LABELS)]
    def score(rule):
        choices = {}
        for k in keys:
            speed = table[*k, "locked30"]["estimated_incoming_m_s"]
            name = route(rule, float(speed) if speed else None)
            choices[k] = table[*k, name]
        return (-sum(int(r["qualified"]) for r in choices.values()),
                -sum(int(r["held"]) for r in choices.values()),
                median(float(choices[k]["impact20_peak_N"]) for k in common) if common else 0.,
                RULES.index(rule))
    return min(RULES, key=score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, nargs="+")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--wide-mid", action="store_true", help="Use the frozen 0.15 m/s certified-span rule")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    winner = select(args.selection) if args.selection else None
    if args.wide_mid and winner is not None:
        parser.error("Choose a single source of the routing rule")
    if args.stress and winner is None and not args.wide_mid:
        parser.error("Stress testing requires a rule chosen on the development throws")
    observer = partial(BallObserver, noise_m=.004, latency_s=.040) if args.stress else BallObserver
    model = build_scene("ball")
    center = capture_center(initial(model))
    results = []
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        f = FixedIntercept()
        try:
            trial(model, args.seed, case, p, v, True, tracker_factory=TRACKER, observer_factory=observer,
                  planner_factory=partial(CalibrationMPC, fixture=f))
        except PreparationFinished:
            pass
        speed = float(np.linalg.norm(f.incoming)) if f.time is not None else None
        selected = route(winner, speed) if winner else None
        policies = list(("free30", *LABELS))
        if args.wide_mid:
            wide = f.reason == "ready" and f.levels[-1]-f.levels[0] >= .15
            selected = "mid" if wide else "locked30"
            policies = list(dict.fromkeys(("free30", "locked30", selected)))
        elif args.stress:
            policies = list(dict.fromkeys(("free30", "locked30", selected)))
        shift = case % len(policies)
        for name in policies[shift:] + policies[:shift]:
            factory = CatchingMPC if name == "free30" else partial(FixedPointMPC, fixture=f, label=name)
            target_error = predicted_terminal_speed = None
            def frame(m, d, info, track, arm):
                nonlocal target_error, predicted_terminal_speed
                plan = info.get("applied_reference")
                if plan is None:
                    return
                if name != "free30" and (abs(plan.intercept_time-f.time)>1e-8 or not np.array_equal(plan.point,f.point)):
                    raise AssertionError("Applied plan does not share the locked endpoint")
                if info["first_contact_s"] is not None and target_error is None:
                    target_error = 1000*float(np.linalg.norm(capture_center(d)-plan.point))
                    if plan.predicted_match is not None:
                        predicted_terminal_speed = plan.predicted_match*plan.incoming_speed
            result = trial(model, args.seed, case, p, v, True, tracker_factory=TRACKER, planner_factory=factory,
                           record_frame=frame, observer_factory=observer)
            measured = result["measured_fraction"]
            result.update(policy=name, selected_rule="wide_mid" if args.wide_mid else winner or "", routed_label=selected or "",
                    stress=int(args.stress), fixture_status=f.reason, observed_s=f.observed_s, release_s=f.release_s,
                    fixed_time=f.time, **{f"fixed_{axis}": None if f.point is None else f.point[i]
                                         for i,axis in enumerate("xyz")},
                    estimated_incoming_m_s=speed, preparation_ms=f.preparation_ms, feasible_levels=len(f.levels),
                    min_speed=f.target_speed("low"), max_speed=f.target_speed("high"),
                    requested_m_s=f.target_speed(name) if name in ("low","mid","high") else None,
                    verified_terminal_m_s=predicted_terminal_speed, onset_target_error_mm=target_error,
                    measured_m_s=None if measured is None else measured*result["contact_ball_speed"])
            results.append(result)
        print(args.seed, case, f.reason, [round(s,3) for s in f.levels], selected, flush=True)
    save_csv(args.output_dir/"trials.csv", results)


if __name__ == "__main__":
    main()
