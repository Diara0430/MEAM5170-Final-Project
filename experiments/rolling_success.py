"""Paired rolling-intercept study with fixed hardware, observation latency and pass criteria."""
import argparse
import csv
import itertools
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from controllers.deadline_arm_filter import DeadlineArmFilter
from controllers.lock_timing import RULES, TimingMPC
from controllers.segmented_flight import SegmentedFlightFit
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.lock_timing import StudyObserver
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


FACTORS = ("track", "reserve", "wrist")
POLICIES = ("rolling", *("_".join(c) for n in (1,2,3) for c in itertools.combinations(FACTORS,n)))
DEV_SEEDS = (971,972,973)
DEV_PROFILES = ("normal","noisy")


class StudyTrack(StudyObserver):
    def __init__(self, *args, long_track, **kwargs):
        super().__init__(*args, **kwargs)
        if long_track:
            self.frames = deque(maxlen=48)
            self.flight_fit = SegmentedFlightFit()
        self.sample_max_ms = 0.

    def sample(self, now, position):
        start = time.perf_counter()
        estimate = super().sample(now, position)
        self.sample_max_ms = max(self.sample_max_ms, 1000*(time.perf_counter()-start))
        return estimate


def run(model, seed, case, start, velocity, profile, policy):
    factors = set(policy.split("_"))
    sensor = None

    def make_sensor(gravity, noise_seed):
        nonlocal sensor
        sensor = StudyTrack(gravity, noise_seed, profile=profile, seed=seed, case=case,
                            long_track="track" in factors)
        return sensor

    def make_tracker(m, observer, enabled):
        tracker = DeadlineArmFilter(m, observer, enabled, fast=True, refine=True)
        if "reserve" in factors:
            tracker.config = replace(tracker.config, speed=.985)
        return tracker

    def make_planner(*args, **kwargs):
        return TimingMPC(*args, observer=sensor, rule=RULES["rolling"],
                         planning_speed_limit=.97 if "reserve" in factors else 1.,
                         orientation_tolerance=.085 if "wrist" in factors else .03, **kwargs)

    row = trial(model, seed, case, start, velocity, True, tracker_factory=make_tracker,
                observer_factory=make_sensor, planner_factory=make_planner)
    row.update(profile=profile, policy=policy,
               detected_shifts=sensor.flight_fit.change_count if sensor.flight_fit else 0,
               camera_max_ms=sensor.sample_max_ms)
    return row


def choose(paths):
    rows = {}
    for path in paths:
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = (int(row["seed"]),int(row["scenario"]),row["profile"],row["policy"])
                if key in rows:
                    raise ValueError("Duplicate development trial")
                rows[key] = row
    expected = set(itertools.product(DEV_SEEDS,range(8),DEV_PROFILES,POLICIES))
    if rows.keys() != expected:
        raise ValueError("Incomplete or unexpected development set")

    def rank(policy):
        groups = [[rows[s,i,p,policy] for s in DEV_SEEDS for i in range(8)] for p in DEV_PROFILES]
        qualified = [sum(int(r["qualified"]) for r in g) for g in groups]
        held = sum(int(r["held"]) for g in groups for r in g)
        return -sum(qualified), -min(qualified), -held, len(policy.split("_")), POLICIES.index(policy)

    # Always test the best *challenger* against a fresh run of the baseline.
    return min(POLICIES[1:],key=rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=(*DEV_PROFILES,"corrected"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--selection", nargs="+", type=Path)
    group.add_argument("--policy", choices=POLICIES)
    args = parser.parse_args()
    if args.selection:
        policies = ("rolling",choose(args.selection))
    elif args.policy:
        policies = (args.policy,)
    else:
        policies = POLICIES
    print("Policies", policies, flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case,p,v,*_ in throws(center,model.opt.gravity,args.seed):
        shift = (args.seed+case) % len(policies)
        for policy in policies[shift:]+policies[:shift]:
            rows.append(run(model,args.seed,case,p,v,args.profile,policy))
        print("Finished throw",case,flush=True)
    save_csv(args.output_dir/"trials.csv",rows)


if __name__ == "__main__":
    main()
