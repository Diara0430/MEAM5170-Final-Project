"""Isolate wrist compensation on replayed and unseen throws with the rolling planner."""
import argparse
from pathlib import Path

from controllers.deadline_arm_filter import DeadlineArmFilter
from controllers.wrist_impact import WristImpactFilter
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


def compare(model, seed, case, p, v, wrist):
    tracker = None
    def make(m, observer, enabled):
        nonlocal tracker
        tracker = (WristImpactFilter(m, observer, enabled) if wrist else
                   DeadlineArmFilter(m, observer, enabled, fast=True, refine=True))
        return tracker
    row = trial(model, seed, case, p, v, True, tracker_factory=make)
    row.update(policy="wrist_guard" if wrist else "reference",
               guard_scored=getattr(tracker, "scored_steps", 0),
               guard_changed=getattr(tracker, "changed_steps", 0))
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        for wrist in ((False,True) if case%2==0 else (True,False)):
            rows.append(compare(model, args.seed, case, p, v, wrist))
    save_csv(args.output_dir/"trials.csv", rows)
