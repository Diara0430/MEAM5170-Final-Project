"""Fresh throws with equal deadline rejection for each inner-loop policy."""
import argparse
from functools import partial
from pathlib import Path

from controllers.deadline_arm_filter import DeadlineArmFilter
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refinement", action="store_true", help="Compare the fast preview with optional torque refinement")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    modes = (("robot_only", False, False, False), ("projected_contact", True, False, False),
             ("fast_robot_only", False, True, False), ("fast_projected_contact", True, True, False))
    if args.refinement:
        modes = (("fast_projected_contact", True, True, False), ("refined_contact", True, True, True))
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        offset = case % len(modes)
        for name, enabled, fast, refine in modes[offset:]+modes[:offset]:
            row = trial(model, args.seed, case, p, v, enabled,
                        tracker_factory=partial(DeadlineArmFilter, fast=fast, refine=refine))
            row["policy"] = name
            rows.append(row)
            print(args.seed, case, name, row["held"], row["qualified"], flush=True)
    save_csv(args.output_dir / "trials.csv", rows)
