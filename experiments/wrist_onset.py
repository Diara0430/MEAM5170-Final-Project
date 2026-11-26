"""Matched long throws for two sixth-joint onset interventions and computation controls."""
import argparse
from functools import partial
from pathlib import Path

from controllers.wrist_onset import WristLeverAim, WristOnsetFilter
from experiments.contact_joint_replay import ContactTrace
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.rolling_reach import compare
from scene_setup.scene import build_scene


METHODS = ("baseline", "wrist", "wrist_shadow", "lever", "lever_shadow")


def simulate(model, seed, case, p, v, profile, method):
    trace = ContactTrace(model)
    tracker = None
    options = {}
    if method.startswith("wrist"):
        def make(*args, **kwargs):
            nonlocal tracker
            tracker = WristOnsetFilter(*args, shadow=method=="wrist_shadow", **kwargs)
            return tracker
        options["tracker_type"] = make
    if method.startswith("lever"):
        options["selector_factory"] = partial(WristLeverAim, shadow=method=="lever_shadow")
    row = compare(model, seed, case, p, v, profile, "track_reserve", True,
                  record_frame=trace.record, **options)
    return dict(**row, **trace.summary(row), **(tracker.summary() if tracker else {}), method=method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("normal","noisy","corrected"), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("Specify each method once")
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        shift = (args.seed+case) % len(args.methods)
        for method in args.methods[shift:]+args.methods[:shift]:
            rows.append(simulate(model, args.seed, case, p, v, args.profile, method))
    columns = dict.fromkeys(k for r in rows for k in r)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir/"trials.csv", [{**columns, **r} for r in rows])


if __name__ == "__main__":
    main()
