"""Fresh same-throw validation of sign-specific wrist preloading, with a timing control."""
import argparse
from pathlib import Path

from controllers.wrist_preload import WristPreload
from experiments.contact_joint_replay import ContactTrace
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.rolling_reach import compare
from scene_setup.scene import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("normal","noisy","corrected"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = build_scene("ball")
    rows = []
    for case, p, v, *_ in throws(capture_center(initial(model)), model.opt.gravity, args.seed):
        methods = ["baseline","preload","preload_shadow"]
        shift = (args.seed+case) % len(methods)
        for method in methods[shift:]+methods[:shift]:
            trace, tracker = ContactTrace(model), None
            def make(*a, **kwargs):
                nonlocal tracker
                tracker = WristPreload(*a, shadow=method=="preload_shadow", **kwargs)
                return tracker
            options = {} if method=="baseline" else {"tracker_type": make}
            row = compare(model,args.seed,case,p,v,args.profile,"track_reserve",True,
                          record_frame=trace.record, **options)
            rows.append(dict(**row, **trace.summary(row), **(tracker.summary() if tracker else {}), method=method))
    cols = dict.fromkeys(k for r in rows for k in r)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir/"trials.csv", [{**cols, **r} for r in rows])


if __name__ == "__main__":
    main()
