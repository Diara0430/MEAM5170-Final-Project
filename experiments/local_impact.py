"""Paired soft-catching trials on the frozen track_reserve_edge baseline."""
import argparse
from functools import partial
from pathlib import Path

from controllers.impact_spot import ImpactSpot
from controllers.short_impact import ShortImpact
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.rolling_reach import compare
from scene_setup.scene import build_scene


FACTORIES = {"baseline": None, "short": ShortImpact, "short_shadow": partial(ShortImpact,shadow=True),
             "spot": ImpactSpot, "spot_shadow": partial(ImpactSpot,shadow=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("normal","noisy","corrected"), required=True)
    parser.add_argument("--methods", nargs="+", choices=FACTORIES, default=list(FACTORIES))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be distinct")
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case,p,v,*_ in throws(center,model.opt.gravity,args.seed):
        shift = (args.seed+case) % len(args.methods)
        for method in args.methods[shift:]+args.methods[:shift]:
            factory = FACTORIES[method]
            row = compare(model,args.seed,case,p,v,args.profile,"track_reserve",True,
                          selector_factory=factory)
            row["method"] = method
            rows.append(row)
    # Baseline trials have no online selector diagnostics.
    columns = dict.fromkeys(k for r in rows for k in r)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    save_csv(args.output_dir/"trials.csv",[{**columns,**row} for row in rows])


if __name__=="__main__":
    main()
