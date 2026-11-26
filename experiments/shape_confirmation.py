"""Choose one cuboid planner on development throws and compare grips on new throws."""
import argparse
import csv
from pathlib import Path

from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.rolling_reach import compare
from experiments.shape_catch import METHODS, simulate
from scene_setup.scene import build_scene


def choose(screen):
    with screen.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(int(r["seed"]), int(r["scenario"]), r["profile"], r["method"]) for r in rows}
    expected = {(s, i, p, m) for s in (1051, 1052, 1053) for i in range(8)
                for p in ("normal","noisy") for m in METHODS}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError("Development selection requires every unique method and throw")
    def score(method):
        counts = []
        holds = 0
        for p in ("normal","noisy"):
            group = [r for r in rows if r["method"]==method and r["profile"]==p]
            counts.append(sum(int(r["qualified"]) for r in group))
            holds += sum(int(r["held"]) for r in group)
        return -sum(counts), -min(counts), -holds, METHODS.index(method)
    return min(METHODS, key=score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("normal","noisy","corrected"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seed in (1051, 1052, 1053):
        parser.error("Verification seeds must be independent of selection")
    winner = choose(args.screen)
    options = list(dict.fromkeys([("ball", "baseline", "original"), ("box", "center", "original"),
                                  ("box", "pose", "original"), ("box", winner, "original"),
                                  ("box", winner, "aperture"), ("box", winner, "aperture_slip"),
                                  ("box", "axis_lever", "original"), ("box", "axis_preload", "original")]))
    models = {shape: build_scene(shape) for shape in ("box", "ball")}
    center = capture_center(initial(models["box"]))
    rows = []
    for case, p, v, *_ in throws(center, models["box"].opt.gravity, args.seed):
        shift = (args.seed+case) % len(options)
        for shape, method, grasp in options[shift:]+options[:shift]:
            if shape == "box":
                row = simulate(models[shape], args.seed, case, p, v, args.profile, method, grasp=grasp)
            else:
                row = compare(models[shape], args.seed, case, p, v, args.profile, "track_reserve", True)
            row.update(shape=shape, method=method, grasp=grasp, selected=winner)
            rows.append(row)
    columns = dict.fromkeys(k for r in rows for k in r)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir/"trials.csv", [{**columns, **r} for r in rows])


if __name__ == "__main__":
    main()
