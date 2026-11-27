"""Paired exploratory cuboid catches with a fixed palm-normal rotation."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from controllers.palm_aim import PalmAimRecovery
from experiments.shape_catch import simulate
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


def run(seed, cases, profile, angles, output):
    model = build_scene("box")
    center = capture_center(initial(model))
    rows = []
    for case, p, v, *_ in throws(center, model.opt.gravity, seed):
        if case >= cases:
            continue
        # 轮换顺序，降低运行顺序与墙钟期限波动造成的偏差。
        k = (seed + case) % len(angles)
        for degrees in angles[k:] + angles[:k]:
            class RotatedPalm(PalmAimRecovery):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.orientation = self.orientation @ Rotation.from_euler("z", degrees, degrees=True).as_matrix()
            with patch("experiments.rolling_reach.PalmAimRecovery", RotatedPalm):
                row = simulate(model, seed, case, p, v, profile, "pose", grasp="original")
            row["palm_angle_deg"] = degrees
            rows.append(row)
            print(f"case={case} angle={degrees} held={row.get('held')} qualified={row.get('qualified')}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    columns = dict.fromkeys(k for row in rows for k in row)
    save_csv(output / "trials.csv", [{**columns, **row} for row in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1161)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--profile", choices=("normal", "noisy", "corrected"), default="normal")
    parser.add_argument("--angles", type=float, nargs="+", default=[0., -10., 10., -20., 20.])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.cases, args.profile, args.angles, args.output_dir)
