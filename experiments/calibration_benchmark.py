"""Compare exhaustive and nominal-speed fixture preparation on paired throws."""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
from controllers.catching_mpc import CatchingMPC
from controllers.fixed_intercept import CalibrationMPC
from experiments.fast_calibration import FastCalibrationMPC
from experiments.fixed_angle_study import run
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


def compare(model, seed, case, p, v, profile):
    results = []
    methods = [('grid', CalibrationMPC), ('nominal', FastCalibrationMPC)]
    if (seed + case) % 2:
        methods.reverse()
    original_solve = CatchingMPC.solve
    for name, calibration in methods:
        profile_data = defaultdict(float)
        def timed_solve(self, *args, **kwargs):
            if not isinstance(self, CalibrationMPC):
                return original_solve(self, *args, **kwargs)
            start = time.perf_counter()
            try:
                return original_solve(self, *args, **kwargs)
            finally:
                label = 'speed' if kwargs.get('speed_request') is not None else 'anchor'
                profile_data[label+'_ms'] += (time.perf_counter()-start)*1000
                profile_data[label+'_calls'] += 1
        with patch.object(CatchingMPC, 'solve', timed_solve), patch('experiments.fixed_angle_study.CalibrationMPC', calibration):
            row = run(model, seed, case, p, v, profile, [0.])[0]
        row.update(method=name, anchor_solve_ms=profile_data['anchor_ms'],
                   anchor_solve_calls=int(profile_data['anchor_calls']),
                   speed_solve_ms=profile_data['speed_ms'], speed_solve_calls=int(profile_data['speed_calls']))
        results.append(row)
        print(f'{seed} {profile} case={case} {name} {row["fixture_status"]} prep={row["calibration_ms"]:.1f}ms speed_calls={row["speed_solve_calls"]} held={row["held"]}', flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cases', type=int, default=8)
    parser.add_argument('--profile', choices=('normal','noisy'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args=parser.parse_args()
    model=build_scene('box')
    center=capture_center(initial(model))
    rows=[]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case,p,v,*_ in throws(center,model.opt.gravity,args.seed):
        if case>=args.cases:
            continue
        rows.extend(compare(model,args.seed,case,p,v,args.profile))
        columns=list(dict.fromkeys(k for r in rows for k in r))
        with (args.output_dir/'trials.csv').open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=columns)
            w.writeheader()
            w.writerows(rows)

if __name__=='__main__':
    main()
