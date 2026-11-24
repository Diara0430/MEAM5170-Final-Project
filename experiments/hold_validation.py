"""在相同 48 组阵风中，分别评价接触、原姿态保持 2 秒、倾腕稳定性。

倾腕结果引用已有 catch_stability 数据；保持试验独立重放。
这两个成功率的测试轨迹不同，不能当作同一试验的连续阶段成功率。
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.catch_stability import ArmBuffer
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed


def trial(job):
    seed, mode = job
    timing = TimingTrial(gust_for_seed(seed), False)
    run("early", .04, .19, True, shape="box", pitch_deg=40., palm_x_mm=0.,
        arrival_speed="fast", timing_trial=timing,
        arm_buffer=ArmBuffer() if mode == "buffer" else None,
        evaluation="hold", hold_duration_s=2.)
    return dict(seed=seed, mode=mode, test="hold_2s", **timing.outcome)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root/"results/catch_stability/trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        prior = [r for r in csv.DictReader(stream) if r["split"] == "validation" and r["mode"] in ("fixed", "buffer")]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = [(int(r["seed"]), r["mode"]) for r in prior]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(trial, jobs))
    summary = []
    for mode in ("fixed", "buffer"):
        hold = [r for r in rows if r["mode"] == mode]
        tilt = [r for r in prior if r["mode"] == mode]
        summary.append(dict(mode=mode, n=len(hold),
            contacted=sum(r["first_contact_s"] is not None for r in hold),
            hold_2s_success=sum(r["passed"] for r in hold),
            median_hold_drift_mm=median(r["max_drift_mm"] for r in hold),
            tilt105_hold05_success=sum(int(r["passed"]) for r in tilt)))
    write_rows(args.output_dir/"hold_trials.csv", rows)
    write_rows(args.output_dir/"summary.csv", summary)
    for row in summary:
        print(row)
