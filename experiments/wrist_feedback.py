"""固定、半速、自适应倾腕的配对阵风实验。图表文字使用英文。"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.wrist import WristProgress
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed


def trial(args):
    seed, mode, split = args
    pulse = gust_for_seed(seed)
    timing = TimingTrial(pulse, False)
    wrist = WristProgress(mode)
    run("early", .04, .19, True, shape="box", pitch_deg=40., palm_x_mm=0.,
        arrival_speed="fast", timing_trial=timing, wrist_control=wrist)
    row = dict(seed=seed, split=split, mode=mode, gust_onset_s=pulse[0],
               gust_duration_s=pulse[1], gust_az_m_s2=pulse[2], **timing.outcome,
               tilt_duration_s=None if wrist.completed_at is None else wrist.completed_at-.9,
               slowed_s=wrist.slow_time)
    trace = [dict(seed=seed, split=split, mode=mode, **r) for r in wrist.trace]
    return row, trace


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["split"], row["mode"]].append(row)
    result = []
    for (split, mode), trials in groups.items():
        baseline = {r["seed"]: r for r in groups[split, "fixed"]}
        result.append(dict(split=split, mode=mode, n=len(trials),
            caught=sum(r["passed"] for r in trials),
            gains=sum(r["passed"] and not baseline[r["seed"]]["passed"] for r in trials),
            losses=sum(not r["passed"] and baseline[r["seed"]]["passed"] for r in trials),
            median_drift_mm=median(r["max_drift_mm"] for r in trials),
            median_tilt_s=median(r["tilt_duration_s"] for r in trials if r["tilt_duration_s"] is not None),
            timeouts=sum(r["tilt_duration_s"] is None for r in trials)))
    return result


def plot(path, summary):
    # 用成功数与倾腕时长同时显示减速的效果和代价。
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="410">',
           '<rect width="900" height="410" fill="white"/>',
           '<g font-family="Arial" fill="#213547">',
           '<text x="35" y="36" font-size="22">Wrist speed feedback under paired gusts</text>',
           '<text x="35" y="61" font-size="13">40-degree fast box, fixed fingers, 0 mm palm offset. All trials require final tilt and 0.5 s hold.</text>']
    colors = {"fixed": "#28648d", "slow": "#81999f", "adaptive": "#bd542c"}
    for panel, split in enumerate(("selection", "validation")):
        x = 50+panel*440
        svg.append(f'<text x="{x}" y="100" font-size="18">{split.title()}</text>')
        for i, r in enumerate(s for s in summary if s["split"] == split):
            y = 140+i*75
            svg.append(f'<text x="{x}" y="{y}" font-size="14">{r["mode"]}</text>')
            svg.append(f'<rect x="{x+80}" y="{y-15}" width="{200*r["caught"]/r["n"]}" height="20" fill="{colors[r["mode"]]}"/>')
            svg.append(f'<text x="{x+290}" y="{y}" font-size="14">{r["caught"]}/{r["n"]}</text>')
            svg.append(f'<text x="{x+80}" y="{y+25}" font-size="12">Median tilt {r["median_tilt_s"]:.2f} s | Gains {r["gains"]}, losses {r["losses"]}</text>')
    svg.append('<text x="50" y="385" font-size="13">Bar length = success fraction. Gains and losses are relative to fixed speed within each split.</text></g></svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # 阈值事先固定，新种子用于本轮比较。验证结果不参与调参。
    jobs = [(seed, mode, split) for split, seeds in
            (("selection", range(408, 432)), ("validation", range(432, 480)))
            for seed in seeds for mode in ("fixed", "slow", "adaptive")]
    rows, traces = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row, trace in pool.map(trial, jobs):
            rows.append(row)
            traces.extend(trace)
    summary = summarize(rows)
    write_rows(args.output_dir/"trials.csv", rows)
    write_rows(args.output_dir/"time_series.csv", traces)
    write_rows(args.output_dir/"summary.csv", summary)
    plot(args.output_dir/"wrist_feedback_en.svg", summary)
    for row in summary:
        print(row)
