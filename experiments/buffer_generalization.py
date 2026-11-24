"""固定参数的新阵风验证：球/盒、三档速度、原姿态保持/倾腕两类独立测试。

python experiments/buffer_generalization.py --output-dir results/buffer_generalization
所有条件使用相同的 24 个新阵风种子，不能把重复使用的种子视为独立样本。
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from statistics import median
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.catch_stability import ArmBuffer
from controllers.wrist import WristProgress
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed


def trial(job):
    seed, shape, speed, evaluation, mode = job
    timing = TimingTrial(gust_for_seed(seed), False)
    buffer = ArmBuffer() if mode == "buffer" else None
    wrist = WristProgress("slow") if evaluation == "tilt" else None
    # 批量运行只输出最终汇总，避免逐次终端消息占用对话上下文。
    with redirect_stdout(StringIO()):
        run("early", .04, .19, True, shape=shape,
            pitch_deg=40. if shape == "box" else 0., palm_x_mm=0.,
            arrival_speed=speed, timing_trial=timing, arm_buffer=buffer,
            evaluation=evaluation, wrist_control=wrist,
            hold_duration_s=2. if evaluation == "hold" else .5)
    return dict(seed=seed, shape=shape, speed=speed, evaluation=evaluation, mode=mode,
        gust_onset_s=timing.onset, gust_duration_s=timing.length, gust_az_m_s2=timing.az,
        **timing.outcome,
        planned_buffer_mm=buffer.planned_mm if buffer else 0.,
        actual_pretilt_buffer_mm=buffer.actual_mm if buffer else 0.,
        buffer_infeasible=int(buffer.infeasible) if buffer else 0,
        max_buffer_command_speed_rad_s=buffer.max_command_speed_rad_s if buffer else 0.)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["shape"], row["speed"], row["evaluation"], row["mode"]].append(row)
    summary, failures = [], []
    for (shape, speed, evaluation, mode), trials in groups.items():
        baseline = {r["seed"]:r for r in groups[shape, speed, evaluation, "fixed"]}
        tags = dict(shape=shape, speed=speed, evaluation=evaluation, mode=mode)
        summary.append(dict(**tags, n=len(trials),
            contacted=sum(r["first_contact_s"] is not None for r in trials),
            succeeded=sum(r["passed"] for r in trials),
            gains=sum(r["passed"] and not baseline[r["seed"]]["passed"] for r in trials),
            losses=sum(not r["passed"] and baseline[r["seed"]]["passed"] for r in trials),
            median_drift_mm=median(r["max_drift_mm"] for r in trials),
            median_buffer_mm=median(r["actual_pretilt_buffer_mm"] for r in trials),
            infeasible=sum(r["buffer_infeasible"] for r in trials)))
        for stage, count in sorted(Counter(r["failure_stage"] for r in trials).items()):
            failures.append(dict(**tags, outcome=stage, count=count))
    return summary, failures


def plot(path, summary):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="590">',
           '<rect width="960" height="590" fill="white"/>',
           '<g font-family="Arial" fill="#18364a">',
           '<text x="25" y="35" font-size="22">Recoil transfer across shapes and arrival speeds</text>',
           '<text x="25" y="61" font-size="13">24 paired gust seeds per condition. Frozen parameters. Blue: fixed. Orange: recoil.</text>',
           '<text x="25" y="83" font-size="13">Hold and tilt are independent replays. Box pitch: 40 degrees. Ball pitch: 0 degrees.</text>']
    for col, evaluation in enumerate(("hold", "tilt")):
        x = 190+col*400
        title = "Hold original pose for 2 s" if evaluation == "hold" else "Tilt 105 degrees + hold 0.5 s"
        svg.append(f'<text x="{x}" y="118" font-size="16">{title}</text>')
        for i, (shape, speed) in enumerate((s, v) for s in ("ball", "box") for v in ("slow", "medium", "fast")):
            y = 156+64*i
            if col == 0:
                svg.append(f'<text x="25" y="{y+12}" font-size="14">{shape.title()} / {speed}</text>')
            for j, mode in enumerate(("fixed", "buffer")):
                row = next(r for r in summary if (r["shape"],r["speed"],r["evaluation"],r["mode"]) == (shape,speed,evaluation,mode))
                width = 225*row["succeeded"]/row["n"]
                color = "#28648d" if mode == "fixed" else "#bd542c"
                svg.append(f'<rect x="{x}" y="{y+21*j}" width="{width}" height="16" fill="{color}"/>')
                svg.append(f'<text x="{x+238}" y="{y+13+21*j}" font-size="13">{row["succeeded"]}/{row["n"]}</text>')
    svg.append('<text x="25" y="575" font-size="13">Success includes no floor contact, continuous required finger contact, and at most 25 mm palm-relative drift.</text></g></svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = [(seed, shape, speed, evaluation, mode)
            for seed in range(544,568) for shape in ("ball", "box")
            for speed in ("slow", "medium", "fast")
            for evaluation in ("hold", "tilt") for mode in ("fixed", "buffer")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(trial, jobs))
    summary, failures = summarize(rows)
    write_rows(args.output_dir/"trials.csv", rows)
    write_rows(args.output_dir/"summary.csv", summary)
    write_rows(args.output_dir/"outcomes.csv", failures)
    plot(args.output_dir/"buffer_generalization_en.svg", summary)
    for r in summary:
        print(r)
