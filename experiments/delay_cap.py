"""比较逐指闭合延迟上限。先选择参数，锁定后再运行新的配对阵风。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed


CAPS_MS = (4, 8, 16, 32)
CONTROLS = ("fixed", "uncapped")


def trial(job):
    split, seed, setting = job
    cap = setting if isinstance(setting, int) else None
    onset, duration, az = gust_for_seed(seed)
    timing = TimingTrial((onset, duration, az), online=False)
    first_support_ms = None
    peak_delay_ms = 0.

    def observe(model, data, elapsed, closure, velocity):
        nonlocal first_support_ms, peak_delay_ms
        # 观察器也用于固定闭合；支撑确认时间取已完成检测的步末。
        if closure.holding and first_support_ms is None:
            first_support_ms = round(1000 * (elapsed + closure.dt), 6)
        if elapsed <= .200 + 1e-12:
            peak_delay_ms = max(peak_delay_ms, 1000 * max(closure.delay))

    run("early", .04, .19, True, shape="box", pitch_deg=40., palm_x_mm=10.,
        arrival_speed="fast", timing_trial=timing, hand_feedback=setting != "fixed",
        max_delay_ms=cap, contact_observer=observe)
    return dict(split=split, seed=seed, controller=setting if cap is None else f"cap_{cap}ms",
                cap_ms=cap, gust_onset_s=onset, gust_duration_s=duration, gust_az_m_s2=az,
                **timing.outcome, first_support_ms=first_support_ms,
                max_delay_first200_ms=round(peak_delay_ms, 6))


def evaluate(pool, split, seeds, settings):
    # 同一 seed 对应完全相同的扰动，各控制器从独立仿真状态启动。
    jobs = [(split, seed, setting) for seed in seeds for setting in settings]
    rows = list(pool.map(trial, jobs))
    by_controller = {}
    for row in rows:
        by_controller.setdefault(row["controller"], {})[row["seed"]] = row
    summaries = []
    for name, cases in by_controller.items():
        supports = [r["first_support_ms"] for r in cases.values() if r["first_support_ms"] is not None]
        summary = dict(split=split, controller=name, cap_ms=next(iter(cases.values()))["cap_ms"],
                       caught=sum(r["passed"] for r in cases.values()), total=len(cases),
                       median_peak_drift_mm=median(r["max_drift_mm"] for r in cases.values()),
                       supports_confirmed=len(supports), median_support_ms=median(supports) if supports else None)
        for ref in CONTROLS:
            delta = [r["passed"] - by_controller[ref][seed]["passed"] for seed, r in cases.items()]
            summary[f"gains_vs_{ref}"] = delta.count(1)
            summary[f"losses_vs_{ref}"] = delta.count(-1)
        summaries.append(summary)
    return rows, summaries


def draw(path, summaries, selected):
    # 两幅图分别展示候选参数比较与独立验证；文字直接给出成功次数与分母。
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="450" viewBox="0 0 820 450">',
           '<rect width="820" height="450" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#243745">',
           '<text x="30" y="40" font-size="23">Catching a tilted box under vertical gusts</text>',
           '<text x="30" y="67" font-size="15">Each bar shows the fraction of paired trials caught.</text>']
    for left, split, title in ((30, "selection", "Cap selection"), (445, "validation", "Fresh gusts")):
        svg.append(f'<text x="{left}" y="116" font-size="19">{title}</text>')
        for i, row in enumerate(r for r in summaries if r["split"] == split):
            top = 146 + 43 * i
            cap = row["cap_ms"]
            color = "#007769" if cap == selected else "#587186"
            label = row["controller"].capitalize() if cap is None else f"{cap} ms"
            width = 175 * row["caught"] / row["total"]
            svg.extend([f'<text x="{left}" y="{top+14}" font-size="15">{label}</text>',
                        f'<rect x="{left+85}" y="{top}" width="175" height="17" fill="#e5eaec"/>',
                        f'<rect x="{left+85}" y="{top}" width="{width:.2f}" height="17" fill="{color}"/>',
                        f'<text x="{left+275}" y="{top+14}" font-size="15">{row["caught"]}/{row["total"]}</text>'])
    svg.append(f'<text x="30" y="425" font-size="14">Teal marks the {selected} ms cap selected before validation.</text>')
    path.write_text("\n".join([*svg, "</g></svg>"]), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows, stats = evaluate(pool, "selection", range(24), (*CONTROLS, *CAPS_MS))
        # 先按成功数，再按全组峰值漂移中位数，最后取较小上限。验证组不参与选择。
        candidates = (s for s in stats if s["cap_ms"] is not None)
        best = min(candidates, key=lambda s: (-s["caught"], s["median_peak_drift_mm"], s["cap_ms"]))
        selected = best["cap_ms"]
        print(f"Locked delay cap: {selected} ms", flush=True)
        validation, valid_stats = evaluate(pool, "validation", range(24, 72), (*CONTROLS, selected))
    write_rows(args.output_dir / "trials.csv", rows + validation)
    write_rows(args.output_dir / "summary.csv", stats + valid_stats)
    draw(args.output_dir / "delay_cap_en.svg", stats + valid_stats, selected)
