"""在配对阵风下改变发射前的掌位，测量首次偏心碰撞和 50 ms 后的旋转。"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.contact_impulse import ImpactWindow, write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed


BASE_MM = 10
OFFSETS_MM = (0, 5, BASE_MM, 15, 20)


def vector(row, field):
    return np.array([row[f"{field}_{axis}"] for axis in "xyz"])


def trial(job):
    split, seed, offset = job
    pulse = gust_for_seed(seed)
    timing = TimingTrial(pulse, online=False)
    window = ImpactWindow(190)

    def observe(model, data, t, first, twist):
        if t - first < .050 - 1e-10:
            window.step(model, data, t, first, twist)

    run("early", .04, .19, True, shape="box", pitch_deg=40., palm_x_mm=offset,
        arrival_speed="fast", timing_trial=timing, after_contact_step=observe)
    lever = impulse_first = impulse_50 = spin = None
    n_first = 0
    if window.samples:
        first, last = window.samples[0], window.samples[-1]
        if not np.isclose(last["elapsed_end_s"], .050):
            raise ValueError("Incomplete 50 ms window")
        if any(not c["other_body"].startswith("hand/") for c in window.contacts):
            raise ValueError("Non-hand contact in the impact window")
        # 首步可能同时碰到多个点。按法向载荷加权，得到质心到各法向作用线的平均垂距。
        points = [c for c in window.contacts if c["step_start_s"] == first["step_start_s"]
                  and c["normal_load_N"] > 0]
        n_first = len(points)
        if points:
            lever = 1000 * sum(np.linalg.norm(np.cross(vector(c, "lever_m"), vector(c, "normal_N")))
                               for c in points) / sum(c["normal_load_N"] for c in points)
        # 先累加含摩擦的接触力矩矢量，再求角冲量大小，单位 mN m s。
        impulse_first = 1000 * np.linalg.norm(vector(first, "angular_impulse_Nms"))
        impulse_50 = 1000 * np.linalg.norm(vector(last, "angular_impulse_Nms"))
        spin = last["spin_rad_s"]
    return dict(split=split, seed=seed, palm_x_mm=offset, gust_onset_s=pulse[0],
                gust_duration_s=pulse[1], gust_az_m_s2=pulse[2], **timing.outcome,
                first_loaded_points=n_first, first_normal_arm_mm=lever,
                first_step_angular_impulse_mN_m_s=impulse_first,
                angular_impulse_50ms_mN_m_s=impulse_50, spin_50ms_rad_s=spin)


def summarize(rows, split):
    baseline = {r["seed"]: r for r in rows if r["palm_x_mm"] == BASE_MM}
    summaries = []
    for offset in sorted({r["palm_x_mm"] for r in rows}):
        group = [r for r in rows if r["palm_x_mm"] == offset]
        changes = [r["passed"] - baseline[r["seed"]]["passed"] for r in group]
        stats = dict(split=split, palm_x_mm=offset, total=len(group),
                     caught=sum(r["passed"] for r in group),
                     dropped_before_tilt=sum(r["failure_stage"] == "dropped_before_tilt" for r in group),
                     gains_vs_10mm=changes.count(1), losses_vs_10mm=changes.count(-1))
        for key in ("first_normal_arm_mm", "first_step_angular_impulse_mN_m_s",
                    "angular_impulse_50ms_mN_m_s", "spin_50ms_rad_s", "max_drift_mm"):
            values = [r[key] for r in group if r[key] is not None]
            stats[f"median_{key}"] = median(values) if values else None
        summaries.append(stats)
    return summaries


def figure(path, summaries):
    panels = (("common_drops", "Previously dropped"), ("selection", "All selection gusts"),
              ("validation", "Fresh validation gusts"))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="910" height="380">',
           '<rect width="910" height="380" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#253745"><text x="28" y="38" font-size="23">'
           'Palm offset and catch success</text><text x="28" y="65" font-size="15">'
           '40-degree box, fast arrival. Offsets follow the local palm x-axis.</text>']
    for j, (label, title) in enumerate(panels):
        left = 28 + 295*j
        svg.append(f'<text x="{left}" y="108" font-size="18">{title}</text>')
        for i, row in enumerate(r for r in summaries if r["split"] == label):
            y = 140 + 39*i
            width = 115 * row["caught"] / row["total"]
            color = "#516d82" if row["palm_x_mm"] == BASE_MM else "#117664"
            svg.extend([f'<text x="{left}" y="{y+13}">{row["palm_x_mm"]} mm</text>',
                        f'<rect x="{left+59}" y="{y}" width="115" height="15" fill="#e4ebef"/>',
                        f'<rect x="{left+59}" y="{y}" width="{width:.2f}" height="15" fill="{color}"/>',
                        f'<text x="{left+185}" y="{y+13}">{row["caught"]}/{row["total"]}</text>'])
    svg.append('<text x="28" y="355" font-size="14">Counts show catches / trials. The first panel is a subset of the second.</text>')
    path.write_text("\n".join([*svg, "</g></svg>"]), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=Path(__file__).resolve().parents[1] /
                        "results/delay_cap/trials.csv")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    with args.prior.open(encoding="utf-8-sig", newline="") as f:
        prior = {}
        for r in csv.DictReader(f):
            if r["split"] == "validation":
                prior.setdefault(int(r["seed"]), {})[r["controller"]] = r
    if set(prior) != set(range(24, 72)) or any(set(g) != {"fixed", "uncapped", "cap_32ms"} for g in prior.values()):
        parser.error("Expected 48 complete paired gusts from the delay-cap experiment")
    failures = {s for s, g in prior.items() if all(r["failure_stage"] == "dropped_before_tilt" for r in g.values())}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(trial, [("selection", s, mm) for s in sorted(prior) for mm in OFFSETS_MM]))
        # 检查 10 mm 对照复现前一轮的结果，再用完整的 48 组选择掌位。
        for r in rows:
            if r["palm_x_mm"] == BASE_MM:
                orig = prior[r["seed"]]["fixed"]
                for key in ("failure_stage", "passed", "max_drift_mm", "first_contact_s", "gust_az_m_s2"):
                    if str(r[key]) != orig[key]:
                        raise ValueError(f"Replay mismatch for seed {r['seed']} at {key}")
        selection = summarize(rows, "selection")
        # 成功次数优先，其次取较少早期掉落，再按漂移中位数和对原掌位的改动排序。
        # 如原掌位最优，同时检验排名最高的替代位置，仍以原掌位为对照。
        ranking = sorted(selection, key=lambda s: (-s["caught"], s["dropped_before_tilt"],
                                                  s["median_max_drift_mm"], abs(s["palm_x_mm"] - BASE_MM)))
        candidate = next(s["palm_x_mm"] for s in ranking if s["palm_x_mm"] != BASE_MM)
        print(f"Selection winner {ranking[0]['palm_x_mm']} mm; locked comparison {BASE_MM} vs {candidate} mm", flush=True)
        fresh = list(pool.map(trial, [("validation", s, mm) for s in range(72, 120) for mm in (BASE_MM, candidate)]))
    for r in rows + fresh:
        r["prior_common_drop"] = int(r["seed"] in failures) if r["split"] == "selection" else None
    stats = summarize([r for r in rows if r["seed"] in failures], "common_drops") + selection + summarize(fresh, "validation")
    write_rows(args.output_dir / "trials.csv", rows + fresh)
    write_rows(args.output_dir / "summary.csv", stats)
    figure(args.output_dir / "palm_offset_en.svg", stats)
