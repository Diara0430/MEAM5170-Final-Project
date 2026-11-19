"""配对复现 0 和 +10 mm 掌位；按掌面、四指分解 0–50 ms 的角冲量。"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from experiments.contact_impulse import ImpactWindow, write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial
from experiments.link_angular_impulse import read_csv, vector


PARTS = ("Palm", "Finger 1", "Finger 2", "Finger 3", "Thumb")
COLORS = ("#313e46", "#966800", "#19786a", "#bd4b31", "#635e99")


def replay(row):
    timing = TimingTrial([float(row[k]) for k in ("gust_onset_s", "gust_duration_s", "gust_az_m_s2")], False)
    window = ImpactWindow(190)
    body_group = {}

    def observe(model, data, t, first, twist):
        if t - first >= .050 - 1e-10:
            return
        if not body_group:
            # 使用控制器的运动树分组；掌面为 0，四指从 1 到 4。
            mapping = ContactClosure(model, LeapHand(model)).group
            body_group.update({model.body(model.geom_bodyid[g]).name: group + 1 for g, group in mapping.items()})
        window.step(model, data, t, first, twist)

    run("early", .04, .19, True, shape="box", pitch_deg=40., palm_x_mm=int(row["palm_x_mm"]),
        arrival_speed="fast", timing_trial=timing, after_contact_step=observe)
    for field in ("passed", "failure_stage", "first_contact_s", "close_s", "max_drift_mm"):
        if str(timing.outcome[field]) != row[field]:
            raise ValueError(f"Replay differs from seed {row['seed']} at {field}")
    if not window.samples or not np.isclose(window.samples[-1]["elapsed_end_s"], .05):
        raise ValueError("Incomplete impact window")
    dt = window.samples[0]["elapsed_end_s"]
    n = len(window.samples)
    # 对每一步的所有接触点积分，保留世界坐标中的方向，避免正反力矩相互抵消前先取绝对值。
    increments = np.zeros((n, len(PARTS), 3))
    onset = [None] * len(PARTS)
    for c in window.contacts:
        i = round((c["step_start_s"] - window.samples[0]["first_hand_s"]) / dt)
        part = body_group[c["other_body"]]
        increments[i, part] += 1000 * dt * vector(c, "moment_Nm")
        if c["normal_load_N"] > 0 and onset[part] is None:
            onset[part] = round(1000 * i * dt, 6)
    impulse = np.cumsum(increments, axis=0)
    net = impulse.sum(axis=1)
    reconstructed = [1000 * vector(s, "angular_impulse_Nms") for s in window.samples]
    np.testing.assert_allclose(net, reconstructed, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(net[-1]), float(row["angular_impulse_50ms_mN_m_s"]), atol=1e-9)
    np.testing.assert_allclose(window.samples[-1]["spin_rad_s"], float(row["spin_50ms_rad_s"]), atol=1e-8)
    return impulse, onset, dt


def analyze_pair(pair):
    r0, r10 = pair
    seed = int(r0["seed"])
    tag = {("1", "1"): "both caught", ("0", "0"): "both failed",
           ("1", "0"): "recovered", ("0", "1"): "lost"}[r0["passed"], r10["passed"]]
    a, onset_a, dt = replay(r0)
    b, onset_b, dt_b = replay(r10)
    if dt != dt_b:
        raise ValueError("Inconsistent time steps")
    net_a, net_b = a.sum(axis=1), b.sum(axis=1)
    norm_a, norm_b = np.linalg.norm(net_a, axis=1), np.linalg.norm(net_b, axis=1)
    # |A|-|B| = (A-B)·(A+B)/(|A|+|B|)。逐点对各部位用同一因子即可精确分解净角冲量大小差。
    # 此因子随时刻和阵风变化；两边皆零时按零处理。
    denominator = (norm_a + norm_b)[:, None]
    factor = np.divide(net_a + net_b, denominator, out=np.zeros_like(net_a), where=denominator > 0)
    effects = np.einsum("tgc,tc->tg", a-b, factor)
    np.testing.assert_allclose(effects.sum(axis=1), norm_a-norm_b, atol=1e-10)
    common = dict(seed=seed, outcome=tag)
    series = []
    for k, part in enumerate(PARTS):
        for i, effect in enumerate(effects[:, k]):
            series.append(dict(**common, elapsed_end_ms=round((i+1)*dt*1000, 6), part=part,
                first_load_0mm_ms=onset_a[k], first_load_10mm_ms=onset_b[k],
                **{f"J_{offset}mm_{axis}_mN_m_s": float(value) for offset, values in ((0,a), (10,b))
                   for axis, value in zip("xyz", values[i,k])}, contribution_mN_m_s=effect))
    summary = dict(**common, failure_stage_0mm=r0["failure_stage"], failure_stage_10mm=r10["failure_stage"],
                   spin_0mm_rad_s=r0["spin_50ms_rad_s"], spin_10mm_rad_s=r10["spin_50ms_rad_s"],
                   net_0mm_mN_m_s=norm_a[-1], net_10mm_mN_m_s=norm_b[-1],
                   delta_mN_m_s=norm_a[-1]-norm_b[-1],
                   **{p.lower().replace(" ", "_")+"_mN_m_s": e for p,e in zip(PARTS,effects[-1])})
    return summary, series


def figure(path, series):
    # 每对阵风先作差，再分组取均值；各曲线是角冲量大小差的可加分量。
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="930" height="420">',
           '<rect width="930" height="420" fill="white"/><g font-family="Arial, sans-serif" fill="#253745">',
           '<text x="30" y="37" font-size="23">Which contacts changed the angular impulse?</text>',
           '<text x="30" y="64" font-size="15">Mean paired contribution, 0 mm minus +10 mm. Negative values reduce net angular impulse at 0 mm.</text>']
    groups = defaultdict(list)
    for r in series:
        for tag in ("all", r["outcome"]):
            groups[tag, r["part"], r["elapsed_end_ms"]].append(r["contribution_mN_m_s"])
    times = sorted({r["elapsed_end_ms"] for r in series})
    tags = ("all", "recovered", "lost")
    means = {(tag, p): [np.mean(groups[tag,p,t]) for t in times] for tag in tags for p in PARTS}
    for tag in tags:
        means[tag, "Net"] = np.sum([means[tag, p] for p in PARTS], axis=0)
    scale = max(.1, np.ceil(10 * max(abs(v) for vs in means.values() for v in vs)) / 10)
    for j, tag in enumerate(tags):
        left, width, mid = 55 + j*305, 221, 220
        count = len(groups[tag, "Palm", times[-1]])
        svg.append(f'<text x="{left}" y="112" font-size="17">{tag.capitalize()} pairs ({count})</text>')
        for value in (-scale, 0., scale):
            y = mid - 80*value/scale
            svg.extend([f'<path d="M{left} {y}h{width}" stroke="#ccd4da"/>',
                        f'<text x="{left-5}" y="{y+4}" font-size="12" text-anchor="end">{value:g}</text>'])
        for t in (0, 10, 20, 30, 40, 50):
            svg.append(f'<text x="{left+width*t/50}" y="321" font-size="12" text-anchor="middle">{t}</text>')
        for part, color in zip((*PARTS, "Net"), (*COLORS, "#111111")):
            points = [(left,mid)]+[(left+width*t/50, mid-80*y/scale) for t,y in zip(times,means[tag,part])]
            coords = " ".join(f"{x:.2f},{y:.2f}" for x,y in points)
            dash = ' stroke-dasharray="4 3"' if part == "Net" else ""
            svg.append(f'<polyline points="{coords}" stroke="{color}" fill="none" stroke-width="2"{dash}/>')
    svg.append('<text x="465" y="349" font-size="14" text-anchor="middle">Time since first hand contact (ms)   |   Vertical axis: mN m s</text>')
    for i,(part,color) in enumerate(zip((*PARTS, "Net"), (*COLORS, "#111111"))):
        x = 85 + i*132
        dash = ' stroke-dasharray="4 3"' if part == "Net" else ""
        svg.append(f'<path d="M{x} 384h22" stroke="{color}" stroke-width="3"{dash}/><text x="{x+28}" y="389">{part}</text>')
    path.write_text("\n".join([*svg, "</g></svg>"]), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=Path(__file__).resolve().parents[1] /
                        "results/palm_offset_gusts/trials.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    pairs = defaultdict(dict)
    for row in read_csv(args.prior):
        if row["split"] == "validation":
            group = pairs[int(row["seed"])]
            offset = int(row["palm_x_mm"])
            if offset in group:
                parser.error("Duplicate replay condition")
            group[offset] = row
    if args.workers < 1 or set(pairs) != set(range(72,120)) or any(set(g) != {0,10} for g in pairs.values()):
        parser.error("Expected 48 complete validation pairs and a positive worker count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze_pair, [(g[0], g[10]) for g in pairs.values()]))
    traces = [row for _, records in results for row in records]
    write_rows(args.output_dir / "pairs.csv", [summary for summary, _ in results])
    write_rows(args.output_dir / "contact_sources.csv", traces)
    cohorts = []
    for tag in ("all", "recovered", "lost", "both caught", "both failed"):
        endpoints = [r for r in traces if r["elapsed_end_ms"] == 50 and (tag == "all" or r["outcome"] == tag)]
        for part in PARTS:
            selected = [r for r in endpoints if r["part"] == part]
            if not selected:
                continue
            loads = [[r[k] for r in selected if r[k] is not None]
                     for k in ("first_load_0mm_ms", "first_load_10mm_ms")]
            cohorts.append(dict(outcome=tag, part=part, pairs=len(selected),
                loaded_0mm=len(loads[0]), loaded_10mm=len(loads[1]),
                median_first_load_0mm_ms=np.median(loads[0]) if loads[0] else None,
                median_first_load_10mm_ms=np.median(loads[1]) if loads[1] else None,
                mean_contribution_50ms_mN_m_s=np.mean([r["contribution_mN_m_s"] for r in selected])))
    write_rows(args.output_dir / "summary.csv", cohorts)
    figure(args.output_dir / "contact_sources_en.svg", traces)
