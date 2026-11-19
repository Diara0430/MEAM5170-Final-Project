"""用 ReportLab 从事件与逐步 CSV 生成可缩放的英文曲线图。"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from reportlab.pdfgen.canvas import Canvas


FIXED, FEEDBACK = "#2166ac", "#b54312"
METRICS = (("finger_normal_N", "Finger normal load (N)"),
           ("finger_slip_m_s", "Finger slip speed (m/s)"), ("spin_rad_s", "Object spin (rad/s)"))


def chart(c, left, base, key, series, events):
    width, height = 191, 108
    values = [float(r[key]) for rows in series.values() for r in rows if r[key]]
    peak = max(values, default=1.) or 1.
    step = 10 ** math.floor(math.log10(peak)) / 2
    limit = math.ceil(peak / step) * step
    # 浅底标示原始 100 ms 分析窗口。缺少指面接触时滑速曲线断开。
    c.setFillColor("#eff4f7")
    c.rect(left, base, width / 2, height, fill=1, stroke=0)
    c.setLineWidth(.4)
    c.setFont("Helvetica", 8)
    for frac in (0, .5, 1):
        y = base + height * frac
        c.setStrokeColor("#cbd3d8")
        c.line(left, y, left + width, y)
        c.setFillColor("#435563")
        c.drawRightString(left - 4, y - 2, f"{limit * frac:g}")
    for ms in (0, 50, 100, 150, 200):
        c.drawCentredString(left + width * ms / 200, base - 11, str(ms))
    for name, color in (("fixed", FIXED), ("feedback", FEEDBACK)):
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(.8)
        for field, dash in (("first_slow_ms", [1, 2]), ("first_confirmed_support_ms", [5, 3])):
            event = events[name][field]
            if event and 0 <= float(event) <= 200:
                x = left + width * float(event) / 200
                c.setDash(dash)
                c.line(x, base, x, base + height)
        c.setDash()
        c.setLineWidth(1.25 if name == "fixed" else 1.)
        prev = None
        for row in series[name]:
            if not row[key]:
                prev = None
                continue
            x = left + width * float(row["ms"]) / 200
            y = base + height * float(row[key]) / limit
            if prev is None:
                c.circle(x, y, .65, fill=1, stroke=0)
            else:
                c.line(*prev, x, y)
            prev = x, y


def export(folder, out):
    series, events = defaultdict(dict), defaultdict(dict)
    with (folder / "impact.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            series[int(r["seed"])].setdefault(r["controller"], []).append(r)
    with (folder / "events.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            events[int(r["seed"])][r["controller"]] = r
    seeds = sorted(events)
    if out.exists():
        raise FileExistsError(out)
    c = Canvas(str(out), pagesize=(792, 612))
    c.setTitle("Contact dynamics in paired catches with changed outcomes")
    for start in range(0, len(seeds), 2):
        c.setFont("Helvetica-Bold", 21)
        c.setFillColor("#1d3343")
        c.drawString(38, 573, "Contact dynamics in catches with changed outcomes")
        c.setFont("Helvetica", 10)
        c.drawString(38, 552, "40-degree box  /  fast arrival  /  identical vertical gust within each pair")
        c.setFillColor(FIXED)
        c.drawString(38, 529, "Blue: fixed closure")
        c.setFillColor(FEEDBACK)
        c.drawString(140, 529, "Orange: contact feedback")
        c.setFillColor("#344757")
        c.drawString(272, 529, "Dotted: first slowed command     Dashed: first confirmed opposing support")
        for i, seed in enumerate(seeds[start:start + 2]):
            top = 486 - 214 * i
            fixed, feedback = events[seed]["fixed"], events[seed]["feedback"]
            gained = feedback["passed"] == "1"
            c.setFillColor("#203746")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(38, top, f'Gust {seed:02d}   |   {"Recovered" if gained else "Lost"} with feedback')
            c.setFont("Helvetica", 10)
            loss = fixed if gained else feedback
            stage = loss["failure_stage"].replace("_", " ")
            c.drawString(38, top - 16, f'Failed trial: {stage}.   Peak drift, fixed / feedback: '
                         f'{float(fixed["max_drift_mm"]):.1f} / {float(feedback["max_drift_mm"]):.1f} mm.')
            for j, (key, title) in enumerate(METRICS):
                left = 70 + 245 * j
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor("#203746")
                c.drawString(left, top - 38, title)
                chart(c, left, top - 158, key, series[seed], events[seed])
            c.setFont("Helvetica", 9)
            c.setFillColor("#344757")
            c.drawCentredString(396, top - 185, "Time since first hand contact (ms)")
        c.setFont("Helvetica", 9)
        c.drawString(38, 51, "Shading marks the first 100 ms. Normal load sums all finger contacts. Slip is their mean tangential speed.")
        c.drawString(38, 36, "Slip gaps indicate no loaded finger contact. A support marker requires 6 ms of continuous opposing contact.")
        c.drawRightString(754, 36, f"{start // 2 + 1} / {(len(seeds) + 1) // 2}")
        c.showPage()
    c.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    export(args.folder, args.folder / "contact_dynamics_en.pdf")
