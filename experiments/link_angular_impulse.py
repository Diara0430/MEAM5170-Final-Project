"""Analyze the existing 40-degree impact CSVs and plot both trials in English.

Run: python -m experiments.link_angular_impulse --input-dir results/impact_40deg_fast --output-dir results/link_analysis
Dependencies: numpy and reportlab.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


COLORS = {"190": "#00766E", "220": "#AE491D"}
LINKS = {"palm_lower": "Palm", "mcp_joint_2": "MCP 2", "mcp_joint_3": "MCP 3",
         "pip_2": "PIP 2", "pip_3": "PIP 3", "dip_2": "DIP 2", "dip_3": "DIP 3"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def vector(row, name):
    return np.array([float(row[f"{name}_{axis}"]) for axis in "xyz"])


def analyze(case, series, contacts):
    # H at first touch is zero in these zero-spin launches, so delta_H equals H.
    dt = float(series[0]["elapsed_end_s"])
    endpoints = {round(float(r["elapsed_end_s"]), 9): r for r in series}
    first, last = endpoints[.020], endpoints[.050]
    initial = vector(first, "delta_H_Nms")
    axis = initial / np.linalg.norm(initial)
    parts = {}
    for r in contacts:
        elapsed = round(float(r["step_start_s"]) - float(first["first_hand_s"]), 9)
        if not .020 <= elapsed < .050:
            continue
        name = r["other_body"].removeprefix("hand/")
        arm = vector(r, "lever_m")
        normal = np.cross(arm, vector(r, "normal_N"))
        tangent = np.cross(arm, vector(r, "tangent_N"))
        moment = vector(r, "moment_Nm")
        # The 3D contact model in this scene has no independent contact couple.
        np.testing.assert_allclose(moment, normal + tangent, atol=1e-10)
        parts.setdefault(name, np.zeros(3))
        parts[name] += 1000 * dt * np.array([normal @ axis, tangent @ axis, moment @ axis])
    total = sum(parts.values())
    integrated = vector(last, "angular_impulse_Nms") - vector(first, "angular_impulse_Nms")
    np.testing.assert_allclose(total[2], 1000 * integrated @ axis, atol=1e-9)
    direct = 1000 * (vector(last, "delta_H_Nms") - initial) @ axis
    parts["Net"] = total
    return [dict(closure_ms=case, link=link, normal_mNms=v[0], tangential_mNms=v[1],
                 projected_mNms=v[2], state_delta_H_projected_mNms=direct if link == "Net" else "",
                 axis_x=axis[0], axis_y=axis[1], axis_z=axis[2])
            for link, v in parts.items()]


def label(c, x, y, text, size=10, bold=False, color="#203542", align="left"):
    from reportlab.lib.colors import HexColor
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.setFillColor(HexColor(color))
    {"left": c.drawString, "right": c.drawRightString, "center": c.drawCentredString}[align](x, y, text)


def legend(c, y, bars=False):
    from reportlab.lib.colors import HexColor
    for x, case, desc in [(55, "190", "190 ms closure  /  caught"), (290, "220", "220 ms closure  /  dropped")]:
        if bars:
            c.setFillColor(HexColor(COLORS[case]))
            c.rect(x, y, 21, 7, stroke=0, fill=1)
        else:
            c.setStrokeColor(HexColor(COLORS[case]))
            c.setLineWidth(2)
            c.setDash([5, 2] if case == "220" else [])
            c.line(x, y + 3, x + 22, y + 3)
            c.setDash([])
        label(c, x + 28, y, desc)


def motion_figure(path, series):
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen.canvas import Canvas
    c = Canvas(str(path), pagesize=(610, 435))
    c.setTitle("Box motion after first hand contact")
    label(c, 42, 401, "Box motion after first hand contact", 20, True)
    label(c, 42, 380, "40-degree initial pitch   |   Fast arrival   |   10 mm palm displacement")
    legend(c, 353)
    for bottom, field, title, limit, ticks in [
        (206, "speed_m_s", "Center-of-mass speed (m/s)", 2.5, np.arange(0, 2.51, .5)),
        (52, "spin_rad_s", "Angular speed (rad/s)", 40, range(0, 41, 10)),
    ]:
        x, width, height = 65, 505, 104
        label(c, x, bottom + height + 11, title, 11, True)
        for tick in ticks:
            y = bottom + tick * height / limit
            c.setStrokeColor(HexColor("#DCE4E3")); c.setLineWidth(.5)
            c.line(x, y, x + width, y)
            label(c, x - 7, y - 3, f"{tick:g}", 9, align="right")
        for t in range(0, 251, 50):
            label(c, x + t / 250 * width, bottom - 12, str(t), 9, align="center")
        for case, rows in series.items():
            p = c.beginPath()
            for i, r in enumerate(rows):
                px = x + float(r["elapsed_end_s"]) / .25 * width
                py = bottom + float(r[field]) / limit * height
                (p.lineTo if i else p.moveTo)(px, py)
            c.setStrokeColor(HexColor(COLORS[case])); c.setLineWidth(1.6)
            c.setDash([5, 2] if case == "220" else []); c.drawPath(p); c.setDash([])
    label(c, 320, 18, "Time since first hand contact (ms)", 11, align="center")
    c.save()


def impulse_figure(path, rows):
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen.canvas import Canvas
    c = Canvas(str(path), pagesize=(700, 430))
    c.setTitle("Angular impulse by hand link")
    label(c, 38, 394, "Angular impulse by hand link", 20, True)
    label(c, 38, 372, "20-50 ms after first contact   |   Projected onto the angular momentum direction at 20 ms")
    legend(c, 345, bars=True)
    left, scale, zero = 120, 32, 355
    for tick in range(-6, 9, 2):
        x = zero + scale * tick
        c.setStrokeColor(HexColor("#859599" if tick == 0 else "#DEE5E5")); c.setLineWidth(.6)
        c.line(x, 70, x, 318)
        label(c, x, 56, str(tick), 9, align="center")
    values = {(r["link"], r["closure_ms"]): r["projected_mNms"] for r in rows}
    labels = ["palm_lower", "mcp_joint_2", "mcp_joint_3", "pip_2", "pip_3", "dip_2", "dip_3", "Net"]
    for i, link in enumerate(labels):
        y = 303 - 31 * i
        label(c, left - 12, y, LINKS.get(link, link), 11, link == "Net", align="right")
        for case, offset in [("190", 5), ("220", -4)]:
            value = values.get((link, case), 0.)
            c.setFillColor(HexColor(COLORS[case]))
            c.rect(min(zero, zero+scale*value), y+offset, abs(scale*value), 6, stroke=0, fill=1)
    label(c, 350, 34, "Projected angular impulse (mN·m·s)", 11, align="center")
    label(c, 350, 15, "Negative values oppose the reference spin. Numbers identify links in the hand model.", 9, align="center")
    c.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    states = read_csv(args.input_dir / "time_series.csv")
    contacts = read_csv(args.input_dir / "contact_points.csv")
    series = {case: [r for r in states if r["closure_ms"] == case] for case in COLORS}
    rows = []
    for case, samples in series.items():
        rows.extend(analyze(case, samples, [r for r in contacts if r["closure_ms"] == case]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "link_impulses.csv").open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    motion_figure(args.output_dir / "impact_motion_en.pdf", series)
    impulse_figure(args.output_dir / "link_impulses_en.pdf", rows)
