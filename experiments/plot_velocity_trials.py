"""Plot paired impacts from the held-out selection and independent contact-preview trials."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def pairs(path, baseline, variant=None):
    groups = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["policy"], {})[row["seed"], row["scenario"]] = row
    if variant is None:
        variant, = [p for p in groups if p != baseline]
    if groups[baseline].keys() != groups[variant].keys():
        raise ValueError("Both policies must contain the same throws")
    return [(r, groups[variant][k]) for k, r in groups[baseline].items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    runs = [pairs(args.directory / "confirmation.csv", "reference"),
            pairs(args.directory / "contact_preview.csv", "robot_only"),
            pairs(args.directory / "deadline.csv", "fast_robot_only", "fast_projected_contact")]
    configs = [("measured_fraction", 100., "Palm / ball speed (%)"),
               ("peak_contact20_accel", 1., "20 ms joint peak (rad/s²)"),
               ("impulse20_Ns", 1., "20 ms opposing impulse (N·s)")]
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "observed-catches",
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(10.3, 9.8))
    colors = ["#176879", "#d47a30"]
    for col, (metric, scale, title) in enumerate(configs):
        vals = [scale*float(r[metric]) for run in runs for pair in run for r in pair if r[metric] != ""]
        low, high = min(vals+[0.]), max(vals+[0.])
        pad = max((high-low)*.06, .001)
        for i, run in enumerate(runs):
            ax = axes[i, col]
            ax.plot([low-pad, high+pad], [low-pad, high+pad], color="#9cabb2", ls="--", lw=1, zorder=0)
            for a, b in run:
                if a[metric] == "" or b[metric] == "":
                    continue
                held = int(a["held"]) and int(b["held"])
                ax.scatter(scale*float(a[metric]), scale*float(b[metric]), s=28, alpha=.78,
                           facecolors=colors[0] if held else "none",
                           edgecolors=colors[0] if held else colors[1])
            if metric == "peak_contact20_accel":
                ax.axhline(4., color="#aa524c", lw=.8, ls=":")
                ax.axvline(4., color="#aa524c", lw=.8, ls=":")
            ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad), aspect="equal")
            ax.grid(alpha=.13)
            if i == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(("Velocity-aware planner", "Ball-state preview", "Fast preview, 2 ms limit")[i])
            if i == 2:
                ax.set_xlabel("Baseline")
    fig.suptitle("Matched long throws in MuJoCo", fontsize=16)
    handles = [Line2D([], [], color=colors[0], marker="o", lw=0, label="Both held for 2 s"),
               Line2D([], [], color=colors[1], marker="o", markerfacecolor="none", lw=0,
                      label="One or both did not hold"),
               Line2D([], [], color="#9cabb2", ls="--", label="Equal result"),
               Line2D([], [], color="#aa524c", ls=":", label="4 rad/s² limit")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(top=.92, bottom=.09, hspace=.23, wspace=.4)
    fig.savefig(args.directory / "paired_impacts.svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
