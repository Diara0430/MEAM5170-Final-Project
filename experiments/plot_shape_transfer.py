"""Plot measured early accelerations and held-out cuboid outcomes from persisted CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def draw(directory):
    trials = read(directory/"ball_replay.csv")
    frames = read(directory/"ball_frames.csv")
    boxes = read(directory/"box_confirmation.csv")
    key = lambda r: (r["profile"], r["seed"], r["scenario"])
    keys = {key(r) for r in trials if int(r["held"]) and float(r["peak_contact20_accel"]) > 4.}
    figure, (impact, outcomes) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    traces = []
    for k in sorted(keys):
        path = sorted((r for r in frames if key(r)==k), key=lambda r: float(r["ms"]))
        traces.append([float(r["a6"]) for r in path])
        times = [float(r["ms"]) for r in path]
        impact.plot(times, traces[-1], color="#8ca8b6", alpha=.48, linewidth=1)
    impact.plot(times, np.median(traces,axis=0), color="#005069", linewidth=2.4, label="Median")
    impact.axhline(-4, color="#bc4b34", linewidth=1., linestyle="--", label="Lower limit")
    impact.axhline(4, color="#bc4b34", linewidth=1., linestyle="--")
    impact.set(title=f"Ball · first 20 ms ({len(traces)} catches)", xlabel="Time since contact (ms)",
               ylabel="Joint 6 acceleration (rad/s²)", xticks=[0,4,8,12,16])
    impact.legend(frameon=False, fontsize=8, loc="lower right")
    x = np.arange(3)
    for method, shift, color, name in (("center",-.18,"#8599a1","Center only"),
                                       ("pose",.18,"#147a74","Measured rotation")):
        counts = []
        for profile in ("normal","noisy","corrected"):
            rs = [r for r in boxes if r["profile"]==profile and r["method"]==method and r["shape"]=="box"
                  and r["grasp"]=="original"]
            if len(rs) != 48:
                raise ValueError("Expected 48 held-out throws per condition")
            counts.append(sum(int(r["qualified"]) for r in rs))
        bars = outcomes.bar(x+shift, counts, .32, color=color, label=name)
        outcomes.bar_label(bars, padding=3, fontsize=9)
    outcomes.set(title="Cuboid · held and all seven joints within limits", ylabel="Qualified catches out of 48",
                 xticks=x, xticklabels=["Standard","Noisier","Corrected bias"], ylim=(0,50))
    outcomes.legend(frameon=False, fontsize=8)
    for ax in (impact,outcomes):
        ax.grid(axis="y",alpha=.14)
        ax.set_axisbelow(True)
        ax.spines[["right","top"]].set_visible(False)
    figure.tight_layout(pad=1.3)
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"shape-transfer"})
    figure.savefig(directory/"contact_and_pose.svg",metadata={"Date":None})
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    draw(parser.parse_args().directory)
