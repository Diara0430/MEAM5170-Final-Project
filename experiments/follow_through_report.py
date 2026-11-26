"""Summarize paired trials and plot reference switches against measured dynamics."""
import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def plot_case(directory, row):
    strategy = row["strategy"]
    prefix = f"seed{row['seed']}_case{row['scenario']}_{strategy}"
    trace = read(directory / f"{prefix}.csv")
    contact = float(row["first_contact_s"])
    ms = 1000 * (np.array([float(r["step_start_s"]) for r in trace]) - contact)
    accel = np.array([[float(r[f"actual_accel_{j}"]) for j in range(1, 8)] for r in trace])
    _, joint = np.unravel_index(np.abs(accel).argmax(), accel.shape)
    number = joint + 1
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
    top.plot(ms, [float(r[f"ref_accel_{number}"]) for r in trace], label="Reference", color="#da8527")
    top.plot(ms, accel[:, joint], label="Measured", color="#21656a")
    for limit in (-4, 4):
        top.axhline(limit, color="#b5443b", linestyle="--", lw=1)
    bottom.plot(ms, [float(r[f"actuator_Nm_{number}"]) for r in trace], color="#414d78")
    for ax in (top, bottom):
        for t, r in zip(ms, trace):
            if int(r["new_reference"]):
                ax.axvline(t, color="0.82", lw=.6, zorder=0)
        ax.axvline(0, color="black", lw=1)
    top.legend(loc="upper left", ncol=2)
    top.set_ylabel("Acceleration (rad/s²)")
    bottom.set_ylabel("Actuator torque (N·m)")
    bottom.set_xlabel("Time relative to first contact (ms)")
    top.set_title(f"{strategy} · seed {row['seed']} · throw {row['scenario']} · joint {number}\n"
                  "Gray lines mark reference changes. Black marks first contact.")
    fig.savefig(directory / f"{prefix}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = read(args.directory / "trials.csv")
    for seed in sorted({r["seed"] for r in rows}):
        groups = {}
        for name in ("headroom_filter", "follow_through"):
            group = [r for r in rows if r["seed"] == seed and r["strategy"] == name]
            groups[name] = {r["scenario"]: r for r in group}
            if not group or len(groups[name]) != len(group):
                raise ValueError(f"Empty group or duplicate throws for {seed}/{name}")
            contacts = [r for r in group if int(r["contacted"])]
            impulse = median(float(r["net_opposing_impulse_20ms_Ns"]) for r in contacts) if contacts else None
            force = median(float(r["peak_net_force_20ms_N"]) for r in contacts) if contacts else None
            replay_error = max(float(r["precontact_replay_velocity_error"]) for r in group)
            print(seed, name, "N", len(group), "contact", len(contacts),
                  "held", sum(int(r["held"]) for r in group),
                  "qualified", sum(int(r["qualified"]) for r in group),
                  "median signed impulse (Ns)", impulse, "median peak force (N)", force,
                  "max replay error (rad/s)", replay_error,
                  "fallbacks", sum(int(r["stop_fallbacks"]) for r in group))
            if replay_error > 1e-7:
                raise ValueError("Ball-removed replay did not reproduce pre-contact joint velocity")
        baseline, candidate = groups.values()
        if baseline.keys() != candidate.keys():
            raise ValueError("Unpaired cases")
        both = [k for k in baseline if int(baseline[k]["contacted"]) and int(candidate[k]["contacted"])]
        for metric in ("peak_net_force_20ms_N", "net_opposing_impulse_20ms_Ns"):
            delta = [float(candidate[k][metric]) - float(baseline[k][metric]) for k in both]
            if delta:
                print(seed, metric, "paired N", len(delta), "candidate lower", sum(x < 0 for x in delta),
                      "median difference", median(delta))
    for row in rows:
        if row["seed"] == "810" and row["scenario"] in ("4", "6") and row["first_contact_s"]:
            plot_case(args.directory, row)


if __name__ == "__main__":
    main()
