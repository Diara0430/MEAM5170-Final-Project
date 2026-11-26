"""Plot archived paired throws and the six onset replays from the experiment CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def read(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return int(row["seed"]), int(row["scenario"])


def onset(directory):
    series = read(directory/"onset_frames.csv")
    shots = {key(r):r for r in read(directory/"onset_audit.csv")}
    fig, panels = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, sharey=True)
    for ax, shot in zip(panels.flat, sorted(set(map(key,series)))):
        summary = shots[shot]
        touch = float(summary["first_contact_s"])
        a = [r for r in series if key(r) == shot]
        tick = 1000*(np.array([float(r["accel_step_s"]) for r in a])-touch)
        pos = 1000*(np.array([float(r["frame_s"]) for r in a])-touch)
        ax.axvline(0, color="#445561", lw=1.3)
        if summary["forecast_s"]:
            delta = 1000*(float(summary["forecast_s"])-touch)
            ax.axvline(delta, color="#de882c", lw=1.7, ls="--")
        ax.axhline(4, color="#9c5966", lw=.9, ls=":")
        ax.plot(tick, [abs(float(r["a6"])) for r in a], color="#13777a", lw=2)
        ax.set(xlim=(-20,20), ylim=(0,24), title=f'Seed {shot[0]}, throw {shot[1]}')
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Joint 6 |accel.| (rad/s²)")
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Time from first contact (ms)")
        side = ax.twinx()
        side.plot(pos, [float(r["palm_along_m_s"]) for r in a], color="#79659e", lw=1.25)
        side.set_ylim(-.28,.65)
        if ax.get_subplotspec().is_last_col():
            side.set_ylabel("Palm speed along flight (m/s)")
        else:
            side.tick_params(labelright=False)
    labels = [("Joint 6 acceleration","#13777a","-"), ("Palm speed","#79659e","-"),
              ("Actual contact","#445561","-"), ("Forecast from 10 ms earlier","#de882c","--"),
              ("Acceleration limit","#9c5966",":")]
    fig.legend(handles=[Line2D([],[],color=c,ls=l,label=n) for n,c,l in labels],
               ncol=3, frameon=False, loc="lower center")
    fig.suptitle("Six catches with an acceleration violation after first contact")
    fig.subplots_adjust(left=.08,right=.94,top=.88,bottom=.20,hspace=.32,wspace=.22)
    fig.savefig(directory/"onset_timelines.svg",metadata={"Date":None},bbox_inches="tight")
    plt.close(fig)


def speed(directory):
    rows = read(directory/"validation.csv")
    table = {(key(r),r["policy"]):r for r in rows}
    cases = sorted(set(map(key,rows)))
    names = ["low","mid","high"]
    paired = [k for k in cases if all(int(table[k,p]["accepted"]) and int(table[k,p]["contacted"])
                                    for p in names)]
    fig, axes = plt.subplots(1,3,figsize=(11,3.8))
    for ax, metric, ylabel in zip(axes[:2], ["measured_m_s","impact20_peak_N"],
                                ["Palm speed along flight (m/s)","First 20 ms peak resultant force (N)"]):
        vals = np.array([[float(table[k,p][metric]) for p in names] for k in paired])
        for v in vals:
            ax.plot(range(3),v,color="#65959b",alpha=.28,lw=.9)
        if len(vals):
            ax.plot(range(3),np.median(vals,axis=0),"o-",color="#094d59",lw=2.3)
        ax.set(xticks=range(3),xticklabels=["Low","Middle","High"],ylabel=ylabel,
               xlabel=f"Paired active contacts (n={len(paired)})")
        ax.grid(axis="y",alpha=.2)
    order = ["free30","locked30",*names]
    x = np.arange(len(order))
    for delta, k, label, color in [(-.19,"held","Held for 2 s","#77acb0"),
                                   (.19,"qualified","All limits met","#244a63")]:
        counts = [sum(int(table[i,p][k]) for i in cases) for p in order]
        bars=axes[2].bar(x+delta,counts,width=.37,color=color,label=label)
        axes[2].bar_label(bars,fontsize=8,padding=1)
    axes[2].set(xticks=x,xticklabels=["Rolling\n30%","Fixed\n30%","Low","Middle","High"],
                ylabel=f"Throws (out of {len(cases)})",ylim=(0,len(cases)+3))
    axes[2].legend(frameon=False,fontsize=8,loc="upper center",ncol=2)
    fig.suptitle("Held-out speed sweep at the same interception point and time")
    fig.tight_layout()
    fig.savefig(directory/"fixed_speed_sweep.svg",metadata={"Date":None},bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"fixed-speed","font.size":9})
    onset(args.directory)
    speed(args.directory)
