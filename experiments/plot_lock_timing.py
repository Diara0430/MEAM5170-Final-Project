"""Draw online catch outcomes, realized lock times and contact-free forecast errors."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROFILES = ("normal","noisy","corrected")
TITLES = ("2 mm noise, 20 ms delay","4 mm noise, 40 ms delay","Noisy, 32 mm bias corrected")
COLORS = ("#126e75","#bf682e","#765593")


def read(p):
    with p.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def render(directory):
    rows = read(directory/"validation.csv")
    policies = set(r["policy"] for r in rows)
    clock, = [k for k in policies if k.startswith(("clock","lead"))]
    adaptive, = [k for k in policies if k not in ("rolling",clock) and not k.endswith("_release")]
    order = ("rolling",clock,adaptive,adaptive+"_release")
    fig, axes = plt.subplots(2,3,figsize=(10.5,6.25),sharey="row")
    for i, profile in enumerate(PROFILES):
        ax, bottom = axes[:,i]
        groups = [[r for r in rows if (r["profile"],r["policy"])==(profile,k)] for k in order]
        n, x = len(groups[0]), np.arange(4)
        for offset,key,color,label in [(-.18,"held","#92bcba","Held for 2 s"),(.18,"qualified","#24576b","All limits met")]:
            counts = [sum(int(r[key]) for r in g) for g in groups]
            b = ax.bar(x+offset,counts,.35,color=color,label=label)
            ax.bar_label(b,padding=2,fontsize=8)
        ax.set(title=TITLES[i],xticks=x,xticklabels=["Roll","Timer","Stable","Release"],ylim=(0,n+5))
        if i==0:
            ax.set_ylabel(f"Throws (out of {n})")
        for y,g in enumerate(groups[1:]):
            t = np.array([float(r["lock_s"]) for r in g if r["lock_s"]])
            if t.size:
                jitter = .12*np.sin(np.arange(len(t))*2.4)
                bottom.scatter(t,y+jitter,color=COLORS[i],s=11,alpha=.4)
                bottom.scatter([np.median(t)],[y],s=42,marker="D",color="#182f3b")
            bottom.text(.98,y+.18,f"{len(t)}/{n}",ha="right",fontsize=8,color="#49555a",
                        transform=bottom.get_yaxis_transform())
        bottom.set(yticks=range(3),yticklabels=["Timer","Stable","Release"],ylim=(2.55,-.5),
                   xlim=(.12,1.12),xlabel="Applied lock time after launch (s)")
        bottom.grid(axis="x",alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False)
    fig.suptitle(f"Held-out catch timing: {clock} and {adaptive}")
    fig.subplots_adjust(top=.88,bottom=.13,hspace=.35,wspace=.16)
    fig.savefig(directory/"online_lock_results.svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)

    rows = read(directory/"forecast_by_time.csv")
    fig, axes = plt.subplots(1,2,figsize=(9.8,3.65))
    for i, profile in enumerate(PROFILES):
        g=sorted((r for r in rows if r["profile"]==profile),key=lambda r:float(r["frame_s"]))
        t=[float(r["frame_s"]) for r in g]
        axes[0].plot(t,[float(r["xy_median_mm"]) for r in g],"o-",color=COLORS[i],label=TITLES[i],ms=3)
        axes[1].plot(t,[int(r["both_within"])/int(r["n"])*100 for r in g],"o-",color=COLORS[i],ms=3)
    axes[0].set_ylabel("Median horizontal crossing error (mm)")
    axes[1].set_ylabel("Within 10 mm and 4 ms (%)")
    axes[1].set_ylim(-4,104)
    for ax in axes:
        ax.set_xlabel("Observation time after launch (s)")
        ax.grid(alpha=.18)
    fig.suptitle("Forecasts against the undisturbed crossing of the initial palm plane")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="lower center",frameon=False)
    fig.subplots_adjust(bottom=.24,top=.85,wspace=.29)
    fig.savefig(directory/"forecast_error.svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"catch-lock-time","font.size":9,
                         "axes.spines.top":False,"axes.spines.right":False})
    render(args.directory)
