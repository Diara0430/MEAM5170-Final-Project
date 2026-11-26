"""Compare impact on the very same qualified catches, with identical computation controls."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(directory):
    with (directory/"local_trials.csv").open(encoding="utf-8-sig") as stream:
        rows = {(r["profile"], r["seed"], r["scenario"], r["method"]): r
                for r in csv.DictReader(stream)}
    fig, axes = plt.subplots(1,2,figsize=(9.3,3.6),sharey=True)
    groups = [(p,m) for p in ("normal","noisy") for m in ("short","spot")]
    for ax,field,title,scale in zip(axes,("impact20_peak_N","impulse20_Ns"),
                                    ("Peak resultant force (N)","Opposing impulse (mN·s)"),(1.,1000.)):
        ax.axvline(0,color="#6b777e",lw=1)
        for i,(profile,method) in enumerate(groups):
            delta = []
            for (p,seed,case,name),r in rows.items():
                if (p,name)!=(profile,method):
                    continue
                shadow = rows[(p,seed,case,name+"_shadow")]
                if int(r["qualified"]) and int(shadow["qualified"]):
                    delta.append(scale*(float(r[field])-float(shadow[field])))
            if delta:
                y = i+.14*np.sin(2.4*np.arange(len(delta)))
                ax.scatter(delta,y,s=15,alpha=.48,color="#288580")
                ax.scatter([np.median(delta)], [i], marker="D", color="#172e3a",s=39,zorder=3)
        ax.set_xlabel(title)
        ax.grid(axis="x",alpha=.15)
    axes[0].set_yticks(range(len(groups)),["Standard · short preview","Standard · palm offset",
                                          "Noisier · short preview","Noisier · palm offset"])
    axes[0].invert_yaxis()
    fig.suptitle("Measured change on jointly qualified throws: active rule minus shadow")
    fig.subplots_adjust(left=.23,bottom=.19,wspace=.13,top=.82)
    fig.savefig(directory/"paired_impacts.svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"impact-choice","font.size":9,
                         "axes.spines.top":False,"axes.spines.right":False})
    draw(args.directory)
