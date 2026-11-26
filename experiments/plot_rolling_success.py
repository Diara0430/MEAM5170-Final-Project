"""Plot how far each of the held-out throws progresses under the two rolling policies."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def plot(directory):
    with (directory / "validation.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    candidate, = {r["policy"] for r in rows if r["policy"] != "rolling"}
    colors = ["#176b63", "#91cabb", "#e8aa72", "#d5d9dd"]
    legend = ["Held and within limits", "Held with limit violation",
              "Plan applied, failed hold", "No plan applied"]
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    yticks, labels = [], []
    n = len([r for r in rows if r["profile"]=="normal" and r["policy"]=="rolling"])
    for i,(profile,title) in enumerate([("normal","Standard observations"), ("noisy","Noisier, longer delay"),
                                        ("corrected","Bias corrected mid-flight")]):
        for j,(policy,name) in enumerate([("rolling","Original"),(candidate,"Selected")]):
            group = [r for r in rows if r["profile"]==profile and r["policy"]==policy]
            if len(group) != n:
                raise ValueError("Unbalanced validation data")
            accepted, held, qualified = [sum(int(r[key]) for r in group)
                                         for key in ("accepted","held","qualified")]
            counts = [qualified, held-qualified, accepted-held, n-accepted]
            y = 3*i+j
            left = 0
            for ix,(value,color) in enumerate(zip(counts,colors)):
                ax.barh(y,value,left=left,height=.71,color=color)
                if value >= 3:
                    ax.text(left+value/2,y,str(value),ha="center",va="center",
                            color="white" if ix==0 else "#202c34",fontsize=9)
                left += value
            yticks.append(y)
            labels.append(title+"  ·  "+name)
    ax.set(yticks=yticks, yticklabels=labels, xlim=(0,n), xlabel=f"Throws per policy and condition (of {n})")
    ax.invert_yaxis()
    ax.tick_params(axis="y",length=0,pad=8,labelsize=9)
    for edge in ("top","right","left"):
        ax.spines[edge].set_visible(False)
    fig.legend([Patch(color=c) for c in colors],legend,loc="lower center",ncol=2,frameon=False,fontsize=9)
    ax.set_title("Held-out rolling catches",loc="left",fontweight="bold",pad=10)
    fig.subplots_adjust(left=.32,bottom=.25,top=.91,right=.97)
    fig.savefig(directory/"validation.svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"rolling-catch-success"})
    plot(args.directory)
