"""Pack the complete contact archive and plot discordant, matched cuboid replays."""
import argparse
import csv
import gzip
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(path, rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outcomes, traces = {}, defaultdict(list)
    def key(r):
        return (r["profile"],int(r["seed"]),int(r["scenario"]),r["method"])
    for folder in sorted(args.archive.glob("trials-*")):
        for row in read(folder/"trials.csv"):
            k = key(row)
            if k in outcomes:
                raise ValueError("Duplicate outcome")
            outcomes[k] = row
        for row in read(folder/"frames.csv"):
            traces[key(row)].append(row)
    if len(outcomes)!=576:
        raise ValueError("Incomplete archived trial set")
    by_profile = defaultdict(list)
    for k, result in outcomes.items():
        path = traces[k]
        if len(path)!=int(result["window_frames"]):
            raise ValueError("Missing contact frames")
        if [float(r["ms"]) for r in path]!=list(range(2,2*len(path)+1,2)):
            raise ValueError("Discontinuous 2 ms grid")
        by_profile[k[0]].extend(path)
    for profile, rows in sorted(by_profile.items()):
        raw = args.output/f"frames_{profile}.csv"
        save(raw, rows)
        with raw.open("rb") as src, raw.with_suffix(".csv.gz").open("wb") as dest:
            with gzip.GzipFile(filename="",fileobj=dest,mode="wb",mtime=0) as zipped:
                shutil.copyfileobj(src,zipped)
        raw.unlink()
    pairs = []
    for k, a in outcomes.items():
        if k[-1] != "pose":
            continue
        alt = (*k[:3],"axis_lever")
        b = outcomes[alt]
        if not (int(a["accepted"]) and int(b["accepted"]) and int(a["contacted"]) and int(b["contacted"])):
            continue
        if int(a["held"])!=int(b["held"]):
            pairs.append((k,alt) if int(a["held"]) else (alt,k))
    times = np.arange(2,101,2)
    fig, axes = plt.subplots(1,3,figsize=(11,3.7))
    labels = ["Relative rotation (deg)","Opposed contacts (%)","Any loaded contact (%)"]
    for i,(name,color) in enumerate((("Held 2 s","#137568"),("Lost","#b24e33"))):
        paths = [traces[pp[i]] for pp in pairs]
        for ax,field,label in zip(axes,("angle_deg","opposed","loaded"),labels):
            vals = [[float(path[j][field]) for path in paths if len(path)>j] for j in range(len(times))]
            y = [np.median(v) if field=="angle_deg" else 100*np.mean(v) for v in vals]
            ax.plot(times,y,label=name,color=color,linewidth=2)
            ax.set(xlabel="Time after first contact (ms)", ylabel=label, xlim=(0,100))
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.18)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Same launch, opposite hold outcomes · {len(pairs)} paired replays")
    fig.tight_layout()
    plt.rcParams.update({"svg.hashsalt":"cuboid-contact-window","svg.fonttype":"none"})
    fig.savefig(args.output/"paired_contact.svg", metadata={"Date":None})
    plt.close(fig)
    # Example at the median of signed between-run travel differences at 100 ms.
    complete = [p for p in pairs if all(len(traces[k])==50 for k in p)]
    ordered = sorted(complete, key=lambda p: (float(traces[p[1]][-1]["travel_mm"])-float(traces[p[0]][-1]["travel_mm"]),p))
    pair = ordered[len(ordered)//2]
    save(args.output/"example_frames.csv",
         [dict(held=int(outcomes[k]["held"]),**r) for k in pair for r in traces[k]])
    print("Complete traces:",sum(map(len,traces.values())),"discordant pairs:",len(pairs),"example:",pair)


if __name__=="__main__":
    main()
