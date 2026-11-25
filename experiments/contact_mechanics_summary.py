"""从已有诊断CSV生成分组统计、配对案例和英文科学图，不重跑仿真。"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.contact_impulse import write_rows


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def group_summary(rows):
    groups=defaultdict(list)
    for r in rows:groups[int(r["seed"]),r["mode"],int(r["held"])].append(r)
    out=[]
    for (seed,mode,held),group in sorted(groups.items()):
        out.append(dict(seed=seed,mode=mode,held=held,n=len(group),
            never_opposed=sum(not r["first_opposed_ms"] for r in group),
            median_opposed_first300_ms=median(float(r["opposed_first300_ms"]) for r in group),
            median_palm_rotation_deg=median(float(r["max_palm_rotation_first300_deg"]) for r in group),
            median_sideways_mm=median(float(r["max_palm_sideways_first300_mm"]) for r in group),
            median_outgoing_normal_peak_m_s=median(float(r["first300_max_outgoing_normal_m_s"]) for r in group),
            median_first_outgoing_ms=median(float(r["first_outgoing_ms"]) for r in group if r["first_outgoing_ms"] is not None),
            median_first60_outgoing_peak_m_s=median(r["first60_peak"] for r in group),
            median_60to160_outgoing_peak_m_s=median(r["middle_peak"] for r in group),
            median_first160_peak_time_ms=median(r["first160_peak_time"] for r in group),
            median_relative_motion_mm=median(float(r["relative_motion_first300_mm"]) for r in group),
            median_net_hand_impulse100_Ns=median(float(r["net_hand_impulse_first100_Ns"]) for r in group),
            median_gross_normal_integral100_Ns=median(float(r["first100_normal_load_integral_Ns"]) for r in group)))
    return out


def plots(output,rows,curves):
    plt.rcParams.update({"svg.fonttype":"none","font.size":10})
    seeds=sorted(set(int(r["seed"]) for r in rows))
    fig,axes=plt.subplots(len(seeds),3,figsize=(12,3*len(seeds)),squeeze=False,layout="constrained")
    fields=(("opposed_first300_ms","Stable opposing support (ms)",300),
            ("max_palm_rotation_first300_deg","Maximum palm rotation (deg)",None),
            ("first300_max_outgoing_normal_m_s","Peak outward normal speed (m/s)",None))
    for i,seed in enumerate(seeds):
        data=[r for r in rows if int(r["seed"])==seed and r["mode"]=="mpc_match"]
        for j,(field,title,limit) in enumerate(fields):
            ax=axes[i,j]
            for held,color in ((0,"#bd542c"),(1,"#257a50")):
                group=sorted((r for r in data if int(r["held"])==held),key=lambda r:int(r["scenario"]))
                x=held+np.linspace(-.08,.08,len(group))
                ax.scatter(x,[float(r[field]) for r in group],color=color,s=36,edgecolor="white",linewidth=.5)
            ax.set_xticks([0,1],[f'Failed (n={sum(int(r["held"])==0 for r in data)})',f'Held (n={sum(int(r["held"])==1 for r in data)})'])
            ax.set_title(title)
            ax.set_ylim(-.02*(limit or max(float(r[field]) for r in data)),(limit or max(float(r[field]) for r in data))*1.08)
            ax.grid(axis="y",alpha=.2)
            ax.set_ylabel(f'{"Replay" if seed==801 else "Fresh"} seed {seed}')
    fig.suptitle("Early contact mechanics for MPC velocity matching\nEach dot is one trajectory; first 300 ms or until floor contact",fontsize=14)
    fig.savefig(output/"contact_groups_en.svg")
    fig.savefig(output/"contact_groups_en.png",dpi=160)
    plt.close(fig)
    # 取历史组中一组新增失败和一组成功，展示相同时间轴上的支撑和反弹。
    timeline_seed=min(seeds)
    data=[r for r in rows if int(r["seed"])==timeline_seed]
    base={r["scenario"]:r for r in data if r["mode"]=="static_contact"}
    failures=[r for r in data if r["mode"]=="mpc_match" and r["held"]=="0"]
    successes=[r for r in data if r["mode"]=="mpc_match" and r["held"]=="1"]
    if not failures or not successes:
        return
    failed=next((r for r in failures if base[r["scenario"]]["held"]=="1"),failures[0])
    passed=successes[0]
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout="constrained")
    for i,case in enumerate((failed,passed)):
        for mode,color,label in (("static_contact","#28648d","Fixed palm"),("mpc_match","#bd542c","MPC matching")):
            curve=curves[timeline_seed,case["scenario"],mode]
            time=[float(r["contact_ms"]) for r in curve]
            for j,field in enumerate(("object_normal_velocity_m_s","opposed","relative_motion_mm")):
                axes[i,j].plot(time,[float(r[field]) for r in curve],color=color,label=label)
        for j,title in enumerate(("Object velocity along initial palm normal (m/s)","Stable opposing support (0/1)","Palm-frame object displacement (mm)")):
            axes[i,j].set_title(title,fontsize=10)
            axes[i,j].set_xlim(0,300);axes[i,j].grid(alpha=.2)
            axes[i,j].set_xlabel("Time after first contact (ms)")
            axes[i,j].set_ylabel(f'Trajectory {case["scenario"]}: MPC {"held" if case["held"]=="1" else "failed"}')
        axes[i,0].legend(fontsize=8)
    fig.suptitle("Rebound, opposing support and capture motion\nPositive normal velocity points away from the original palm surface",fontsize=13)
    fig.savefig(output/"contact_timelines_en.svg")
    fig.savefig(output/"contact_timelines_en.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs",nargs="+",type=Path,default=[root/"results/mpc_contact_mechanics",root/"results/mpc_contact_mechanics_validation"])
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[];curves=defaultdict(list);first_out={}
    windows=defaultdict(lambda:[(-float("inf"),0.),(-float("inf"),0.)])
    for folder in args.input_dirs:
        data=read(folder/"summary.csv")
        outcomes={(r["scenario"],r["mode"]):r for r in read(folder/"replayed_trials.csv")}
        for r in data:
            r["hold_drift_mm"]=outcomes[r["scenario"],r["mode"]]["drift_mm"] if r["longest_hold_finger_contact_gap_ms"] else None
        rows.extend(data)
        with (folder/"time_series.csv").open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                key=(int(r["seed"]),r["scenario"],r["mode"])
                if float(r["contact_ms"])>=300:continue
                if float(r["object_normal_velocity_m_s"])>.05 and key not in first_out:first_out[key]=float(r["contact_ms"])
                t=float(r["contact_ms"]);value=float(r["object_normal_velocity_m_s"])
                if t<160:
                    slot=0 if t<60 else 1
                    windows[key][slot]=max(windows[key][slot],(value,t))
                if r["mode"] in ("static_contact","mpc_match"):curves[key].append(r)
    for r in rows:
        key=(int(r["seed"]),r["scenario"],r["mode"])
        r["first_outgoing_ms"]=first_out.get(key)
        r["first60_peak"],r["middle_peak"]=windows[key][0][0],windows[key][1][0]
        r["first160_peak_time"]=max(windows[key])[1]
    summary=group_summary(rows)
    pairs=[]
    for seed in sorted(set(r["seed"] for r in rows)):
        base={r["scenario"]:r for r in rows if r["seed"]==seed and r["mode"]=="static_contact"}
        for r in (x for x in rows if x["seed"]==seed and x["mode"]=="mpc_match"):
            b=base[r["scenario"]]
            change="same" if r["held"]==b["held"] else "recovered" if r["held"]=="1" else "lost"
            pairs.append(dict(seed=seed,scenario=r["scenario"],change=change,static_held=b["held"],mpc_held=r["held"],
                static_first_support_ms=b["first_opposed_ms"],mpc_first_support_ms=r["first_opposed_ms"],
                mpc_support_status="held" if r["held"]=="1" else "support_formed_hold_failed" if r["first_opposed_ms"] else "no_support_first300ms",
                mpc_hold_drift_mm=r["hold_drift_mm"],mpc_hold_contact_gap_ms=r["longest_hold_finger_contact_gap_ms"],
                static_turn_deg=b["max_palm_rotation_first300_deg"],mpc_turn_deg=r["max_palm_rotation_first300_deg"],
                static_peak_outgoing_m_s=b["first300_max_outgoing_normal_m_s"],mpc_peak_outgoing_m_s=r["first300_max_outgoing_normal_m_s"]))
    write_rows(args.output_dir/"group_summary.csv",summary)
    write_rows(args.output_dir/"matched_pairs.csv",pairs)
    plots(args.output_dir,rows,curves)
    for r in summary:
        if r["mode"]=="mpc_match":print(r)
