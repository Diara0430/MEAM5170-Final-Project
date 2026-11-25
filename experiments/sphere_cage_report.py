"""汇总食指让位与拇指预整形对照，保留自碰、接触连续性和载荷代价。"""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read
from experiments.sphere_cage_ablation import CAGE_VARIANTS


def plot(path,rows,summary,selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    seeds=sorted(set(r["seed"] for r in rows));colors=["#274b77","#2b9b8f","#8565ac","#ce7b35"]
    labels=["Original","Index\nclearance","Late\nthumb","Coordinated\npreshape"]
    fig,axes=plt.subplots(len(seeds),4,figsize=(15,4*len(seeds)),squeeze=False)
    for i,seed in enumerate(seeds):
        group=[r for r in summary if r["seed"]==seed]
        axes[i,0].bar(range(4),[int(r["held"]) for r in group],color=colors)
        axes[i,0].set_ylim(0,max(int(r["n"]) for r in group)+1);axes[i,0].set_title("Held for 2 s, original criterion")
        axes[i,0].set_ylabel(f"Seed {seed}: trials")
        axes[i,1].bar(range(4),[int(r["thumb_bearing"]) for r in group],color=colors)
        axes[i,1].set_ylim(0,max(int(r["n"]) for r in group)+1);axes[i,1].set_title("Thumb load >=0.1 N for >=50 ms*")
        for col,key,title in ((2,"all_phase_thumb_self_penetration_mm","Peak thumb-hand mesh overlap (mm)"),
                              (3,"peak_capture_normal_load_N","Peak summed normal load, 0-300 ms (N)")):
            for j,(name,_) in enumerate(CAGE_VARIANTS):
                values=[float(r[key]) for r in rows if r["seed"]==seed and r["strategy"]==name]
                axes[i,col].scatter(j+np.linspace(-.1,.1,len(values)),values,color=colors[j],s=20,alpha=.7)
                axes[i,col].plot([j-.2,j+.2],[median(values)]*2,"k-",lw=2)
            axes[i,col].set_title(title)
        axes[i,2].set_yscale("symlog",linthresh=.01);axes[i,2].axhline(1.,color="gray",ls=":",lw=1)
        for ax in axes[i]:ax.set_xticks(range(4),labels);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Coordinated finger geometry: clearance must be paired with timely caging",fontsize=15)
    fig.text(.5,.01,"*Cumulative duration in the hold window. Summed normal load includes internal grip forces, not net impact force. Arm-qualified trials remain zero.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(path/"sphere_cage_en.svg");fig.savefig(path/"sphere_cage_en.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for i,(seed,case) in enumerate((("807","1"),("809","5"))):
        for name,color in (("middle_compliant",colors[0]),("thumb_preshape",colors[3])):
            group=selected[(seed,case,name)]
            origin=next((r for r in group if float(r["contact_ms"])>=300.),None)
            if not origin:continue
            ref=np.array([float(origin[f"object_palm_{a}_mm"]) for a in "xyz"])
            t=[float(r["contact_ms"]) for r in group]
            motion=[np.linalg.norm(np.array([float(r[f"object_palm_{a}_mm"]) for a in "xyz"])-ref) if float(r["contact_ms"])>=300 else np.nan for r in group]
            axes[i,0].plot(t,motion,color=color,label=name)
            axes[i,1].plot(t,[float(r["thumb_N"]) for r in group],color=color)
            axes[i,2].plot(t,[int(r["wrench_rank"]) for r in group],color=color)
        axes[i,0].axhline(25.,color="gray",ls=":");axes[i,0].set_ylabel(f"Seed {seed} / case {case}\nHold displacement (mm)")
        axes[i,1].set_ylabel("Actual thumb normal load (N)");axes[i,2].set_ylabel("Scaled contact-map rank")
        for ax in axes[i]:ax.set_xlim(0,2300);ax.grid(alpha=.2);ax.set_xlabel("Time since first contact (ms)")
    axes[0,0].set_title("Position retention");axes[0,1].set_title("Actual load, not the force target")
    axes[0,2].set_title("Rank is a proxy, not force-closure proof");axes[0,0].legend(fontsize=8)
    fig.suptitle("Example captures: earlier thumb placement and a clear index corridor",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(path/"sphere_cage_timelines_en.svg");fig.savefig(path/"sphere_cage_timelines_en.png",dpi=150);plt.close(fig)


def main():
    import csv
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path)
    args=parser.parse_args();path=args.directory
    rows=read(path/"trials.csv");summary=read(path/"summary.csv")
    stats=defaultdict(lambda:dict(peak_capture_normal_load_N=0.,peak_thumb_capture_N=0.,hold_contact_loss_ms=0.,thumb_streak=0.,thumb_longest_ms=0.))
    selected=defaultdict(list)
    with (path/"time_series.csv").open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            key=(row["seed"],row["scenario"],row["mode"]);s=stats[key];age=float(row["contact_ms"])
            if age<300.:
                s["peak_capture_normal_load_N"]=max(s["peak_capture_normal_load_N"],sum(float(row[p+"_N"]) for p in ("palm","finger1","finger2","finger3","thumb")))
                s["peak_thumb_capture_N"]=max(s["peak_thumb_capture_N"],float(row["thumb_N"]))
            if age>=298.-1e-8:
                s["hold_contact_loss_ms"]+=2.*(not int(row["finger_contact"]))
                s["thumb_streak"]=s["thumb_streak"]+2. if float(row["thumb_N"])>=.1 else 0.
                s["thumb_longest_ms"]=max(s["thumb_longest_ms"],s["thumb_streak"])
            if key[0:2] in (("807","1"),("809","5")) and key[2] in ("middle_compliant","thumb_preshape"):selected[key].append(row)
    pairs=[]
    for row in rows:
        row.update(stats[(row["seed"],row["scenario"],row["strategy"])])
        if row["strategy"]=="middle_compliant":continue
        base=next(r for r in rows if r["seed"]==row["seed"] and r["scenario"]==row["scenario"] and r["strategy"]=="middle_compliant")
        pairs.append(dict(seed=row["seed"],scenario=row["scenario"],strategy=row["strategy"],
            held_before=int(base["held"]),held_after=int(row["held"]),held_delta=int(row["held"])-int(base["held"]),
            drift_before_mm=float(base["drift_mm"]),drift_after_mm=float(row["drift_mm"]),drift_delta_mm=float(row["drift_mm"])-float(base["drift_mm"]),
            self_depth_delta_mm=float(row["all_phase_thumb_self_penetration_mm"])-float(base["all_phase_thumb_self_penetration_mm"])))
    write_rows(path/"enriched_trials.csv",rows);write_rows(path/"paired_differences.csv",pairs)
    plot(path,rows,summary,selected)
    lines=["食指让位与拇指包围：验证结果与边界", "",
        "无机器学习。使用已知球体几何、网格距离、雅可比、局部QP和真实有限力矩执行器。",
        "807用于开发检查；早期808实验揭示自碰后，修正食指通道并以未见过的809作为新验证集。最终807/809为96次对照（24条来球×4策略）。",
        "中段柔顺基线保持不变；index_clearance仅改变食指；thumb_contact在此基础上于保持阶段接近；thumb_preshape再提前摆放拇指。",
        "单独食指让位与晚接近可能破坏原有捕获通道，属于消融对照，不推荐单独替换基线。",
        "最终协调策略：食指目标[0,0.8,0.9,0.5] rad、290 ms共享五次进度；拇指来球前向[0.8,0.5,1.0,0.65] rad整形。",
        "拇指预整形400 ms，参考峰值约4.69 rad/s；食指参考峰值约5.17 rad/s，实际速度单独记录。",
        "拇指晚接近使用4变量距离QP、关节/偏移约束、已有自接触线性约束及离散路径网格复核。",
        "触球载荷>=0.10 N持续6 ms后撤去推进速度，以当前姿态保持，并加入0.30 N的轻度法向前馈目标。",
        "0.30 N不是实际接触力硬限制；载荷超过1 N时仅撤去额外法向目标。惯性、位置反馈和其他手指挤压仍可能造成较大实际力。",
        "食指让位与预整形有意改变捕获；仅index_clearance/thumb_contact两组在前300 ms检查状态一致，不能声称全部组前300 ms相同。",
        "保持判定仍为接触300 ms后连续手指接触两秒、无落地、相对漂移<=25 mm；没有放宽阈值。", "",
        "seed | strategy | held | thumb bearing >=50ms* | three parts / rank6 >=50ms* | max thumb-hand overlap over full trial (mm)"]
    for seed in sorted(set(r["seed"] for r in rows)):
        for name,_ in CAGE_VARIANTS:
            group=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            source=next(r for r in summary if r["seed"]==seed and r["strategy"]==name)
            lines.append(f'{seed} | {name} | {source["held"]}/{source["n"]} | {source["thumb_bearing"]}/{source["n"]} | '
                f'{source["three_part_rank6"]}/{source["n"]} | {max(float(r["all_phase_thumb_self_penetration_mm"]) for r in group):.3f}')
    lines.extend(["*前两项50 ms为保持窗口累计时长，最长连续承载另见enriched_trials.csv。秩6仅为局部线性力矩映射可控性代理，不能替代摩擦锥力闭合/动态稳定性证明。", ""])
    for seed in sorted(set(r["seed"] for r in rows)):
        group=[r for r in pairs if r["seed"]==seed and r["strategy"]=="thumb_preshape"]
        lines.append(f'{seed}完整协调版：新增成功{sum(r["held_delta"]>0 for r in group)}条，原成功退步{sum(r["held_delta"]<0 for r in group)}条，逐例漂移差中位数{median(r["drift_delta_mm"] for r in group):+.2f} mm。')
        for name in ("middle_compliant","thumb_preshape"):
            g=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            lines.append(f'  {name}: 首300 ms总法向载荷峰值中位数{median(r["peak_capture_normal_load_N"] for r in g):.2f} N，拇指总法向载荷瞬态最大值{max(r["peak_thumb_capture_N"] for r in g):.2f} N。')
    lines.extend(["", "关键限制：",
        "接触载荷峰值没有全面降低，拇指的瞬态载荷可能增加；不能将保持成功改善等同于柔和接取或易碎物体安全。",
        "总法向载荷包含相互抵消的夹持载荷，不能当作物体净冲量。",
        "修正后完整策略的拇指相关手部网格重叠明显减小；这不等于已验证机器人所有部件的动态无碰撞。",
        "原基线存在约10 mm的食指尖/拇指尖重叠；相关基线模型局限已保留记录，没有通过关闭碰撞来改善结果。",
        "手臂实际加速度超限仍未解决，所有组qualified均为0。仅验证球体、已知状态、重力来球，尚未验证不规则物体、噪声、随机外力或实机。",
        "早期808有自碰的探索性结果不作为最终验证结论；最终以食指通道修正后的807/809数据为准。",
        "参考：DLR2010对预成形、包围和及时关闭逃逸通道的讨论；本项目自行实现，未复现其硬件或成功率。",
        "https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf",
        "网格距离定义：https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance", "",
        "源代码：controllers/sphere_cage.py；experiments/sphere_cage_ablation.py；静态审查：thumb_sphere_reach.py、index_thumb_clearance_audit.py。",
        "复现：python -m experiments.sphere_cage_ablation --output-dir results/<new-folder> --seeds 807 809",
        "报告：python -m experiments.sphere_cage_report results/<folder>",
        "回放：python -m experiments.watch_catch --experiment cage --seed 809 --scenario 5 --speed .25 --loop",
        "另看809/2，完整策略仍未保持；回放红点表示拇指相关>1 mm网格重叠。",
        "下一步优先排查实际手臂加速度超限和接触载荷，再扩展物体形状/随机扰动。"])
    (path/"experiment_notes.txt").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[14:]),flush=True)


if __name__=="__main__":main()
