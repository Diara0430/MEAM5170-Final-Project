"""实际加速度与规划余量：英文图表、逐例配对、校验及中文局限说明。"""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from experiments.arm_filter_ablation import ARM_VARIANTS
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path)
    args=parser.parse_args();path=args.directory;rows=read(path/"trials.csv");summary=read(path/"summary.csv")
    assert len(rows)==96 and len({(r["seed"],r["scenario"],r["strategy"]) for r in rows})==96
    assert max(float(r["max_prediction_error"]) for r in rows)<1e-7
    assert max(float(r["max_hand_torque_Nm"]) for r in rows)<=.950001
    assert max(float(r["force_fraction"]) for r in rows)<=1.000001
    assert all(int(r["computation_delay_charged"])==0 and int(r["realtime_certified"])==0 for r in rows)
    pairs=[]
    for row in rows:
        if row["strategy"]=="coordinated":continue
        base=next(r for r in rows if (r["seed"],r["scenario"],r["strategy"])==(row["seed"],row["scenario"],"coordinated"))
        pairs.append(dict(seed=row["seed"],scenario=row["scenario"],strategy=row["strategy"],
            held_before=int(base["held"]),held_after=int(row["held"]),held_delta=int(row["held"])-int(base["held"]),
            qualified_after=int(row["qualified"]),accel_before=float(base["peak_accel"]),accel_after=float(row["peak_accel"]),
            load_before_N=float(base["peak_normal_first100ms_N"]),load_after_N=float(row["peak_normal_first100ms_N"]),
            drift_before_mm=float(base["drift_mm"]),drift_after_mm=float(row["drift_mm"])))
    write_rows(path/"paired_differences.csv",pairs)
    selected=defaultdict(list);walls=defaultdict(list)
    with (path/"time_series.csv").open(encoding="utf-8-sig",newline="") as stream:
        for r in csv.DictReader(stream):
            walls[(r["seed"],r["strategy"])].append(float(r["controller_wall_ms"]))
            if r["seed"]=="810" and r["scenario"] in ("3","6") and r["strategy"] in ("coordinated","headroom_filter"):
                selected[(r["scenario"],r["strategy"])].append(r)
    colors=["#264e78","#42988e","#8966a5","#cf7c38"]
    labels=["Baseline","Filter","Filter +\nadmittance","Reserve +\nfilter"]
    fig,axes=plt.subplots(2,4,figsize=(16,8),squeeze=False)
    for i,seed in enumerate(("809","810")):
        group=[r for r in summary if r["seed"]==seed]
        x=np.arange(4)
        axes[i,0].bar(x-.17,[int(r["held"]) for r in group],width=.32,color=colors,label="Held 2 s")
        axes[i,0].bar(x+.17,[int(r["qualified"]) for r in group],width=.32,facecolor="none",edgecolor=colors,lw=2,label="Motion-qualified")
        axes[i,0].set_ylim(0,13);axes[i,0].set_ylabel(f"Seed {seed}: trials");axes[i,0].set_title("Capture and original motion criteria")
        if i==0:axes[i,0].legend(fontsize=8)
        for j,(name,_,_,_) in enumerate(ARM_VARIANTS):
            g=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            for col,key in ((1,"peak_accel"),(2,"peak_normal_first100ms_N")):
                values=[float(r[key]) for r in g]
                axes[i,col].scatter(j+np.linspace(-.11,.11,len(values)),values,color=colors[j],s=18,alpha=.8)
                axes[i,col].plot([j-.2,j+.2],[median(values)]*2,"k-",lw=2)
            values=walls[(seed,name)]
            axes[i,3].plot([j,j],np.percentile(values,[50,99]),color=colors[j],lw=3)
            axes[i,3].scatter([j],[np.percentile(values,95)],color=colors[j],s=35)
        axes[i,1].set_yscale("log");axes[i,1].axhline(4.,ls=":",color="gray");axes[i,1].set_title("Actual peak joint acceleration (rad/s2)")
        axes[i,2].set_title("Peak summed normal load, first 100 ms (N)")
        axes[i,3].set_yscale("log");axes[i,3].axhline(2.,ls=":",color="gray")
        axes[i,3].set_title("Control time: p50-p99 line, p95 dot (ms)")
        for ax in axes[i]:ax.set_xticks(x,labels,fontsize=9);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Reserve planning authority before correcting actual arm dynamics",fontsize=15)
    fig.text(.5,.008,"96 trials; 809 development, 810 new validation. Computation delays are uncharged. Summed normal load includes internal grip forces.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(path/"arm_filter_en.png",dpi=150);fig.savefig(path/"arm_filter_en.svg");plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,case in enumerate(("3","6")):
        for name,color in (("coordinated",colors[0]),("headroom_filter",colors[3])):
            group=selected[(case,name)];post=[r for r in group if float(r["contact_ms"])>=0.]
            t=[float(r["contact_ms"])+2. for r in post]
            for col,key in ((0,"peak_actual_accel"),(1,"summed_normal_N"),(2,"drift_mm")):
                axes[i,col].plot(t,[float(r[key]) if col!=2 or float(r["contact_ms"])>=298. else np.nan for r in post],color=color,label=name)
        axes[i,0].axhline(4.,color="gray",ls=":");axes[i,2].axhline(25.,color="gray",ls=":")
        axes[i,0].set_ylabel(f"Seed 810 / case {case}\nActual max joint |a| (rad/s2)")
        axes[i,1].set_ylabel("Summed normal load (N)");axes[i,2].set_ylabel("Hold drift (mm)")
        axes[i,2].set_yscale("symlog",linthresh=25.)
        axes[i,2].set_ylim(bottom=0.)
        for ax in axes[i]:ax.set_xlim(0,2300);ax.set_xlabel("Time since first contact (ms)");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,0].set_title("Actual acceleration, including impact")
    axes[0,1].set_title("Loads are not uniformly reduced");axes[0,2].set_title("Strict 25 mm hold threshold")
    fig.suptitle("Keep both a recovered capture (case 3) and a constraint failure (case 6)",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(path/"arm_filter_timelines_en.png",dpi=150);fig.savefig(path/"arm_filter_timelines_en.svg");plt.close(fig)
    checks=dict(trials=96,arrivals=24,baseline_809_historical_matches=12,
        max_prediction_error_rad_s2=max(float(r["max_prediction_error"]) for r in rows),
        max_hand_torque_Nm=max(float(r["max_hand_torque_Nm"]) for r in rows),
        max_arm_force_fraction=max(float(r["force_fraction"]) for r in rows),
        contact_state_matching_filter_vs_admittance=True,thresholds_unchanged=True,computation_delay_charged=False)
    (path/"validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    notes=["机械臂实际加速度与接触缓冲：本轮对照", "",
        "无机器学习；809为开发轨迹，810为控制参数固定后新验证轨迹。24条来球×4策略=96次。",
        "基线为上一轮食指/拇指协调控制；三种新策略均保留该手部控制和原模型、摩擦、碰撞、执行器限制。",
        "acceleration_filter：在原MPC之后，预测实际一步积分加速度，必要时用局部QP修正七轴力矩。",
        "filter_admittance：再加入原25 mm接触导纳，用于检验能否直接组合。仅此组与filter组首触状态逐元素相同。",
        "headroom_filter：MPC规划加速度上限从4收紧至3 rad/s²，给跟踪留余量，再用3.8 rad/s²一步滤波目标。",
        "判据仍是实际|速度|<=1 rad/s、实际|加速度|<=4 rad/s²、关节/执行器限位、两秒连续手指接触、无落地、漂移<=25 mm。",
        "没有把实际qacc或速度裁剪。所有真实状态由原执行器和mj_step积分得到，探针不改真实状态。", "",
        "控制计算：用当前完整状态和当前手指命令，在独立MjData预测下一步，有限差分得到加速度对七轴力矩的局部灵敏度。",
        "QP有7个力矩变量和7个非负松弛变量，目标是少改名义力矩；执行器/控制范围为硬约束，加速度/速度/一步位置限制允许显式松弛。",
        "上一步力矩也作为候选并用真实一步积分复核；接触或静摩擦切换时最多重新线性化3次。失败和剩余超限全部保留。",
        "预测与实际误差很小是因为使用同一已知仿真模型和当前状态，不能当成模型误差或传感噪声鲁棒性证明。", "",
        "seed | strategy | held | motion-qualified | actual accel <=4 | peak normal load median (N)"]
    for r in summary:
        notes.append(f'{r["seed"]} | {r["strategy"]} | {r["held"]}/{r["n"]} | {r["qualified"]}/{r["n"]} | {r["accel_within4"]}/{r["n"]} | {float(r["median_peak_normal_first100_N"]):.2f}')
    for seed in ("809","810"):
        g=[r for r in pairs if r["seed"]==seed and r["strategy"]=="headroom_filter"]
        notes.append(f'{seed}余量+滤波：新增保持{sum(r["held_delta"]>0 for r in g)}条，原成功退步{sum(r["held_delta"]<0 for r in g)}条。')
        s=next(r for r in summary if r["seed"]==seed and r["strategy"]=="headroom_filter")
        times=walls[(seed,"headroom_filter")]
        notes.append(f'  计算p50/p95/p99: {np.percentile(times,[50,95,99]).round(3).tolist()} ms；超2 ms共{s["filter_overruns"]}/{s["filter_steps"]}步。')
    notes.extend(["", "仍未解决：",
        "本轮滤波计算在物理步前同步完成，没有把墙钟耗时和超时造成的通信/执行延迟加入物理仿真；realtime_certified全部为0。运动约束合格不是实时部署合格。",
        "810/6完整策略仍保持，但峰值实际加速度约4.94 rad/s²，未通过运动约束；保留失败，不放宽阈值。",
        "810/4、810/7完整策略仍未保持。仅一组新验证12条来球，不能声称已对任意来球可靠。",
        "直接叠加原导纳并不稳定：809/4出现约126.8 rad/s²实际峰值；说明独立参考发生相互作用，不能把有导纳等同于已经缓冲好。",
        "载荷没有逐例降低：810/6法向载荷峰值从约68.6增加到209.5 N。总法向载荷含内部夹持载荷，不等于物体净力或冲量。",
        "局部QP允许松弛，且接触非线性和3次迭代上限可能导致实际超限；没有全局安全保证。",
        "只测试已知状态、球体和重力来球。暂未覆盖不规则物体、随机外力、噪声和实机。", "",
        "下一步：让MPC与缓冲共用受约束的状态参考，加入实际计算/执行延迟验证，并约束接触净力和载荷，随后再扩展物体。", "",
        "原理依据：MuJoCo动力学 M a + c = tau + J^T f，以及隐式积分和离散状态转移。",
        "https://mujoco.readthedocs.io/en/stable/computation/index.html",
        "https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-copydata", "",
        "复现：python -m experiments.arm_filter_ablation --output-dir results/<new-folder> --seeds 809 810",
        "报告：python -m experiments.arm_filter_report results/<folder>",
        "回放：python -m experiments.watch_catch --experiment arm --seed 810 --scenario 3 --speed .25 --loop",
        "按1看基线，4看余量+滤波；改为--scenario 6看约束仍失败的案例。"])
    (path/"experiment_notes.txt").write_text("\n".join(notes),encoding="utf-8")
    print("\n".join(notes[15:]),flush=True)


if __name__=="__main__":main()
