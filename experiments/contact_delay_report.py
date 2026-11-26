"""合并固定参数的延迟对照，校验排队时序，输出英文图和中文研究边界。"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from experiments.buffer_ablation import write_rows
from experiments.contact_delay_ablation import DELAY_VARIANTS,DELAY_REPLAY_NAMES
from experiments.finger_closure_ablation import read


def merge(inputs,target):
    target.mkdir(parents=True,exist_ok=False);provenance=[]
    # 小文件按相同列合并；来源保留哈希。8组数据互不重复，不重复计算已有6组。
    for name in ("trials.csv","summary.csv","solver_times.csv","command_pipeline.csv","contact_reference.csv"):
        fields=None
        with (target/name).open("w",newline="",encoding="utf-8-sig") as output:
            writer=None
            for folder in inputs:
                source=folder/name
                if not source.exists():continue
                with source.open(encoding="utf-8-sig",newline="") as stream:
                    reader=csv.DictReader(stream)
                    if writer is None:fields=reader.fieldnames;writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
                    if reader.fieldnames!=fields:raise ValueError(f"Different columns: {source}")
                    count=0
                    for row in reader:writer.writerow(row);count+=1
                provenance.append(dict(source=str(source.resolve().relative_to(ROOT)),rows=count,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    # 按种子拆分逐步CSV，避免单个Git文件过大，仍保留普通CSV供分析。
    for seed in ("810","811"):
        with (target/f"time_series_{seed}.csv").open("w",encoding="utf-8-sig",newline="") as output:
            writer=None
            for folder in inputs:
                source=folder/"time_series.csv"
                with source.open(encoding="utf-8-sig",newline="") as stream:
                    reader=csv.DictReader(stream)
                    if writer is None:writer=csv.DictWriter(output,fieldnames=reader.fieldnames);writer.writeheader()
                    for row in reader:
                        if row["seed"]==seed:writer.writerow(row)
    for folder in inputs:
        source=folder/"time_series.csv"
        provenance.append(dict(source=str(source.resolve().relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (target/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path)
    parser.add_argument("--inputs",type=Path,nargs="+")
    args=parser.parse_args();path=args.directory
    if args.inputs:merge(args.inputs,path)
    rows=read(path/"trials.csv");summary=read(path/"summary.csv")
    assert len(rows)==192 and len({(r["seed"],r["scenario"],r["strategy"]) for r in rows})==192
    assert max(float(r["max_hand_torque_Nm"]) for r in rows)<=.950001
    assert max(float(r["force_fraction"]) for r in rows)<=1.000001
    assert all(int(r["realtime_certified"])==0 for r in rows)
    lookup={(r["seed"],r["scenario"],r["strategy"]):r for r in rows}
    pairs=[]
    for r in rows:
        base=lookup[(r["seed"],r["scenario"],"reference_2ms")]
        pairs.append(dict(seed=r["seed"],scenario=r["scenario"],strategy=r["strategy"],
            held_delta=int(r["held"])-int(base["held"]),qualified_delta=int(r["qualified"])-int(base["qualified"]),
            accel_before=float(base["peak_accel"]),accel_after=float(r["peak_accel"]),
            drift_before_mm=float(base["drift_mm"]),drift_after_mm=float(r["drift_mm"])))
    write_rows(path/"paired_vs_delayed.csv",pairs)
    stats=defaultdict(lambda:dict(n=0,peak=0.,peak_load_300=0.));selected=defaultdict(list);walls=defaultdict(list)
    sources=sorted(path.glob("time_series_*.csv")) or [path/"time_series.csv"]
    for source in sources:
        with source.open(encoding="utf-8-sig",newline="") as stream:
            for r in csv.DictReader(stream):
                key=(r["seed"],r["scenario"],r["strategy"]);trial=lookup[key];s=stats[key]
                delay=round(float(trial["delay_ms"])/2.);step=round((float(r["time_s"])-.002)/.002)
                assert int(r["applied_age_steps"])==(-1 if step<delay else delay)
                assert int(r["applied_generation_step"])==(-1 if step<delay else step-delay)
                s["n"]+=1;s["peak"]=max(s["peak"],float(r["actual_peak_accel"]))
                if 0<=float(r["contact_ms"])<300:s["peak_load_300"]=max(s["peak_load_300"],float(r["summed_normal_N"]))
                walls[(r["seed"],r["strategy"])].append(float(r["controller_wall_ms"]))
                if r["seed"]=="811" and r["scenario"] in ("1","9") and r["strategy"] in DELAY_REPLAY_NAMES:selected[(r["scenario"],r["strategy"])].append(r)
    for key,s in stats.items():
        assert abs(s["peak"]-float(lookup[key]["peak_accel"]))<1e-8
        assert abs(s["peak_load_300"]-float(lookup[key]["peak_normal_first300_N"]))<1e-8
        assert s["n"]==int(lookup[key]["steps"])
    colors=["#315a82","#b54c50","#d88635","#7465a4"]
    labels=["No delay","2 ms\nnaive","2 ms\npredicted","4 ms\npredicted"]
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for i,seed in enumerate(("810","811")):
        group=[next(r for r in summary if r["seed"]==seed and r["strategy"]==name) for name in DELAY_REPLAY_NAMES]
        x=np.arange(4)
        axes[i,0].bar(x-.17,[int(r["held"]) for r in group],width=.32,color=colors,label="Held 2 s")
        axes[i,0].bar(x+.17,[int(r["qualified"]) for r in group],width=.32,facecolor="none",edgecolor=colors,lw=2,label="Motion-qualified")
        axes[i,0].set_ylim(0,13);axes[i,0].set_title("Capture versus original motion criteria");axes[i,0].set_ylabel(f"Seed {seed}: trials")
        if not i:axes[i,0].legend(fontsize=8)
        for j,name in enumerate(DELAY_REPLAY_NAMES):
            g=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            for col,key in ((1,"peak_accel"),(2,"peak_normal_first100ms_N")):
                y=[float(r[key]) for r in g];axes[i,col].scatter(j+np.linspace(-.1,.1,len(y)),y,color=colors[j],s=20)
                axes[i,col].plot([j-.2,j+.2],[np.median(y)]*2,"k-",lw=2)
            y=walls[(seed,name)]
            axes[i,3].plot([j,j],np.percentile(y,[50,99]),color=colors[j],lw=3)
            axes[i,3].scatter(j,np.percentile(y,95),color=colors[j],s=30)
        axes[i,1].axhline(4.,color="gray",ls=":");axes[i,1].set_title("Actual peak joint acceleration (rad/s2)")
        axes[i,2].set_title("Peak summed normal load, 0-100 ms (N)")
        axes[i,3].set_title("Control time: p50-p99 line, p95 dot (ms)");axes[i,3].set_yscale("log")
        axes[i,3].axhline(2.,color="gray",ls=":")
        for ax in axes[i]:ax.set_xticks(x,labels);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Known arm-command latency must be included in the dynamics prediction",fontsize=15)
    fig.text(.5,.008,"192 trials including buffer ablations. Fixed arm delays are charged; fingers update each 2 ms. Runtime overruns are logged, not added to the fixed delay.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(path/"contact_delay_en.svg");fig.savefig(path/"contact_delay_en.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,case in enumerate(("1","9")):
        for name,color,label in zip(DELAY_REPLAY_NAMES,colors,labels):
            g=selected[(case,name)];trial=lookup[("811",case,name)]
            t=[1000.*(float(r["time_s"])-.002-float(trial["first_contact_s"])) for r in g]
            for col,key in ((0,"actual_peak_accel"),(1,"summed_normal_N"),(2,"drift_mm")):
                axes[i,col].plot(t,[float(r[key]) if col!=2 or float(r["contact_ms"])>=298. else np.nan for r in g],color=color,label=label.replace("\n"," "),lw=1.3)
        axes[i,0].axhline(4.,color="gray",ls=":");axes[i,0].set_xlim(-650,400)
        axes[i,1].set_xlim(0,300);axes[i,2].set_xlim(300,2300);axes[i,2].axhline(25.,color="gray",ls=":")
        axes[i,2].set_yscale("symlog",linthresh=25.);axes[i,2].set_ylim(bottom=0.)
        axes[i,0].set_ylabel(f"Seed 811 / case {case}\nActual max joint |a| (rad/s2)")
        axes[i,1].set_ylabel("Summed normal load (N)");axes[i,2].set_ylabel("Hold drift (mm)")
        for ax in axes[i]:ax.set_xlabel("Time since first contact (ms)");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,0].set_title("Include pre-contact acceleration")
    axes[0,1].set_title("Loads remain a separate constraint");axes[0,2].set_title("Keep both recovered and failed cases")
    fig.tight_layout();fig.savefig(path/"contact_delay_timelines_en.svg");fig.savefig(path/"contact_delay_timelines_en.png",dpi=150);plt.close(fig)
    checks=dict(trials=192,arrivals=24,strategies=8,queue_timing_checked_trials=len(stats),
        physics_aggregate_checks=len(stats),baseline_810_historical_matches=12,
        max_hand_torque_Nm=max(float(r["max_hand_torque_Nm"]) for r in rows),
        max_arm_force_fraction=max(float(r["force_fraction"]) for r in rows),
        actual_state_clipping=False,fixed_arm_delay_charged=True,wall_clock_overruns_charged=False,realtime_certified=False)
    (path/"validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    notes=["机械臂指令延迟与接触缓冲：本轮结论", "",
        "无机器学习。810开发，811新检查；参数未根据811结果调整。24条来球×8策略=192次。",
        "第一批为6种缓冲/延迟组合144次；根据810对照补充保留原制动参考的两组预测延迟补偿48次。来源和哈希见provenance.json。",
        "reference_now=上一轮headroom_filter无延迟；reference_2ms=同控制器指令延后一个2 ms物理步交付。",
        "reference_predictive_2ms/4ms保留原制动参考，但先按已知队列预测到交付时刻，再生成该时刻的力矩命令。",
        "measured_now/2ms从实测状态生成无独立关节积分的缓冲参考；predictive_2ms/4ms在此实验参考上加预测补偿。",
        "新缓冲参考没有稳定改善，属于失败对照，不作为推荐默认策略。没有把它的负面结果删掉。", "",
        "延迟如何真正加入仿真：",
        "只对七轴机械臂的ctrl排队；生成于k的命令在k+d生效，此前继续执行旧目标。球体和机器人照常mj_step。",
        "每个物理步均校验applied_generation_step=k-d及applied_age_steps=d，启动时则保留初始命令。",
        "手指仍每2 ms更新；不代表全系统传感、手指、网络、操作系统调度的总延迟测试。",
        "预测从当前状态复制独立MjData，按已知待交付队列传播；尚未知的未来手指ctrl保持当前值。没有读取真实未来轨迹。",
        "预测补偿会有误差：手指继续运动、接触切换、局部QP松弛等仍可能导致实际加速度超限。", "",
        "seed | strategy | held | motion-qualified | accel<=4 | median peak accel | median load (N)"]
    for seed in ("810","811"):
        for name,_,_,_ in DELAY_VARIANTS:
            r=next(r for r in summary if r["seed"]==seed and r["strategy"]==name)
            notes.append(f'{seed} | {name} | {r["held"]}/{r["n"]} | {r["qualified"]}/{r["n"]} | {r["accel_within4"]}/{r["n"]} | {float(r["median_peak_accel"]):.3f} | {float(r["median_peak_normal_N"]):.2f}')
    notes.extend(["", "实际结论：",
        "2 ms未经补偿的机械臂延迟使两组运动约束合格数均为0，即使球仍被抓住，也不能认定动作合格。",
        "保留原制动参考的2 ms预测补偿：810保持5→9、运动约束合格0→8；811保持9→7、运动约束合格0→5。",
        "因此不能宣称预测补偿全面提高保持率；811更明确体现保持率与加速度/关节约束不是同一指标。",
        "4 ms预测补偿仍有退步，811保持5/12、运动约束合格4/12。",
        "811/1：2 ms未经补偿失败，2 ms预测补偿保持并合格；4 ms补偿漂移约30.4 mm，仍失败。",
        "811/9：2 ms预测补偿仍保持，但峰值约4.033 rad/s²，因此严格判为未合格，没有放宽4的阈值。",
        "810/1：无延迟保持，2 ms预测补偿漂移约26.0 mm仍失败，说明补偿没有完全恢复无延迟表现。",
        "实测状态缓冲参考在810出现60.15 rad/s²峰值；消除虚拟状态积分本身不足以保证动态稳定。该设计保留为实验分支。", "",
        "原判据保留：实际速度<=1 rad/s、实际加速度<=4 rad/s²、原关节/执行器限制，以及连续手指接触两秒、无落地、漂移<=25 mm。",
        "计算时间包含本轮机械臂缓冲参考更新和跟踪/预测/滤波；大于注入延迟的步数另行记录。",
        "本轮注入的是固定2/4 ms延迟，未把每一步实际墙钟超时额外加入延迟，也没有模拟单CPU忙碌调度，故realtime_certified仍全部为0。",
        "接触载荷没有全局上限；法向载荷总和含内部夹持力，不能等同于物体净力或冲量。",
        "只测试已知状态的球体重力来球；不规则物体、随机外力、噪声与实机尚未覆盖。", "",
        "下一步：预测未来手指闭合及接触模式，收紧预测误差余量；再将缓冲与力矩QP共同设计，并加入可复现的时延抖动/计算超时调度。", "",
        "数学与API依据：MuJoCo动力学及mj_step/mj_copyData/mj_jacDot，https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html", "",
        "回放：python -m experiments.watch_catch --experiment delay --seed 811 --scenario 1 --speed .25 --loop",
        "按1：无延迟；2：2 ms直接延迟；3：2 ms预测补偿；4：4 ms预测补偿。",
        "另看811/9和810/1，保留未通过案例。",
        "完整复现：python -m experiments.contact_delay_ablation --output-dir results/<new-folder> --baseline-dir results/contact_delay_final --seeds 810 811",
        "报告：python -m experiments.contact_delay_report results/<new-folder>。逐步CSV按种子拆分保存。"])
    (path/"experiment_notes.txt").write_text("\n".join(notes),encoding="utf-8")
    print("\n".join(notes[16:]),flush=True)


if __name__=="__main__":main()
