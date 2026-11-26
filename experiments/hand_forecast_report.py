"""核对手指反馈预测、排队时序和保持结果，生成英文图与中文说明。"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read
from experiments.hand_forecast_ablation import HAND_FORECAST_VARIANTS,HAND_REPLAY_NAMES


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path)
    args=parser.parse_args();path=args.directory;rows=read(path/"trials.csv");summary=read(path/"summary.csv")
    assert len(rows)==144 and len({(r["seed"],r["scenario"],r["strategy"]) for r in rows})==144
    lookup={(r["seed"],r["scenario"],r["strategy"]):r for r in rows}
    stats=defaultdict(lambda:dict(n=0,peak=0.,snapshot=0,active_match=0,active_n=0))
    walls=defaultdict(list);selected=defaultdict(list)
    for seed in ("811","812"):
        with (path/f"time_series_{seed}.csv").open(encoding="utf-8-sig",newline="") as stream:
            for r in csv.DictReader(stream):
                key=(r["seed"],r["scenario"],r["strategy"]);s=stats[key];trial=lookup[key]
                step=round((float(r["time_s"])-.002)/.002);delay=round(float(trial["delay_ms"])/2.)
                assert int(r["applied_age_steps"])==(-1 if step<delay else delay)
                assert int(r["applied_generation_step"])==(-1 if step<delay else step-delay)
                assert int(r["snapshot_unchanged"])==1
                assert float(r["hand_peak_torque_Nm"])<=.950001
                s["n"]+=1;s["peak"]=max(s["peak"],float(r["actual_peak_accel"]));s["snapshot"]+=1
                if 0<=float(r["contact_ms"])<300. and float(r["summed_normal_N"])>=.02:
                    s["active_n"]+=1;s["active_match"]+=int(r["contact_mode_match"])
                walls[(seed,r["strategy"])].append(float(r["wall_ms"]))
                if seed=="812" and r["scenario"] in ("3","5") and r["strategy"] in HAND_REPLAY_NAMES:selected[(r["scenario"],r["strategy"])].append(r)
    for key,s in stats.items():
        assert s["n"]==int(lookup[key]["steps"]) and abs(s["peak"]-float(lookup[key]["peak_accel"]))<1e-8
    pairs=[]
    for r in rows:
        base_name="frozen_4ms" if float(r["delay_ms"])==4 else "frozen_2ms"
        base=lookup[(r["seed"],r["scenario"],base_name)]
        pairs.append(dict(seed=r["seed"],scenario=r["scenario"],strategy=r["strategy"],reference_strategy=base_name,
            held_delta=int(r["held"])-int(base["held"]),qualified_delta=int(r["qualified"])-int(base["qualified"]),
            drift_before_mm=float(base["drift_mm"]),drift_after_mm=float(r["drift_mm"]),
            accel_before=float(base["peak_accel"]),accel_after=float(r["peak_accel"])))
    write_rows(path/"paired_vs_frozen.csv",pairs)
    audits=[dict(seed=r["seed"],scenario=r["scenario"],old_held=r["historical_held"],new_held=r["held"],
        old_drift_mm=r["historical_drift_mm"],new_drift_mm=r["drift_mm"],full_match=r["historical_full_match"]) for r in rows if r["historical_full_match"]]
    write_rows(path/"baseline_history_audit.csv",audits)
    modes=[]
    for seed in ("811","812"):
        for name,_,_,_ in HAND_FORECAST_VARIANTS:
            g=[s for key,s in stats.items() if key[0]==seed and key[2]==name]
            total=sum(s["active_n"] for s in g)
            modes.append(dict(seed=seed,strategy=name,loaded_capture_steps=total,
                loaded_capture_mode_match=sum(s["active_match"] for s in g)/max(total,1)))
    write_rows(path/"loaded_contact_mode.csv",modes)
    colors=["#315b83","#6d94ae","#cf7f35","#479783","#8768a3","#ad5c70"]
    labels=["Frozen\n2 ms","Frozen\n3.5 / 2 ms","Hand\n2 ms","Hand\n3.5 / 2 ms","Frozen\n4 ms","Hand\n3.5 / 4 ms"]
    fig,axes=plt.subplots(2,4,figsize=(18,8.5))
    for i,seed in enumerate(("811","812")):
        group=[next(r for r in summary if r["seed"]==seed and r["strategy"]==name) for name,_,_,_ in HAND_FORECAST_VARIANTS]
        x=np.arange(6)
        axes[i,0].bar(x-.17,[int(r["held"]) for r in group],width=.32,color=colors,label="Held 2 s")
        axes[i,0].bar(x+.17,[int(r["qualified"]) for r in group],width=.32,facecolor="none",edgecolor=colors,lw=2,label="Motion-qualified")
        axes[i,0].set_ylim(0,13);axes[i,0].set_ylabel(f"Seed {seed}: trials");axes[i,0].set_title("Holding and original motion criteria")
        if i==0:axes[i,0].legend(fontsize=8)
        for j,(name,_,_,_) in enumerate(HAND_FORECAST_VARIANTS):
            g=[r for r in rows if r["seed"]==seed and r["strategy"]==name];offset=j+np.linspace(-.12,.12,len(g))
            axes[i,1].scatter(offset,[float(r["peak_accel"]) for r in g],color=colors[j],s=18)
            axes[i,2].scatter(offset,[float(r["p95_prediction_error"]) for r in g],color=colors[j],s=16)
            axes[i,2].scatter(offset,[float(r["peak_prediction_error"]) for r in g],color=colors[j],marker="x",s=18,alpha=.7)
            w=walls[(seed,name)];axes[i,3].plot([j,j],np.percentile(w,[50,99]),color=colors[j],lw=3)
            axes[i,3].scatter(j,np.percentile(w,95),color=colors[j],s=25)
        axes[i,1].axhline(4.,color="gray",ls=":");axes[i,1].set_title("Actual peak joint acceleration (rad/s2)")
        axes[i,2].set_yscale("symlog",linthresh=1e-5);axes[i,2].set_title("Acceleration error: p95 dot, peak x (rad/s2)")
        axes[i,3].axhline(2.,color="gray",ls=":");axes[i,3].axhline(4.,color="gray",ls="--")
        axes[i,3].set_yscale("log");axes[i,3].set_title("Control time: p50-p99, p95 dot (ms)")
        for ax in axes[i]:ax.set_xticks(x,labels,fontsize=8);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Forecast hand feedback and contact events before the delayed arm command is delivered",fontsize=15)
    fig.text(.5,.008,"144 trials; 3.5 denotes a tighter filter target, not a changed 4 rad/s2 criterion. Known-model prediction accuracy does not guarantee feasible or real-time control.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(path/"hand_forecast_en.svg");fig.savefig(path/"hand_forecast_en.png",dpi=150);plt.close(fig)
    palette=[colors[0],colors[2],colors[3],colors[5]];legend=["Frozen 2 ms","Hand 2 ms","Hand + reserve 2 ms","Hand + reserve 4 ms"]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,case in enumerate(("3","5")):
        for name,color,label in zip(HAND_REPLAY_NAMES,palette,legend):
            g=selected[(case,name)];trial=lookup[("812",case,name)]
            t=[1000.*(float(r["time_s"])-.002-float(trial["first_contact_s"])) for r in g]
            for col,key in ((0,"actual_peak_accel"),(1,"prediction_error"),(2,"drift_mm")):
                axes[i,col].plot(t,[float(r[key]) if r[key] and (col!=2 or float(r["contact_ms"])>=298.) else np.nan for r in g],color=color,label=label,lw=1.2)
        axes[i,0].axhline(4.,color="gray",ls=":");axes[i,0].set_xlim(-650,350)
        axes[i,1].set_xlim(-650,350);axes[i,1].set_yscale("symlog",linthresh=1e-5)
        axes[i,1].set_ylim(bottom=0.)
        axes[i,2].set_xlim(300,2300);axes[i,2].axhline(25.,color="gray",ls=":");axes[i,2].set_ylim(bottom=0.)
        axes[i,0].set_ylabel(f"Seed 812 / case {case}\nActual max joint |a| (rad/s2)")
        axes[i,1].set_ylabel("Applied acceleration error (rad/s2)");axes[i,2].set_ylabel("Hold drift (mm)")
        for ax in axes[i]:ax.set_xlabel("Time since first contact (ms)");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[0,0].set_title("Actual acceleration")
    axes[0,1].set_title("Prediction error");axes[0,2].set_title("Original hold threshold")
    fig.suptitle("Seed 812: retain the repaired case 3 and the unsafe case 5",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(path/"hand_forecast_timelines_en.svg");fig.savefig(path/"hand_forecast_timelines_en.png",dpi=150);plt.close(fig)
    checks=dict(trials=144,arrivals=24,strategies=6,queue_and_snapshot_checked_steps=sum(s["n"] for s in stats.values()),
        actual_acceleration_aggregates_checked=144,history_prefix_checks=12,
        historical_full_matches=sum(r["full_match"]=="1" for r in audits),historical_comparisons=len(audits),
        actual_state_clipping=False,fixed_arm_delay_charged=True,wall_clock_overruns_charged=False,realtime_certified=False)
    (path/"validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    notes=["手指反馈与接触事件预测：结果与边界", "",
        "无机器学习。811用于开发检查，812为设计固定后的新验证。24条近距离来球×6策略=144次。",
        "保留原制动参考、MPC规划上限3，以及原模型质量、摩擦、碰撞、执行器限制。",
        "冻结基线在机械臂延迟期间保持手指ctrl；联合预测复制当前手指控制状态，传播闭合动作、承载反馈及首触后的闭指/制动事件。",
        "还使用当前已经知道的待生效MPC参考，不读取未来尚未计算的计划、真实未来手指命令或未来状态。",
        "新拇指IK重规划不在短时预测里重新求解，仍可能产生罕见预测误差；真实控制器照常运行并记录规划接收/超时决定。",
        "预测副本逐步检查真实控制器数组、计时、日志长度和探针未被修改，真实手指ctrl也保持不变。",
        "3.5是更紧的滤波目标；原实际加速度判据始终为4 rad/s²，没有降低成功标准。", "",
        "seed | strategy | held | motion-qualified | accel<=4 | median actual peak accel | median p95 acceleration error"]
    for r in summary:notes.append(f'{r["seed"]} | {r["strategy"]} | {r["held"]}/{r["n"]} | {r["qualified"]}/{r["n"]} | {r["accel_within4"]}/{r["n"]} | {float(r["median_peak_accel"]):.3f} | {float(r["median_error_p95"]):.6g}')
    notes.extend(["", "结果解释：",
        "联合预测包含手指反馈、首触事件和已知待生效参考；本轮不能把整个收益归因于单独某一项。",
        "预测误差显著下降是同一已知MuJoCo模型、精确状态、确定性控制的结果，不代表真实硬件或模型误差下仍能如此准确。",
        "812/3：冻结2 ms预测下仍保持，但实际加速度约4.049，联合预测约3.800，修复该运动约束失败。",
        "812/5：联合预测虽准确，实际加速度仍约9 rad/s²，未通过；QP松弛、数值求解失败/迭代上限等问题仍需单独排查。",
        "812/1：不加额外余量的联合预测会从保持变为掉落；更紧余量的2 ms版本又保持，4 ms版本仍失败。不能声称预测更准必然更稳。",
        "3.5余量不是全局保证，例如811/4仍可能接近5 rad/s²；保持率与运动约束必须分别评估。", "",
        "历史复现问题：",
        "旧811/4冻结基线漂移约22.50 mm，本轮相同冻结控制逻辑、同MPC超时决策多次运行约6.11 mm。旧日志没有晚期拇指IK接收决定。",
        "诊断中拒绝全部晚期拇指重规划，漂移约22.498 mm，说明这类墙钟接收决定能显著改变保持阶段；未逐步确认原运行的全部决定。",
        "本轮仅将历史首触与前100 ms载荷作为严格前缀检查，历史保持差异全部写入baseline_history_audit.csv，不假称完整历史匹配。",
        f'历史完整数值匹配{sum(r["full_match"]=="1" for r in audits)}/{len(audits)}；本轮历史保持判定{sum(r["old_held"]==r["new_held"] for r in audits)}/{len(audits)}相同。',
        "各方法比较使用本轮重新运行的冻结基线；所有真实晚期拇指规划决定保存为hand_planner.csv，回放严格重用并再次核对最终指标。",
        "保持率差异还包含晚期重规划及其耗时决定；短时预测误差的对照与完整两秒保持不是同一个因果指标。", "",
        "还未解决：",
        "固定2/4 ms仅加在机械臂ctrl，手指每2 ms更新。墙钟超时未额外加入固定延迟，未模拟单CPU忙碌调度；所有realtime_certified仍为0。",
        "联合预测增加计算开销，2 ms预算可能超时；参见逐步wall_ms与over_budget，不能只看成功率。",
        "关节/速度/力矩/加速度和接触载荷仍需独立核查，预测准确并不消除QP不可行、松弛或实际关节越界。",
        "原保持判据仍为接触300 ms后连续手指接触两秒、无落地、漂移<=25 mm；运动判据为实际速度<=1、加速度<=4及原限位。",
        "仅球体、已知状态、重力来球，尚无不规则形状、噪声、随机外力或实机验证。", "",
        "下一步优先区分接触时QP的真实不可行与数值失败，加入拒绝/降级动作；再考虑计算调度和未知模型误差。", "",
        "回放：python -m experiments.watch_catch --experiment hand --seed 812 --scenario 3 --speed .25 --loop",
        "1冻结2 ms，2联合2 ms，3联合+3.5余量2 ms，4联合+3.5余量4 ms。另看812/5和812/1。",
        "复现：python -m experiments.hand_forecast_ablation --output-dir results/<new-folder> --seeds 811 812",
        "报告：python -m experiments.hand_forecast_report results/<folder>"])
    (path/"experiment_notes.txt").write_text("\n".join(notes),encoding="utf-8")
    print("\n".join(notes[10:]),flush=True)


if __name__=="__main__":main()
