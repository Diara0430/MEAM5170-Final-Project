"""汇总中段指节闭合对照的配对变化及使用边界，不重新运行仿真。"""
import argparse
from pathlib import Path
from statistics import median
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from experiments.finger_closure_ablation import FINGER_VARIANTS,read,write_rows


def plot_timelines(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    states=read(path/"finger_states.csv");motion=read(path/"time_series.csv")
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    colors=["#274b77","#2b9b8f","#8565ac","#ce7b35"]
    for row,case in enumerate((3,11)):
        for color,(name,_) in zip(colors,FINGER_VARIANTS):
            group=[r for r in states if r["seed"]=="806" and int(r["scenario"])==case and r["strategy"]==name]
            ball=[r for r in motion if r["seed"]=="806" and int(r["scenario"])==case and r["strategy"]==name]
            # 指节状态为步末，接触速度来自步初，分别保留各自时间戳。
            t=[float(r["contact_ms"])+2. for r in group]
            axes[row,0].plot(t,[np.rad2deg(float(r["q_6_rad"])) for r in group],color=color,label=name)
            axes[row,1].plot(t,[float(r["v_6_rad_s"]) for r in group],color=color)
            axes[row,2].plot([float(r["contact_ms"]) for r in ball],[float(r["normal_velocity_m_s"]) for r in ball],color=color)
        axes[row,0].axhline(np.rad2deg(1.3),color="gray",ls=":",lw=1)
        axes[row,1].axhline(8.5,color="gray",ls=":",lw=1)
        axes[row,0].set_ylabel(f"Seed 806 / case {case}\nMiddle joint 6 angle (deg)")
        axes[row,1].set_ylabel("Middle joint 6 speed (rad/s)")
        axes[row,2].set_ylabel("Outward ball velocity (m/s)")
        for ax in axes[row]:
            ax.set_xlim(0,300);ax.axvspan(60,160,color="#b9d7e8",alpha=.18);ax.grid(alpha=.2)
            ax.set_xlabel("Time since first contact (ms)")
    axes[0,0].set_title("Finger 2: actual middle angle");axes[0,1].set_title("Finger 2: actual middle speed")
    axes[0,2].set_title("Outward ball motion; shaded 60-160 ms")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Joint motion and rebound: a gain case (3) and a regression case (11)",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(path/"finger_timelines_en.svg");fig.savefig(path/"finger_timelines_en.png",dpi=150);plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path)
    args=parser.parse_args();path=args.directory
    rows=read(path/"trials.csv");summary=read(path/"summary.csv");calls=read(path/"solver_times.csv")
    pairs=[]
    for row in rows:
        if row["strategy"]=="original":continue
        base=next(r for r in rows if r["seed"]==row["seed"] and r["scenario"]==row["scenario"] and r["strategy"]=="original")
        pairs.append(dict(seed=row["seed"],scenario=row["scenario"],strategy=row["strategy"],
            held_before=int(base["held"]),held_after=int(row["held"]),held_delta=int(row["held"])-int(base["held"]),
            middle_speed_delta_rad_s=float(row["middle_actual_peak_speed"])-float(base["middle_actual_peak_speed"]),
            middle_angle_delta_deg=float(np.rad2deg(float(row["middle_actual_peak_angle"])-float(base["middle_actual_peak_angle"]))),
            early_rebound_delta_m_s=float(row["rebound_0_60_m_s"])-float(base["rebound_0_60_m_s"]),
            later_rebound_delta_m_s=float(row["rebound_60_160_m_s"])-float(base["rebound_60_160_m_s"])))
    write_rows(path/"paired_differences.csv",pairs)
    lines=["中段指节运动改进：结果与边界", "",
        "目的：验证用户指出的中段关节弯曲过冲、运动速度过快及触球推挤问题。无机器学习。",
        "模型映射：三根普通手指hand/2、6、10（body hand/dip、dip_2、dip_3），拇指同级hand/14。",
        "原控制：所有手指用140 ms smoothstep追踪固定终点，中段目标1.30 rad，峰值目标速度约13.93 rad/s。",
        "改进：中段采用限速度8.5 rad/s、限参考加速度80 rad/s²的梯形速度轨迹，提前减速至零。",
        "保留中段1.30 rad、拇指1.10 rad的终点以维持包围几何；不是把必要的最终弯曲角度一味减小。",
        "用当前模型手指惯性矩阵块、偏置/被动力及PD反馈跟踪中段位置/速度参考；手臂耦合加速度仍属近似误差。",
        "指令反解回原位置执行器，原ctrlrange、±0.95 Nm关节执行器限制及物体质量/摩擦均保留。",
        "原控制的实际峰值约91°大于目标74.5°，所以本轮重点是减少实际过冲角度。",
        "另设预成形：预测穿过捕获平面前200 ms开始，120 ms内轻弯中段0.30 rad（17.2°），其余关节保持张开。",
        "另设柔顺：中段/远端载荷>=0.10 N时，10 ms渐入低刚度6（对向支撑时10）Nm/rad及1.5倍阻尼，40 ms渐出。",
        "柔顺阶段减弱运动前馈，120-240 ms平滑恢复原保持刚度，避免长期过软而丢失夹持。",
        "四组：original；middle_limited（轨迹整形+模型跟踪）；middle_preshape（再加预成形）；middle_compliant（再加触球柔顺）。",
        "同一来球初始条件一致。original/limited及preshape/compliant分别核对首触与应用规划数；预成形有意改变接触前手形。",
        "已有规划时刻复用基线超时决策；如手形改变导致延长规划，额外时刻采用实测超时决定并标记measured_extension。",
        "805是已观察开发集；参数冻结后806首次用于独立验证。此前805小样本试过较低角度/速度，未用于独立验证。",
        "初步把中段终点降到0.95 rad、限速5 rad/s时，4条开发来球中原控制保持3条，新控制保持0条；说明必要的包围行程不可直接去掉。",
        "保持/qualified仍沿用原两秒连续手指接触、无落地、漂移<=25 mm及实际手臂运动约束，没有放宽判定。", "",
        "seed | strategy | held | qualified | median middle peak speed | median middle peak angle | later rebound (60-160 ms)"]
    for row in summary:
        lines.append(f'{row["seed"]} | {row["strategy"]} | {row["held"]}/{row["n"]} | {row["qualified"]}/{row["n"]} | '
            f'{float(row["median_middle_actual_peak_speed"]):.3f} rad/s | {np.rad2deg(float(row["median_middle_actual_peak_angle"])):.2f} deg | '
            f'{float(row["median_later_rebound"]):.3f} m/s')
    lines.extend(["", "同轨迹配对变化（后段速度差<0表示改善，不等同于两个组中位数之差）："])
    for seed in sorted(set(r["seed"] for r in rows)):
        for name,_ in FINGER_VARIANTS[1:]:
            group=[r for r in pairs if r["seed"]==seed and r["strategy"]==name]
            lines.append(f'{seed}/{name}: 保持改善{sum(r["held_delta"]>0 for r in group)}例、退步{sum(r["held_delta"]<0 for r in group)}例；'
                f'后段反弹差中位数{median(r["later_rebound_delta_m_s"] for r in group):+.4f} m/s；'
                f'逐例改善{sum(r["later_rebound_delta_m_s"]< -1e-5 for r in group)}/{len(group)}。')
    lines.extend(["",f'总试验{len(rows)}次（24条来球×4策略），规划时序延长后按实测处理的调用{sum(r["decision_source"]=="measured_extension" for r in calls)}次。',
        "结论：中段实际速度与过冲已降低；后段反弹在部分轨迹降低，但保持率尚未整体提高，并出现成功/失败互换。",
        "首60 ms反弹基本未改变，因此不能把全部反弹归因于该指节运动；首次撞击、抓取几何和后续抗滑仍需处理。",
        "所有组qualified仍为0，之前发现的实际手臂加速度超限不在本轮手指修改的解决范围内。",
        "当前仅球体、已知状态、重力来球；没有验证不规则物体、随机扰动或实机。参考限速也不等于实际速度硬约束。",
        "moving-contact work是中段及远端接触边界运动的正功积分，不是物体净能量，也不保证所有预成形轨迹都降低该值。",
        "理论来源：DLR2010的快速柔顺闭指/预成形/包围，以及Uchiyama2015的接触阶段切换；自行实现，非论文复现或成功率移植。",
        "https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf",
        "https://doi.org/10.1299/mej.14-00074",
        "对照代码：experiments/finger_closure_ablation.py；控制器：controllers/joint_aware_closure.py。",
        "可视化：python -m experiments.watch_catch --experiment finger --seed 806 --scenario 3 --speed .25 --loop",
        "建议同时观察806/3（原控制失败、新控制保持）和806/11（原控制保持、新控制失败），不要只看收益案例。",
        "图表英文；逐例/逐步/指节状态及配对变化见同目录CSV。",
    ])
    (path/"experiment_notes.txt").write_text("\n".join(lines),encoding="utf-8")
    if {"805","806"}.issubset({r["seed"] for r in rows}):plot_timelines(path)
    print("\n".join(lines[19:]),flush=True)


if __name__=="__main__":main()
