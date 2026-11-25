"""生成保持控制的配对报告及历史滚动/摩擦诊断图，不重新运行物理仿真。"""
import argparse
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read


def diagnosis_plot(path,source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=read(source/"time_series.csv")
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True)
    for i,case in enumerate((3,11)):
        for name,color in (("original","#274b77"),("middle_compliant","#ce7b35")):
            group=[r for r in rows if int(r["scenario"])==case and r["mode"]==name and float(r["contact_ms"])>=300]
            t=[float(r["contact_ms"]) for r in group]
            for col,key in enumerate(("hold_motion_mm","friction_utilization_weighted","relative_angular_speed_rad_s","max_slip_mm_s")):
                axes[i,col].plot(t,[float(r[key]) for r in group],color=color,label=name,lw=1.1)
        axes[i,0].set_ylabel(f"Seed 806 / case {case}\nHold displacement (mm)")
        axes[i,1].set_ylabel("Load-weighted friction utilization")
        axes[i,2].set_ylabel("Relative angular speed (rad/s)")
        axes[i,3].set_ylabel("Maximum contact slip (mm/s)");axes[i,3].set_yscale("symlog",linthresh=5)
        axes[i,0].axhline(25,color="gray",ls=":");axes[i,1].axhline(1,color="gray",ls=":")
        for ax in axes[i]:ax.set_xlim(300,2300);ax.grid(alpha=.2);ax.set_xlabel("Time since first contact (ms)")
    for ax,title in zip(axes[0],("Displacement criterion","Friction boundary","Rotation / migration","Slip at material contact points")):ax.set_title(title)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Historical diagnosis: drift can exceed its limit while contact remains",fontsize=14)
    fig.text(.5,.01,"Original/compliant capture controllers, separate from the three-strategy holding-QP experiment. Contact measurements use step-start velocities.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(path/"rolling_friction_diagnosis_en.svg");fig.savefig(path/"rolling_friction_diagnosis_en.png",dpi=150);plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path)
    parser.add_argument("--diagnosis-dir",type=Path,default=ROOT/"results/holding_rolling_diagnosis")
    args=parser.parse_args();path=args.directory
    rows=read(path/"trials.csv");summary=read(path/"summary.csv");control=read(path/"holding_control.csv")
    pairs=[]
    for row in rows:
        if row["strategy"]=="middle_compliant":continue
        base=next(r for r in rows if r["seed"]==row["seed"] and r["scenario"]==row["scenario"] and r["strategy"]=="middle_compliant")
        pairs.append(dict(seed=row["seed"],scenario=row["scenario"],strategy=row["strategy"],
            held_before=int(base["held"]),held_after=int(row["held"]),held_delta=int(row["held"])-int(base["held"]),
            drift_before_mm=float(base["drift_mm"]),drift_after_mm=float(row["drift_mm"]),
            drift_delta_mm=float(row["drift_mm"])-float(base["drift_mm"])))
    write_rows(path/"paired_differences.csv",pairs)
    lines=["保持阶段：接触力、摩擦裕度与力/力矩反馈", "",
        "工作顺序：先推送已有项目至GitHub main，提交0d21ba7da02fdc31e5cf28e6e1785ddb77a102c4，再开展本轮本地工作。",
        "本轮无机器学习，无接触材料/摩擦系数/质量变更，无物体外力注入，原执行器限位及保持判定保留。",
        "24条来球×3策略=72次试验；806是已观察开发集，807在参数冻结后首次用于独立验证。",
        "策略：middle_compliant基线；normal_hold仅高摩擦利用率且明显滑动时响应；wrench_hold还抑制相对位置迁移与转动。",
        "控制在闭指后300 ms且连续对向支撑40 ms后介入；逐条校验首次接触后300 ms的q、v、ctrl与基线一致。",
        "QP每10 ms更新：接触点法向力为正、滑动摩擦锥内接矩形、原手指力矩上限、单次接触力变化上限。",
        "每指与掌面选主要承载点，其他接触视为已测背景力；掌面力变量用于反力耦合，不虚构可独立驱动的掌面执行器。",
        "位置误差15 N/m、速度反馈1.8 Ns/m、相对转动阻尼0.0008 Nm s/rad，力修正上限0.75 N。",
        "接触力修正通过手指雅可比变为力矩修正，经20 ms滤波及有界平衡点适配，额外参考速率<=0.05 rad/s、偏移<=0.08 rad。",
        "12 N是优化目标的法向载荷上限；若基线接触力已高于12 N，允许维持/减小，不要求瞬间骤降，也不是实际力硬限位。",
        "QP不可行时原始力矩增量置零并记录，已有修正滤波衰减且保留限幅平衡点偏移；原±0.95 Nm限位始终生效。",
        "优化使用冻结接触几何/雅可比及被动掌面反力近似；QP可行不等于实际接触力精准实现，也不等于六维力闭合证明。",
        "力、摩擦系数、法向/切向速度读取MuJoCo同一个已解物理步，未加入传感噪声或测量延迟。",
        "", "seed | strategy | held | qualified | median maximum drift (mm) | QP calls/failures"]
    for r in summary:
        lines.append(f'{r["seed"]} | {r["strategy"]} | {r["held"]}/{r["n"]} | {r["qualified"]}/{r["n"]} | '
            f'{float(r["median_drift_mm"]):.3f} | {r["qp_calls"]}/{r["qp_failed"]}')
    lines.extend(["", "逐轨迹配对（负漂移差为改善）："])
    for seed in sorted(set(r["seed"] for r in rows)):
        for name in ("normal_hold","wrench_hold"):
            group=[r for r in pairs if r["seed"]==seed and r["strategy"]==name]
            lines.append(f'{seed}/{name}: 保持改善{sum(r["held_delta"]>0 for r in group)}例、退步{sum(r["held_delta"]<0 for r in group)}例；'
                f'漂移差中位数{median(r["drift_delta_mm"] for r in group):+.4f} mm。')
    calls=[r for r in control if int(r["qp_called"])]
    lines.extend(["",f'保持QP调用{len(calls)}次、回退{sum(int(r["qp_failed"]) for r in calls)}次；'
                  f'墙钟耗时超过10 ms共{sum(float(r["wall_ms"])>10 for r in calls)}次。',
        "同步仿真未将墙钟超时转换成额外控制延迟，不能据此保证实时部署。",
        "结论：新验证组保持成功数均为7/12，尚无成功率提升；力/力矩反馈仅带来很小漂移改善。",
        "806/11基线约40.0 mm，wrench_hold约39.3 mm，仍未满足25 mm阈值；807/1约35.0降至34.6 mm，仍未满足。",
        "这些持续接触案例的失败是保持位置漂移超标，不代表球已经掉地面；不能把全部漂移直接归因于摩擦不足。",
        "原实际手臂加速度超限尚未解决，各组qualified仍为0。首次捕获阶段失败的案例也不会被保持控制补救。", ""])
    if (args.diagnosis_dir/"summary.csv").exists():
        diagnosis=read(args.diagnosis_dir/"summary.csv")
        row=next(r for r in diagnosis if r["scenario"]=="11" and r["strategy"]=="middle_compliant")
        lines.extend(["806/11的完整接触复盘：",
            f'对向支撑时间占比{100.*float(row["opposed_fraction"]):.1f}%，手指几何接触丢失{row["finger_contact_loss_ms"]} ms。',
            f'接近摩擦边界且滑动>20 mm/s的载荷时间积分占比{100.*float(row["near_sliding_load_fraction"]):.5f}%。',
            f'旋转表面速度>=10 mm/s而材料接触点滑动<=5 mm/s的载荷时间积分占比{100.*float(row["rolling_proxy_load_fraction"]):.1f}%。',
            "后者是符合滚动/接触迁移的运动学代理，不是严格因果证明。高摩擦并不自动消除无滑动滚动自由度。",
            f'实际接触对有效滑动摩擦系数为{row["mu1_min"]}到{row["mu1_max"]}；使用运行时接触系数，不直接把物体geom的0.8当作最终接触值。',
            "该复盘的手-物净接触力、重力与物体动量变化平衡误差约1e-14 N量级，验证了力方向和时间对齐。", ""])
        diagnosis_plot(path,args.diagnosis_dir)
    lines.extend(["下一步：优先增加拇指/对侧手指的稳定承载并检查抓取可控方向，形成对滚动更强的几何约束；保持力分配和实际手臂跟踪继续分别验证。",
        "出处：MuJoCo接触与椭圆摩擦锥文档；Amanhoud等RSS2019中的接触运动/力生成及力再平衡思想。",
        "https://mujoco.readthedocs.io/en/stable/computation/index.html#contact",
        "https://www.roboticsproceedings.org/rss15/p21.pdf",
        "未使用论文中学习表面几何的SVR部分；本项目直接采用已知几何与MuJoCo接触点，自行实现QP，并非论文复现。",
        "复现：python -m experiments.holding_control_ablation --output-dir results/<new-folder> --seeds 806 807",
        "报告：python -m experiments.holding_control_report results/<folder>",
        "回放：python -m experiments.watch_catch --experiment holding --seed 806 --scenario 11 --speed .25 --loop",
        "图表英文，原始逐例/逐步数据、接触点10 ms采样及控制QP记录均在同目录CSV。"])
    (path/"experiment_notes.txt").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[15:]),flush=True)


if __name__=="__main__":main()
