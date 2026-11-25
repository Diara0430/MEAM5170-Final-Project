"""从缓冲配对实验 CSV 生成可审核摘要，不重新运行仿真。"""
import argparse
import csv
import sys
from pathlib import Path
from statistics import median


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def write(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path);args=parser.parse_args();path=args.directory
    rows=read(path/"trials.csv");summary=read(path/"summary.csv");controls=read(path/"buffer_control.csv")
    seeds=sorted(set(r["seed"] for r in rows));pairs=[]
    for seed in seeds:
        cases=sorted(set(r["scenario"] for r in rows if r["seed"]==seed),key=int)
        for case in cases:
            group={r["strategy"]:r for r in rows if r["seed"]==seed and r["scenario"]==case}
            base=group["brake_140"]
            for name,row in group.items():
                if name=="brake_140":continue
                pairs.append(dict(seed=seed,scenario=case,strategy=name,baseline_held=base["held"],held=row["held"],
                    held_delta=int(row["held"])-int(base["held"]),
                    early_rebound_delta_m_s=float(row["rebound_0_60_m_s"])-float(base["rebound_0_60_m_s"]),
                    later_rebound_delta_m_s=float(row["rebound_60_160_m_s"])-float(base["rebound_60_160_m_s"]),
                    peak_force_delta_N=float(row["peak_normal_first100ms_N"])-float(base["peak_normal_first100ms_N"])))
    write(path/"paired_differences.csv",pairs)
    sources=[
        dict(title="Adaptive control of a robotic manipulator for soft catching of a falling object",year=2015,
             url="https://doi.org/10.1299/mej.14-00074",idea="Pre-contact velocity matching and contact-triggered position-based impedance",
             used="Fixed-parameter admittance after contact; existing 20% velocity-target MPC before contact",
             limitation="Architecture inspiration only; no reproduction of the adaptive feedforward controller; abstract consulted"),
        dict(title="Optimal Passive Dynamics for Physical Interaction: Catching a Mass",year=2013,
             url="https://mime.engineering.oregonstate.edu/research/drl/_documents/kemper_2013a.pdf",
             idea="Passive compliance and damping under force/stroke/inertia constraints; avoid contact loss and rebound",
             used="Explicit 25 mm virtual stroke and damping; separate actuator/reference limits from actual impact",
             limitation="Our software admittance is not the paper's passive spring-damper mechanism"),
        dict(title="Kinematically Optimal Catching a Flying Ball with a Hand-Arm-System",year=2010,
             url="https://elib.dlr.de/67146/1/B%C3%A4uml2010.pdf",
             idea="Hand preshape/caging, compliant fast closure, joint/workspace limits and impact-control bandwidth",
             used="Separate buffering and closure-duration factors; preserve existing grasp criteria",
             limitation="Different hand and launch conditions; no imported robot code or replicated success rate"),
    ]
    write(path/"literature.csv",sources)
    lines=["接触后缓冲与闭指时长配对实验", "",
        "目的：区分腕部让步与手指闭合对反弹、对向支撑及两秒保持的影响。所有控制均为力学/模型控制，无机器学习。",
        "固定参数：Md=0.5 kg，Dd=12 N s/m，Kd=0；虚拟行程25 mm，虚拟速度0.20 m/s，虚拟加速度2 m/s²。",
        "前250 ms接受顺来球方向的净手部接触力；此后撤去力驱动，由阻尼衰减，不主动弹回原点。",
        "掌部任务通过7变量QP转换为加速度参考，关节参考速度<=1 rad/s、加速度<=4 rad/s²，含位置/刹停距离及冻结模型力矩约束。",
        "参考位姿通过原有有力矩限制的执行器跟踪；25 mm是虚拟目标行程，不保证实际掌部运动也<=25 mm。",
        "参考动力学已隔离球体接触；真实碰撞加速度/力矩/关节边界另外计入qualified。",
        "比较：brake_140、admittance_140、brake_220、admittance_220。接触前MPC及其超时决策在同一发射内一致。",
        "原基线采用实测规划时间；其余组回放同一规划是否超时的决策。首触时间/首触速度/应用规划次数逐条一致性校验。",
        f"本次轨迹种子：{', '.join(seeds)}。803用于开发检查，804用于早期实现检查；修正离散刹停保护后805首次用于独立检查。每个完整种子12条近距球体发射。",
        "保持成功：首次接触300 ms后连续手指接触并保持2 s，无落地，掌坐标系漂移<=25 mm。qualified另检查实际关节及力矩限制。",
        "反弹指标：物体速度投影到首次接触时向外的掌面法向；窗口峰值含初次/后续接触，不是材料恢复系数。",
        "",
        "seed | strategy | held | qualified | support formed | median early rebound | median later rebound | median peak normal load",
    ]
    for r in summary:
        lines.append(f'{r["seed"]} | {r["strategy"]} | {r["held"]}/{r["n"]} | {r["qualified"]}/{r["n"]} | '
                     f'{r["support_formed"]}/{r["n"]} | {float(r["median_rebound_0_60"]):.3f} m/s | '
                     f'{float(r["median_rebound_60_160"]):.3f} m/s | {float(r["median_peak_normal_N"]):.1f} N')
    lines.extend(["", "同轨迹相对于brake_140的配对变化（负速度差表示改善）："])
    for seed in seeds:
        for name in ("admittance_140","brake_220","admittance_220"):
            group=[r for r in pairs if r["seed"]==seed and r["strategy"]==name]
            wins=sum(r["held_delta"]>0 for r in group);losses=sum(r["held_delta"]<0 for r in group)
            lines.append(f'{seed} {name}: 保持改善{wins}例、退步{losses}例；早期反弹差中位数'
                         f'{median(r["early_rebound_delta_m_s"] for r in group):+.3f} m/s，后段差'
                         f'{median(r["later_rebound_delta_m_s"] for r in group):+.3f} m/s。')
    wall=[float(r["wall_ms"]) for r in controls]
    lines.extend(["",f'缓冲QP参考更新共{len(wall)}次；耗时中位数{median(wall):.3f} ms，'
                  f'超过2 ms的更新{sum(t>2 for t in wall)}次；不可行/未解出回退{sum(int(r["qp_failed"]) for r in controls)}次。',
        "计时为Python进程墙钟耗时，仿真每步仍同步执行更新；未把超时作为额外控制延迟注入，不能据此宣称500 Hz实时部署成立。",
        "静态掌面基线此前更强，本实验只隔离MPC接触后的缓冲变化，不声称整体最优。",
        "仅球体、已知状态和重力轨迹；未覆盖不规则物体、感知噪声和随机外力。样本量小，保持改善不等同于反弹被消除。",
        "",
        "文献与课程对应：接触前预测/速度目标 -> MPC；xddot=J qddot+Jdot qdot -> 操作空间QP；"
        "Md sddot+Dd sdot=F -> 导纳及耗散；接触支撑与执行器限制 -> 接触力学及约束控制。",
        "导纳连续模型在无饱和时具有阻尼耗散项Dd*sdot²；经过QP、饱和、采样及位置跟踪后的系统尚无完整被动性/稳定性证明。",
        "下一步：针对首20 ms优先验证接触前可行速度匹配及手指预成形；针对60-160 ms弹出验证分指速度/承载反馈，"
        "不能仅靠把全部手指闭合时间拉长。若控制带宽不足，再单独研究真实弹性/阻尼缓冲结构，标明材料参数。",
        "代码：controllers/contact_admittance.py；experiments/buffer_ablation.py；experiments/buffer_report.py。",
        "完整复现：python -m experiments.buffer_ablation --output-dir results/<new-folder> --seeds 803 805",
        "报告复现：python -m experiments.buffer_report results/<folder>",
        "文献详情和使用边界见literature.csv。图表均为英文，原始逐例/逐步数据保存在同目录CSV。",
    ])
    (path/"experiment_notes.txt").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines[14:]),flush=True)


if __name__=="__main__":main()
