"""分阶段力学参数对照，不使用学习模型。

1. 五个掌面角度，按形状/速度选候选；2. 固定位姿比较七种动作；
3. 新种子验证原始方案、候选位姿、候选位姿加动作。
所有试验使用相同世界系发射条件，评价原姿态保持两秒。
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.catch_stability import ArmBuffer
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial,gust_for_seed

# 提前量相对预计 0.20 s 到达时刻，duration 为闭指曲线时长。
ACTIONS={"standard":(.04,.19,0.),"early":(.06,.19,0.),"late":(.02,.19,0.),
         "close_fast":(.04,.14,0.),"close_slow":(.04,.24,0.),
         "buffer_half":(.04,.19,.5),"buffer_full":(.04,.19,1.)}


def trial(job):
    seed,shape,speed,angle,action,split,label=job
    lead,duration,scale=ACTIONS[action]
    timing=TimingTrial(gust_for_seed(seed),False)
    buffer=ArmBuffer(scale) if scale else None
    forces=[]
    pose_error=""

    def observe(model,data,elapsed,closure,velocity):
        if elapsed >= .100-1e-10:
            return
        total=0.
        for i,c in enumerate(data.contact):
            if closure.projectile not in (c.geom1,c.geom2) or c.efc_address<0:
                continue
            other=c.geom2 if c.geom1==closure.projectile else c.geom1
            if other not in closure.group:
                continue
            wrench=np.empty(6)
            mujoco.mj_contactForce(model,data,i,wrench)
            total+=max(0.,wrench[0])
        forces.append(total)

    try:
        with redirect_stdout(StringIO()):
            run("early",lead,duration,True,shape=shape,pitch_deg=40. if shape=="box" else 0.,
                palm_x_mm=0.,arrival_speed=speed,timing_trial=timing,
                evaluation="hold",hold_duration_s=2.,pretilt_deg=angle,
                arm_buffer=buffer,contact_observer=observe)
    except ValueError as exc:
        if not str(exc).startswith(("Requested palm pose", "Pre-catch pose tracking")):
            raise
        pose_error=str(exc)
        timing.outcome=dict(passed=0,failure_stage="pose_infeasible",close_s=None,
                            first_contact_s=None,max_drift_mm=None)
    return dict(seed=seed,shape=shape,speed=speed,angle_deg=angle,action=action,split=split,label=label,
        **timing.outcome,pose_error=pose_error,lead_s=lead,closure_duration_s=duration,
        peak_normal_first100ms_N=max(forces,default=0.),
        actual_buffer_mm=buffer.actual_mm if buffer else 0.,
        buffer_infeasible=int(buffer.infeasible) if buffer else 0,
        max_buffer_command_speed_rad_s=buffer.max_command_speed_rad_s if buffer else 0.)


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["split"],r["shape"],r["speed"],r["angle_deg"],r["action"],r["label"]].append(r)
    out=[]
    for (split,shape,speed,angle,action,label),group in groups.items():
        drift=[r["max_drift_mm"] for r in group if r["max_drift_mm"] is not None]
        out.append(dict(split=split,shape=shape,speed=speed,angle_deg=angle,action=action,label=label,
            n=len(group),succeeded=sum(r["passed"] for r in group),
            contacted=sum(r["first_contact_s"] is not None for r in group),
            infeasible=sum(r["failure_stage"]=="pose_infeasible" or r["buffer_infeasible"] for r in group),
            median_drift_mm=median(drift) if drift else None,
            median_peak_normal_N=median(r["peak_normal_first100ms_N"] for r in group)))
    return out


def select(summary,shape,speed):
    options=[r for r in summary if r["shape"]==shape and r["speed"]==speed]
    # 优先成功数，平分时优先较小角度或原有动作，减少不必要改动。
    return max(options,key=lambda r:(r["succeeded"],-r["angle_deg"],r["action"]=="standard",
                                     -(r["median_drift_mm"] if r["median_drift_mm"] is not None else float("inf"))))


def plot(path,summary,choices):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620">',
         '<rect width="1000" height="620" fill="white"/>',
         '<g font-family="Arial" fill="#17354a">',
         '<text x="25" y="35" font-size="22">Capture pose and action: held-out validation</text>',
         '<text x="25" y="62" font-size="13">24 paired gust seeds per condition. Success requires a 2-second hold at the capture pose.</text>',
         '<text x="25" y="85" font-size="13">Blue: original. Gray: selected pose. Orange: selected pose and action.</text>']
    for i,c in enumerate(choices):
        y=122+73*i
        svg.append(f'<text x="25" y="{y+15}" font-size="15">{c["shape"].title()} / {c["speed"]}</text>')
        svg.append(f'<text x="25" y="{y+36}" font-size="12">{c["angle_deg"]:g} deg, {c["action"]}</text>')
        for j,(label,color) in enumerate((("original","#28648d"),("pose_only","#83979e"),("pose_action","#bd542c"))):
            r=next(r for r in summary if (r["shape"],r["speed"],r["label"])==(c["shape"],c["speed"],label))
            svg.append(f'<rect x="220" y="{y+19*j}" width="{640*r["succeeded"]/r["n"]}" height="14" fill="{color}"/>')
            svg.append(f'<text x="880" y="{y+12+19*j}" font-size="13">{r["succeeded"]}/{r["n"]}</text>')
    svg.append('<text x="25" y="590" font-size="13">Candidates selected using earlier, disjoint seeds. Fixed launch direction, size and mass.</text></g></svg>')
    path.write_text("\n".join(svg),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    cells=[(shape,speed) for shape in ("ball","box") for speed in ("slow","medium","fast")]
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[(seed,s,v,a,"standard","pose_selection","candidate") for s,v in cells
              for seed in range(640,656) for a in (0.,30.,45.,60.,90.)]
        rows=list(pool.map(trial,jobs))
        pose_summary=summarize(rows)
        write_rows(args.output_dir/"pose_trials.csv",rows)
        write_rows(args.output_dir/"pose_summary.csv",pose_summary)
        poses={(s,v):select(pose_summary,s,v)["angle_deg"] for s,v in cells}
        print("Selected angles",poses,flush=True)
        jobs=[(seed,s,v,poses[s,v],action,"action_selection","candidate") for s,v in cells
              for seed in range(656,668) for action in ACTIONS]
        rows=list(pool.map(trial,jobs))
        action_summary=summarize(rows)
        write_rows(args.output_dir/"action_trials.csv",rows)
        write_rows(args.output_dir/"action_summary.csv",action_summary)
        actions={(s,v):select(action_summary,s,v)["action"] for s,v in cells}
        print("Selected actions",actions,flush=True)
        jobs=[(seed,s,v,angle,action,"validation",label) for s,v in cells for seed in range(668,692)
              for label,angle,action in (("original",0.,"standard"),("pose_only",poses[s,v],"standard"),
                                         ("pose_action",poses[s,v],actions[s,v]))]
        rows=list(pool.map(trial,jobs))
    write_rows(args.output_dir/"validation_trials.csv",rows)
    write_rows(args.output_dir/"validation_summary.csv",summarize(rows))
    choices=[dict(shape=s,speed=v,angle_deg=poses[s,v],action=actions[s,v]) for s,v in cells]
    write_rows(args.output_dir/"selected_parameters.csv",choices)
    plot(args.output_dir/"pose_action_validation_en.svg",summarize(rows),choices)
    for r in summarize(rows):
        print(r)
