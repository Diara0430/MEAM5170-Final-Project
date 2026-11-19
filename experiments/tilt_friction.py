"""复现第 1 指让位改变成败的配对试验，按手-物接触点测量倾腕和保持时的摩擦与滑动。"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import mujoco
import numpy as np

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.thumb_face_feedback import ThumbFaceFeedback
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial
from experiments.index_clearance import LiveRelease
from experiments.link_angular_impulse import read_csv


PARTS = ("palm","finger1","finger2","finger3","thumb")
TILT_MS, HOLD_MS = 900, 1900  # 此实验与既有运行配置一致；在回放中核对阈值时间和峰值。


def sustained(rows, predicate, dt_ms, min_ms=10):
    """连续满足条件至少 min_ms；报告区间开头，相对倾腕起点计时。"""
    duration = 0.
    for r in rows:
        duration = duration+dt_ms if predicate(r) else 0.
        if duration >= min_ms-1e-9:
            return round(r["tilt_ms"]+dt_ms-duration,6)
    return None


def replay(prior, direction):
    seed,mode=int(prior["seed"]),prior["mode"]
    tags=dict(seed=seed,mode=mode,change=direction)
    first_ms=round(float(prior["first_contact_s"])*1000)
    timing=TimingTrial(tuple(float(prior[k]) for k in ("gust_onset_s","gust_duration_s","gust_az_m_s2")),False)
    ctrl=LiveRelease() if mode=="live30" else ThumbFaceFeedback()
    steps,points=[],[]
    reference=None

    def observe(model,data,elapsed,closure,velocity):
        nonlocal reference
        t=first_ms+round(1000*elapsed)-TILT_MS
        if t < 0:
            return
        palm=data.body(closure.palm)
        box=model.geom_bodyid[closure.projectile]
        pos=palm.xmat.reshape(3,3).T@(data.xipos[box]-palm.xpos)
        if reference is None:
            if t != 0:
                raise ValueError("Missing tilt onset")
            reference=pos.copy()
        loads=np.zeros(5)
        saturated_sliding=np.zeros(5)
        weighted_slip=0.
        current=[]
        for i,c in enumerate(data.contact):
            if c.efc_address < 0 or closure.projectile not in (c.geom1,c.geom2):
                continue
            other=c.geom2 if c.geom1==closure.projectile else c.geom1
            group=closure.group.get(other)
            if group is None:
                continue
            wrench=np.empty(6)
            mujoco.mj_contactForce(model,data,i,wrench)
            fn=wrench[0]
            if fn < .02:
                continue
            # 使用求解器为该接触实际配置的两个滑动摩擦系数。
            # 这里只测量切平面上的利用率；当前场景的承载点均为三维椭圆摩擦锥。
            if c.dim != 3 or model.opt.cone!=mujoco.mjtCone.mjCONE_ELLIPTIC or min(c.friction[:2])<=0:
                raise ValueError("Expected an elliptic sliding-friction contact")
            ft=wrench[1:3]
            mu=c.friction[:2]
            eta=np.linalg.norm(ft/mu)/fn
            jb,jh=np.empty((3,model.nv)),np.empty((3,model.nv))
            mujoco.mj_jac(model,data,jb,None,c.pos,box)
            mujoco.mj_jac(model,data,jh,None,c.pos,model.geom_bodyid[other])
            slip=np.linalg.norm(c.frame.reshape(3,3)[1:]@((jb-jh)@velocity))
            j=group+1
            loads[j]+=fn
            weighted_slip+=fn*slip
            if eta>=.90 and slip>=.020:
                saturated_sliding[j]+=fn
            local=data.xmat[box].reshape(3,3).T@(c.pos-data.xpos[box])
            current.append(dict(**tags,tilt_ms=t,contact_index_this_step=i,part=PARTS[j],
                body=model.body(model.geom_bodyid[other]).name,normal_N=fn,t1_N=ft[0],t2_N=ft[1],
                tangent_N=np.linalg.norm(ft),mu1=mu[0],mu2=mu[1],utilization=eta,slip_m_s=slip,
                box_x_mm=1000*local[0],box_y_mm=1000*local[1],box_z_mm=1000*local[2]))
        points.extend(current)
        weight=sum(loads)
        steps.append(dict(**tags,tilt_ms=t,dt_ms=1000*model.opt.timestep,phase="tilt" if t<HOLD_MS-TILT_MS else "hold",
            drift_mm=1000*np.linalg.norm(pos-reference),opposed=int(closure.opposed_time>0),
            normal_load_N=weight,max_utilization=max((p["utilization"] for p in current),default=None),
            weighted_slip_m_s=weighted_slip/weight if weight else None,
            near_sliding_load_pct=100*sum(saturated_sliding)/weight if weight else None,
            **{p+"_load_N":loads[j] for j,p in enumerate(PARTS)},
            **{p+"_near_sliding_N":saturated_sliding[j] for j,p in enumerate(PARTS)}))

    run("early",.04,.19,True,shape="box",pitch_deg=40.,palm_x_mm=0.,arrival_speed="fast",
        timing_trial=timing,post_close_control=ctrl,contact_observer=observe)
    for key in ("passed","failure_stage","first_contact_s","max_drift_mm","close_s"):
        if str(timing.outcome[key])!=prior[key]:
            raise ValueError(f"Replay mismatch for {seed}, {mode}, {key}")
    peak=max(r["drift_mm"] for r in steps)
    if abs(peak-float(prior["max_drift_mm"])) > .00051:
        raise ValueError("Palm-relative drift mismatch")
    dt=steps[0]["dt_ms"]
    if len(steps)!=round(1500/dt):
        raise ValueError("Incomplete tilt and hold trace")
    # 记录每个部位首次出现同一个点同时接近极限且发生滑动的持续区间。
    # 由于求解器每步重新生成接触点，跨步只按部位汇总。
    onsets={p:sustained(steps,lambda r,p=p:r[p+"_near_sliding_N"]>0,dt) for p in PARTS}
    earliest=min((v for v in onsets.values() if v is not None),default=None)
    def first_drift(mm):
        return next((r["tilt_ms"] for r in steps if r["drift_mm"]>mm),None)
    summary=dict(**tags,passed=int(prior["passed"]),failure_stage=prior["failure_stage"],max_drift_mm=peak,
        first_near_limit_ms=next((r["tilt_ms"] for r in steps if (r["max_utilization"] or 0)>=.90),None),
        first_near_sliding_ms=earliest,first_parts="|".join(p for p in PARTS if earliest is not None and onsets[p]==earliest),
        first_drift5_ms=first_drift(5),first_drift25_ms=first_drift(25),
        first_no_load_10ms=sustained(steps,lambda r:r["normal_load_N"]==0,dt),
        first_no_opposition_10ms=sustained(steps,lambda r:not r["opposed"],dt),
        **{p+"_onset_ms":onsets[p] for p in PARTS})
    return summary,steps,points


def plot(path,steps):
    series=defaultdict(lambda:defaultdict(list))
    for r in steps:
        series[r["seed"]][r["mode"]].append(r)
    height=164+119*len(series)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}"><rect width="100%" height="100%" fill="white"/>',
         '<g font-family="Arial, sans-serif" fill="#223745"><text x="24" y="35" font-size="22">Friction and drift during wrist tilt</text>',
         '<text x="24" y="61" font-size="14">Blue: no release. Orange: 30% release with replanning. Shading: tilt. White: final hold.</text>',
         '<text x="24" y="82" font-size="13">Friction and speed gaps indicate no loaded hand contact. Drift dashed line: 25 mm. Drift and speed are clipped for display.</text>']
    for j,(field,title,limit,factor) in enumerate((("near_sliding_load_pct","Normal load on near-limit sliding points (%)",100,1),
                                                  ("weighted_slip_m_s","Load-weighted slip (mm/s)",150,1000),
                                                  ("drift_mm","Palm-relative drift (mm)",50,1))):
        x=91+302*j
        svg.append(f'<text x="{x}" y="112" font-size="13" font-weight="bold">{title}</text>')
        for i,(seed,modes) in enumerate(sorted(series.items())):
            y=129+119*i; w,h=252,69
            if not j:
                status=modes["live30"][0]["change"]
                svg.append(f'<text x="9" y="{y+25}" font-size="13">Gust {seed}</text><text x="9" y="{y+43}" font-size="12">{status}</text>')
            svg.append(f'<rect x="{x}" y="{y}" width="{w*2/3}" height="{h}" fill="#edf2f4"/>')
            for v in (0,limit):
                yy=y+h*(1-v/limit)
                svg.append(f'<path d="M{x} {yy}h{w}" stroke="#b3c1c8"/><text x="{x-4}" y="{yy+3}" font-size="10" text-anchor="end">{v}</text>')
            if field=="drift_mm":
                svg.append(f'<path d="M{x} {y+h/2}h{w}" stroke="#6a7478" stroke-dasharray="3 3"/>')
            for t in (0,500,1000,1500):
                svg.append(f'<text x="{x+w*t/1500}" y="{y+h+14}" font-size="10" text-anchor="middle">{t}</text>')
            for mode,color in (("no_release","#236497"),("live30","#ae4925")):
                commands=[]; continuous=False
                for r in modes[mode]:
                    value=r[field]
                    if value is None:
                        continuous=False
                        continue
                    commands.append(f'{"L" if continuous else "M"}{x+w*r["tilt_ms"]/1500:.2f} {y+h*(1-min(limit,value*factor)/limit):.2f}')
                    continuous=True
                svg.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="1"/>')
    svg.append(f'<text x="500" y="{height-20}" text-anchor="middle" font-size="14">Time since tilt onset (ms)</text></g></svg>')
    path.write_text("\n".join(svg),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    prior=read_csv(Path(__file__).resolve().parents[1]/"results/index_clearance/trials.csv")
    pairs=defaultdict(dict)
    for r in prior:
        if r["split"]=="live_validation" and r["mode"] in ("no_release","live30"):
            pairs[int(r["seed"])][r["mode"]]=r
    args.output_dir.mkdir(parents=True,exist_ok=False)
    summaries,steps,points=[],[],[]
    for seed,pair in sorted(pairs.items()):
        a,b=pair["no_release"],pair["live30"]
        if a["passed"]==b["passed"]:
            continue
        direction="recovered" if b["passed"]=="1" else "lost"
        for prior in (a,b):
            summary,trace,contacts=replay(prior,direction)
            summaries.append(summary)
            steps.extend(trace)
            points.extend(contacts)
    write_rows(args.output_dir/"summary.csv",summaries)
    write_rows(args.output_dir/"time_series.csv",steps)
    write_rows(args.output_dir/"contacts.csv",points)
    plot(args.output_dir/"tilt_friction_en.svg",steps)
