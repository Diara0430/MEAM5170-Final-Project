"""五个历史反例与独立阵风下，比较固定闭指、单步 IK 和分段闭环 IK。"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.thumb_face import ThumbFaceStep
from controllers.thumb_face_feedback import ThumbFaceFeedback
from experiments.contact_impulse import write_rows
from experiments.link_angular_impulse import read_csv
from experiments.thumb_face_reach import trial


MODES = ("fixed", "ik", "feedback")


def replay(job, headless=True):
    seed, mode, split = job
    step = ThumbFaceFeedback() if mode == "feedback" else ThumbFaceStep(mode)
    samples = []

    def measure(model,data,t,control,gap,total,pad_load):
        if split != "five_cases":
            return
        # 距离和接触力对应步初构型；实际关节角记录积分后的步末值。
        cmd = [data.actuator(f"hand/{i}").ctrl[0] for i in range(12,16)]
        actual = data.qpos[control.adr]
        torques = data.qfrc_actuator[control.dof]
        limits = model.jnt_actfrcrange[model.dof_jntid[control.dof]]
        limited = np.any(np.isclose(torques[:,None],limits,rtol=0,atol=1e-8),axis=1)
        # 单独检查拇指碰其他手部部位，避免把自碰撞误认成支撑盒子。
        self_load = 0.
        pairs = defaultdict(float)
        for i,c in enumerate(data.contact):
            a,b = c.geom1,c.geom2
            if c.efc_address < 0 or ((a in control.thumb_geoms) == (b in control.thumb_geoms)):
                continue
            other = b if a in control.thumb_geoms else a
            if (model.body(model.geom_bodyid[other]).name or "").startswith("hand/"):
                wrench = np.empty(6)
                mujoco.mj_contactForce(model,data,i,wrench)
                self_load += max(0.,wrench[0])
                if wrench[0] > 0:
                    thumb = a if a in control.thumb_geoms else b
                    names = tuple(model.body(model.geom_bodyid[g]).name for g in (thumb,other))
                    pairs[" / ".join(names)] += wrench[0]
        samples.append(dict(seed=seed,mode=mode,t_s=round(t,6),segment=getattr(control,"segment",1),
            gap_mm=gap,thumb_box_N=total,pad_face_N=pad_load,thumb_hand_N=self_load,
            largest_thumb_hand_pair=max(pairs,key=pairs.get) if pairs else None,
            largest_pair_N=max(pairs.values(),default=0.),saturated_joints=int(sum(limited)),
            saturated_ids="|".join(str(j+12) for j in np.flatnonzero(limited)),
            **{f"q{j}_end_rad":v for j,v in enumerate(actual,12)},
            **{f"q{j}_command_rad":v for j,v in enumerate(cmd,12)}))

    result = trial(job,headless=headless,controller=step,observe=measure)
    if result is None:
        return None,[]
    result["segments"] = getattr(step,"segment",int(step.face is not None and mode=="ik"))
    result["load_triggered_hold_s"] = getattr(step,"hold_time",None)
    return result,samples


def plot(path, samples, stats):
    seeds = sorted({r["seed"] for r in samples})
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="555">',
           '<rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif" fill="#203542">',
           '<text x="28" y="36" font-size="22">Closed-loop thumb approach before wrist tilt</text>',
           '<text x="28" y="63" font-size="14">Distance from the same pad point to the moving box face (mm). Each panel shows one diagnostic gust.</text>']
    colors = dict(fixed="#566675",ik="#bb6429",feedback="#087565")
    for j,(name,title) in enumerate(zip(MODES,("Fixed closure","Single step","Closed loop"))):
        x=35+175*j
        svg.append(f'<path d="M{x} 88h26" stroke="{colors[name]}" stroke-width="3"/><text x="{x+32}" y="93">{title}</text>')
    maxgap = max(20,10*np.ceil(max(r["gap_mm"] for r in samples)/10))
    for k,seed in enumerate(seeds):
        x,y=45+295*(k%3),142+195*(k//3)
        svg.append(f'<text x="{x}" y="{y-13}" font-size="15">Gust {seed}</text>')
        for g in (0,maxgap/2,maxgap):
            yy=y+115*(1-g/maxgap)
            svg.append(f'<path d="M{x} {yy}h220" stroke="#d4dcde"/><text x="{x-5}" y="{yy+4}" font-size="11" text-anchor="end">{g:g}</text>')
        for t in (.35,.60,.90):
            svg.append(f'<text x="{x+220*(t-.35)/.55:.1f}" y="{y+132}" font-size="11" text-anchor="middle">{t:.2f}</text>')
        for mode in MODES:
            coords = " ".join(f'{x+220*(r["t_s"]-.35)/.55:.2f},{y+115*(1-r["gap_mm"]/maxgap):.2f}'
                for r in samples if r["seed"]==seed and r["mode"]==mode)
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[mode]}" stroke-width="1.7"/>')
    svg.append('<text x="635" y="329" font-weight="bold">Fresh gusts caught</text>')
    for i,row in enumerate(r for r in stats if r["split"]=="fresh_gusts"):
        y=361+i*29
        label={"ik":"Single step","fixed":"Fixed closure","feedback":"Closed loop"}[row["mode"]]
        svg.append(f'<text x="635" y="{y}" fill="{colors[row["mode"]]}">{label}</text><text x="775" y="{y}">{row["caught"]}/{row["n"]}</text>')
    svg.append('<text x="28" y="530" font-size="14">Horizontal axis: time since release (s). Wrist tilt starts at 0.90 s.</text>')
    path.write_text("\n".join([*svg,"</g></svg>"]),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--view-seed",type=int)
    parser.add_argument("--mode",choices=MODES,default="feedback")
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    if args.view_seed is not None:
        replay((args.view_seed,args.mode,"viewer"),headless=False)
        sys.exit()
    if args.output_dir is None or args.workers < 1:
        parser.error("Supply --output-dir and a positive worker count")
    prior = read_csv(Path(__file__).resolve().parents[1]/"results/thumb_face_reach/trials.csv")
    old = {(int(r["seed"]),r["mode"]):r for r in prior if r["split"]=="five_cases"}
    cases = sorted({s for s,_ in old})
    jobs = [(s,m,split) for split,seeds in (("five_cases",cases),("fresh_gusts",range(264,312)))
            for s in seeds for m in MODES]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(replay,jobs))
    rows = [r for r,_ in results]
    samples = [x for _,trace in results for x in trace]
    groups = defaultdict(list)
    for r in rows:
        groups[r["split"],r["mode"]].append(r)
        ref = old.get((r["seed"],r["mode"]))
        if ref:
            for key in ("passed","failure_stage","max_drift_mm"):
                if str(r[key]) != ref[key]:
                    raise ValueError(f"Baseline mismatch at {key}")
            if not np.isclose(r["median_pretilt_mm"],float(ref["median_pretilt_mm"]),atol=1e-8,rtol=0):
                raise ValueError("Baseline pre-tilt gap differs")
    stats=[]
    for (split,mode),grp in groups.items():
        base = {r["seed"]:r for r in groups[split,"fixed"]}
        for r in grp:
            for k in ("first_contact_s","spin50_rad_s","initial_mm","target_face"):
                if r[k]!=base[r["seed"]][k]:
                    raise ValueError(f"Unmatched initial condition: {k}")
        delta=[r["passed"]-base[r["seed"]]["passed"] for r in grp]
        planned=[r for r in grp if r["target_face"]]
        stats.append(dict(split=split,mode=mode,n=len(grp),planned=len(planned),caught=sum(r["passed"] for r in grp),
            gains_vs_fixed=delta.count(1),losses_vs_fixed=delta.count(-1),within_2mm=sum(r["min_before_tilt_mm"]<=2 for r in planned),
            target_loaded=sum(r["first_pad_face_load_s"] is not None for r in planned),
            target_loaded_pretilt=sum(r["median_pad_face_N"]>=.02 for r in planned),
            contact_triggered_hold=sum(r["load_triggered_hold_s"] is not None for r in planned),
            median_pretilt_gap_mm=median(r["median_pretilt_mm"] for r in planned) if planned else None,
            median_peak_drift_mm=median(r["max_drift_mm"] for r in grp)))
    write_rows(args.output_dir/"summary.csv",stats)
    write_rows(args.output_dir/"trials.csv",rows)
    write_rows(args.output_dir/"diagnostic_steps.csv",samples)
    plot(args.output_dir/"feedback_en.svg",samples,stats)
