"""比较原始闭指、以实测姿态为目标、以及在该姿态上加一次四关节 IK 的接取结果。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

import numpy as np

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.thumb_face import ThumbFaceStep
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed
from experiments.link_angular_impulse import read_csv


MODES = ("fixed","anchor","ik")


def trial(job, headless=True, controller=None, observe=None):
    seed, mode, split = job
    pulse = gust_for_seed(seed)
    timing = TimingTrial(pulse,False)
    step = ThumbFaceStep(mode) if controller is None else controller
    gaps, loads, target_loads = [],[],[]
    first_loaded = spin50 = min_gap = None

    def measure(model,data,t,first,twist):
        nonlocal first_loaded, spin50, min_gap
        if abs(t+model.opt.timestep-first-.05) < 1e-10:
            spin50 = np.linalg.norm(data.joint("flight").qvel[3:])
        if step.face is None or t >= .900:
            return
        gap = 1000*step.gap(data)
        min_gap = gap if min_gap is None else min(min_gap,gap)
        total, pad_load = step.contact_loads(model,data)
        if observe is not None:
            observe(model,data,t,step,gap,total,pad_load)
        if first_loaded is None and pad_load >= .02:
            first_loaded = t
        if t >= .800-1e-10:
            gaps.append(gap)
            loads.append(total)
            target_loads.append(pad_load)

    run("early",.04,.19,headless,shape="box",pitch_deg=40.,palm_x_mm=0.,arrival_speed="fast",
        timing_trial=timing,post_close_control=step,after_contact_step=measure)
    if not timing.outcome:
        return None
    if step.face and len(gaps)!=50:
        raise ValueError("Incomplete pre-tilt window")
    return dict(split=split,seed=seed,mode=mode,gust_onset_s=pulse[0],gust_duration_s=pulse[1],
        gust_az_m_s2=pulse[2],**timing.outcome,spin50_rad_s=spin50,plan_status=step.reason,
        plan_time_s=step.start,target_face=step.face,initial_mm=1000*step.initial_gap if gaps else None,
        predicted_step_mm=1000*step.predicted_gap if gaps else None,
        best_static_mm=1000*step.best_static_gap if gaps else None,min_before_tilt_mm=min_gap,
        median_pretilt_mm=median(gaps) if gaps else None,first_pad_face_load_s=first_loaded,
        median_thumb_N=median(loads) if loads else None,median_pad_face_N=median(target_loads) if loads else None)


def summarize(rows):
    stats = []
    for split in ("five_cases","fresh_gusts"):
        baseline = {r["seed"]:r for r in rows if r["mode"]=="fixed" and r["split"]==split}
        for mode in MODES:
            group = [r for r in rows if r["split"]==split and r["mode"]==mode]
            planned = [r for r in group if r["target_face"]]
            for r in group:
                ref = baseline[r["seed"]]
                for key in ("spin50_rad_s","first_contact_s","close_s","initial_mm","target_face","plan_status"):
                    if r[key] != ref[key]:
                        raise ValueError(f"Unmatched initial states: {r['seed']}, {key}")
            changes = [r["passed"]-baseline[r["seed"]]["passed"] for r in group]
            stats.append(dict(split=split,mode=mode,n=len(group),planned=len(planned),
                caught=sum(r["passed"] for r in group),gains=changes.count(1),losses=changes.count(-1),
                static_within_2mm=sum(r["best_static_mm"]<=2 for r in planned),
                within_2mm_before_tilt=sum(r["min_before_tilt_mm"]<=2 for r in planned),
                ever_loaded_target=sum(r["first_pad_face_load_s"] is not None for r in planned),
                pretilt_loaded_target=sum(r["median_pad_face_N"]>=.02 for r in planned),
                pretilt_loaded_thumb=sum(r["median_thumb_N"]>=.02 for r in planned),
                median_pretilt_mm=median(r["median_pretilt_mm"] for r in planned) if planned else None,
                median_peak_drift_mm=median(r["max_drift_mm"] for r in group)))
    return stats


def figure(path,rows,stats):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="395">',
         '<rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif" fill="#253745">',
         '<text x="28" y="37" font-size="22">Thumb reach to the opposing box face</text>',
         '<text x="28" y="65" font-size="14">Left: distance from a tracked pad point to the finite face. Right: paired trials under new gusts.</text>',
         '<text x="28" y="110" font-weight="bold">Five diagnostic gusts (mm)</text>',
         '<text x="480" y="110" font-weight="bold">Fresh gusts caught (out of 48)</text>']
    for i,r in enumerate(x for x in rows if x["split"]=="five_cases" and x["mode"]=="ik"):
        y=142+36*i
        svg.append(f'<text x="28" y="{y+15}" font-size="14">Gust {r["seed"]}</text>')
        if r["initial_mm"] is None:
            continue
        for j,(key,color) in enumerate((("initial_mm","#617c91"),("median_pretilt_mm","#147469"))):
            v=r[key]
            svg.append(f'<rect x="111" y="{y+j*13}" width="{v*2.2:.2f}" height="9" fill="{color}"/>')
            svg.append(f'<text x="{118+v*2.2:.1f}" y="{y+j*13+8}" font-size="11">{v:.1f}</text>')
    for i,r in enumerate(x for x in stats if x["split"]=="fresh_gusts"):
        y=153+52*i
        label={"fixed":"Baseline","anchor":"Measured pose","ik":"Measured + IK"}[r["mode"]]
        svg.extend([f'<text x="480" y="{y}" font-size="14">{label}</text>',
                    f'<rect x="601" y="{y-12}" width="178" height="14" fill="#e3e9ed"/>',
                    f'<rect x="601" y="{y-12}" width="{178*r["caught"]/r["n"]:.1f}" height="14" fill="#147469"/>',
                    f'<text x="792" y="{y}" font-size="14">{r["caught"]}/{r["n"]}</text>'])
    svg.append('<text x="28" y="352" font-size="14">Gray: at closure. Green, left panel: median gap in the last 100 ms before tilt with IK.</text>')
    path.write_text("\n".join([*svg,"</g></svg>"]),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--view-seed",type=int)
    parser.add_argument("--mode",choices=MODES,default="ik")
    args=parser.parse_args()
    if args.view_seed is not None:
        trial((args.view_seed,args.mode,"viewer"),headless=False)
        sys.exit()
    if not args.output_dir or args.workers < 1:
        parser.error("Supply --output-dir and a positive worker count")
    root=Path(__file__).resolve().parents[1]/"results"
    cases=[int(r["seed"]) for r in read_csv(root/"palm_contact_sources/pairs.csv")
           if r["outcome"]=="lost" and float(r["delta_mN_m_s"])<0]
    jobs=[(s,m,split) for split,seeds in (("five_cases",cases),("fresh_gusts",range(216,264)))
          for s in seeds for m in MODES]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(trial,jobs))
    prior={int(r["seed"]):r for r in read_csv(root/"late_grasp/summary.csv") if r["palm_mm"]=="0"}
    for r in rows:
        if r["split"]=="five_cases" and r["mode"]=="fixed":
            for key in ("passed","failure_stage","max_drift_mm"):
                if str(r[key])!=prior[r["seed"]][key]:
                    raise ValueError("Diagnostic baseline differs from prior experiment")
    stats=summarize(rows)
    write_rows(args.output_dir/"trials.csv",rows)
    write_rows(args.output_dir/"summary.csv",stats)
    figure(args.output_dir/"thumb_face_en.svg",rows,stats)
