"""0 mm 掌位下，隔离拇指根部目标的影响。先选一个角度，再对新阵风配对验证。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed
from experiments.link_angular_impulse import read_csv


SWEEPS = {"initial": ((-30, -15, 0, 15, 30), range(120,168)),
          "extended": ((-75, -60, -45, 0), range(168,216))}


def trial(job, headless=True):
    split, seed, angle = job
    pulse = gust_for_seed(seed)
    timing = TimingTrial(pulse, online=False)
    loads, targets, actual, at_limit = [], [], [], []
    spin50 = None
    thumb_geoms = None
    # 本组提前 40 ms 闭指、190 ms 完成，场景从释放后 900 ms 开始倾腕。
    def measure(model, data, t, first, twist):
        nonlocal spin50, thumb_geoms
        if abs(t + model.opt.timestep - first - .050) < 1e-10:
            spin50 = float(np.linalg.norm(data.joint("flight").qvel[3:]))
        if not .800 <= round(t, 9) < .900:
            return
        if thumb_geoms is None:
            groups = ContactClosure(model, LeapHand(model)).group
            thumb_geoms = {g for g, finger in groups.items() if finger == 3}
        # 汇总拇指各接触点对物体的法向载荷。拇指碰掌或碰其他指不计入。
        force = 0.
        obj = model.geom("projectile").id
        for i,c in enumerate(data.contact):
            if obj not in (c.geom1, c.geom2) or c.efc_address < 0:
                continue
            other = c.geom1 if c.geom2 == obj else c.geom2
            if other in thumb_geoms:
                wrench = np.empty(6)
                mujoco.mj_contactForce(model, data, i, wrench)
                force += max(wrench[0], 0.)
        loads.append(force)
        targets.append(float(data.actuator("hand/12").ctrl[0]))
        joint = model.joint("hand/12")
        actual.append(float(data.joint("hand/12").qpos[0]))
        # 使用关节限幅后的实际驱动力矩，而非执行器限幅前的力。
        torque = data.qfrc_actuator[joint.dofadr[0]]
        lo, hi = model.jnt_actfrcrange[joint.id]
        at_limit.append(min(abs(torque-lo),abs(torque-hi)) < 1e-8)

    run("early", .04, .19, headless, shape="box", pitch_deg=40., palm_x_mm=0., arrival_speed="fast",
        timing_trial=timing, after_contact_step=measure, thumb_base_offset_deg=angle)
    if not timing.outcome:
        return None  # 用户提前关闭可视化窗口。
    if len(loads) != 50 or spin50 is None:
        raise ValueError(f"Missing measurement window for seed {seed}")
    return dict(split=split,seed=seed,thumb_deg=angle,gust_onset_s=pulse[0],gust_duration_s=pulse[1],
                gust_az_m_s2=pulse[2],**timing.outcome,spin_50ms_rad_s=spin50,
                thumb_target_pretilt_rad=median(targets),thumb_pretilt_median_N=median(loads),
                thumb_actual_pretilt_rad=median(actual),thumb_torque_limited_pct=100*np.mean(at_limit),
                thumb_loaded_samples_pct=100*np.mean(np.array(loads) >= .02))


def summarize(rows):
    baseline = {r["seed"]:r for r in rows if r["thumb_deg"] == 0}
    stats = []
    for angle in sorted({r["thumb_deg"] for r in rows}):
        group = [r for r in rows if r["thumb_deg"] == angle]
        gains = losses = 0
        for r in group:
            ref = baseline[r["seed"]]
            for key in ("first_contact_s","close_s","spin_50ms_rad_s"):
                if not np.isclose(r[key], ref[key], atol=1e-10, rtol=0):
                    raise ValueError(f"Early catch differs for seed {r['seed']} at {key}")
            if not np.isclose(r["thumb_target_pretilt_rad"] - ref["thumb_target_pretilt_rad"],
                              np.deg2rad(angle), atol=1e-10, rtol=0):
                raise ValueError("Thumb command did not reach the specified angle")
            gains += r["passed"] > ref["passed"]
            losses += r["passed"] < ref["passed"]
        stats.append(dict(split=group[0]["split"],thumb_deg=angle,caught=sum(r["passed"] for r in group),
            total=len(group),gains_vs_zero=gains,losses_vs_zero=losses,
            thumb_bearing_trials=sum(r["thumb_pretilt_median_N"] >= .02 for r in group),
            median_thumb_N=median(r["thumb_pretilt_median_N"] for r in group),
            median_actual_angle_deg=np.rad2deg(median(r["thumb_actual_pretilt_rad"] for r in group)),
            median_torque_limited_pct=median(r["thumb_torque_limited_pct"] for r in group),
            median_peak_drift_mm=median(r["max_drift_mm"] for r in group)))
    return stats


def plot(path, stats):
    # 用同一分母对照接住次数和拇指承载次数；载荷窗口为倾腕前最后 100 ms。
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="405">',
           '<rect width="900" height="405" fill="white"/><g font-family="Arial, sans-serif" fill="#243643">',
           '<text x="28" y="36" font-size="22">Late thumb adjustment at the 0 mm palm position</text>',
           '<text x="28" y="62" font-size="14">Each count is out of 48 identical gusts within a panel. Angles are offsets from the original joint target.</text>']
    for x,split,title in ((28,"selection","Selection gusts"),(476,"validation","Fresh validation gusts")):
        svg.append(f'<text x="{x}" y="105" font-size="18">{title}</text>')
        group = [s for s in stats if s["split"] == split]
        for i,s in enumerate(group):
            y = 133+43*i
            svg.append(f'<text x="{x}" y="{y+12}" font-size="14">{s["thumb_deg"]:+d} deg</text>')
            for off,key,color in ((0,"caught","#15776b"),(15,"thumb_bearing_trials","#b6652c")):
                w = 225*s[key]/s["total"]
                svg.extend([f'<rect x="{x+69}" y="{y+off}" width="225" height="10" fill="#e9edef"/>',
                            f'<rect x="{x+69}" y="{y+off}" width="{w:.2f}" height="10" fill="{color}"/>',
                            f'<text x="{x+302}" y="{y+off+10}" font-size="12">{s[key]}/{s["total"]}</text>'])
    svg.append('<text x="28" y="377" font-size="14">Green: caught.   Orange: median thumb normal load of at least 0.02 N in the final 100 ms before tilt.</text>')
    path.write_text("\n".join([*svg,"</g></svg>"]),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--sweep",choices=SWEEPS,default="initial")
    parser.add_argument("--view-seed",type=int)
    parser.add_argument("--angle-deg",type=float,default=0.)
    args = parser.parse_args()
    if args.view_seed is not None:
        trial(("viewer",args.view_seed,args.angle_deg),headless=False)
        sys.exit()
    if args.output_dir is None or args.workers < 1:
        parser.error("Supply --output-dir and a positive worker count")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    prior_path = Path(__file__).resolve().parents[1]/"results/palm_offset_gusts/trials.csv"
    prior = {int(r["seed"]):r for r in read_csv(prior_path)
             if r["split"]=="validation" and r["palm_x_mm"]=="0"}
    angles, validation_seeds = SWEEPS[args.sweep]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(trial,[("selection",s,a) for s in range(72,120) for a in angles]))
        for r in rows:
            if r["thumb_deg"] == 0:
                for key in ("passed","failure_stage","first_contact_s","close_s","max_drift_mm"):
                    if str(r[key]) != prior[r["seed"]][key]:
                        raise ValueError(f"Baseline replay differs at {key}")
        stats = summarize(rows)
        # 验证前锁定最佳非零调整。平局按峰值漂移中位数、偏移绝对值、角度排序。
        winner = min((s for s in stats if s["thumb_deg"] != 0),
                     key=lambda s:(-s["caught"],s["median_peak_drift_mm"],abs(s["thumb_deg"]),s["thumb_deg"]))
        angle = winner["thumb_deg"]
        print(f"Locked comparison: 0 vs {angle} degrees",flush=True)
        fresh = list(pool.map(trial,[("validation",s,a) for s in validation_seeds for a in (0,angle)]))
    stats.extend(summarize(fresh))
    write_rows(args.output_dir/"trials.csv",rows+fresh)
    write_rows(args.output_dir/"summary.csv",stats)
    plot(args.output_dir/"thumb_opposition_en.svg",stats)
