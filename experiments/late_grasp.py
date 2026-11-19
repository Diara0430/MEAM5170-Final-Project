"""复现早期角冲量较小但最终失败的试验，比较闭合后、倾腕和保持阶段。"""
import argparse
from itertools import groupby
from pathlib import Path
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial
from experiments.link_angular_impulse import read_csv
from experiments.static_grasp import contacts_with


def replay(prior):
    seed, mm = int(prior["seed"]), int(prior["palm_x_mm"])
    first_ms = round(float(prior["first_contact_s"]) * 1000)
    timing = TimingTrial([float(prior[k]) for k in ("gust_onset_s", "gust_duration_s", "gust_az_m_s2")], False)
    samples = []

    def observe(model, data, elapsed, closure, velocity):
        ms = round(1000 * elapsed)
        palm = data.body(closure.palm)
        obj = model.geom_bodyid[closure.projectile]
        pos = 1000 * (palm.xmat.reshape(3, 3).T @ (data.xipos[obj] - palm.xpos))
        # 在当前世界接触点求相对速度，并投影到切平面；自由飞行时留空。
        loads = np.zeros(5)  # 顺序为掌面、指1、指2、指3、拇指。
        slips = []
        for i, c in enumerate(data.contact):
            if closure.projectile not in (c.geom1, c.geom2) or c.efc_address < 0:
                continue
            other = c.geom1 if c.geom2 == closure.projectile else c.geom2
            group = closure.group.get(other)
            if group is None:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            if wrench[0] <= 0:
                continue
            loads[group+1] += wrench[0]
            if group >= 0 and wrench[0] >= .02:
                jo, jf = np.empty((3, model.nv)), np.empty((3, model.nv))
                mujoco.mj_jac(model, data, jo, None, c.pos, obj)
                mujoco.mj_jac(model, data, jf, None, c.pos, model.geom_bodyid[other])
                slips.append(np.linalg.norm(c.frame.reshape(3, 3)[1:] @ ((jo-jf) @ velocity)))
        touching = contacts_with(data, closure.projectile)
        samples.append(dict(seed=seed, palm_mm=mm, elapsed_ms=ms, release_ms=first_ms+ms,
            dt_ms=round(1000*closure.dt, 6),
            **{part+"_normal_N":load for part,load in zip(("palm","finger1","finger2","finger3","thumb"),loads)},
            finger3_loaded=int(2 in closure.loaded), opposed=int(closure.opposed_time > 0),
            confirmed_at_end=int(closure.holding),
            finger_touch=int(any(closure.group.get(g,-1) >= 0 for g in touching)),
            floor_touch=int(model.geom("floor").id in touching),
            mean_finger_slip_m_s=np.mean(slips) if slips else None,
            **{f"palm_{axis}_mm":v for axis,v in zip("xyz",pos)}))

    class Summary:
        def writerow(self, row):
            self.tilt, self.hold, self.first_excess = row[-3:]
    schedule = Summary()
    run("early", .04, .19, True, shape="box", pitch_deg=40., palm_x_mm=mm,
        arrival_speed="fast", timing_trial=timing, contact_observer=observe, summary=schedule)
    for k in ("failure_stage", "passed", "max_drift_mm", "first_contact_s", "close_s"):
        if str(timing.outcome[k]) != prior[k]:
            raise ValueError(f"Replay mismatch for seed {seed}, {mm} mm, {k}")
    tilt, hold = round(1000*schedule.tilt), round(1000*schedule.hold)
    # 50 ms 基准用来观察接取后的继续移动；倾腕开始时重新置零才对应原 25 mm 判据。
    pos = np.array([[r[f"palm_{axis}_mm"] for axis in "xyz"] for r in samples])
    ref50 = pos[next(i for i,r in enumerate(samples) if r["elapsed_ms"] == 50)]
    ref_tilt = pos[next(i for i,r in enumerate(samples) if r["release_ms"] == tilt)]
    for p,r in zip(pos,samples):
        t = r["release_ms"]
        r["phase"] = "pretilt" if t < tilt else "tilt" if t < hold else "hold"
        r["motion_from_50ms_mm"] = np.linalg.norm(p-ref50) if r["elapsed_ms"] >= 50 else None
        r["drift_from_tilt_mm"] = np.linalg.norm(p-ref_tilt) if t >= tilt else None
    late = [r for r in samples if r["elapsed_ms"] >= 50]
    peak = max(r["drift_from_tilt_mm"] for r in late if r["phase"] != "pretilt")
    if abs(peak-float(prior["max_drift_mm"])) > .00051:
        raise ValueError("Palm-frame drift does not reproduce original evaluation")
    events = []
    # 输出完整区间，不把开始观察时已处于无载的样本误当作新发生的卸载。
    checks = {"finger3_unloaded":lambda r:not r["finger3_loaded"],
              "opposing_support_absent":lambda r:not r["opposed"],
              "finger_slip_over_20mm_s":lambda r:r["mean_finger_slip_m_s"] is not None and r["mean_finger_slip_m_s"] > .020,
              "no_finger_touch":lambda r:not r["finger_touch"]}
    for name, check in checks.items():
        for active, rows in groupby(samples, check):
            run_rows = list(rows)
            start, end = run_rows[0]["elapsed_ms"], run_rows[-1]["elapsed_ms"] + run_rows[-1]["dt_ms"]
            if active and end > 50:
                events.append(dict(seed=seed,palm_mm=mm,event=name,start_after_touch_ms=start,
                    end_after_touch_ms=end,duration_ms=end-start,start_release_ms=first_ms+start))
    pre = [r for r in late if tilt-100 <= r["release_ms"] < tilt]
    tilt_row = next(r for r in late if r["release_ms"] == tilt)
    def first_event(check):
        return next((r["release_ms"] for r in late if check(r)),None)
    meta = dict(seed=seed,palm_mm=mm,**timing.outcome,tilt_start_release_ms=tilt,hold_start_release_ms=hold,
        **{part+"_median_last100_before_tilt_N":np.median([r[part+"_normal_N"] for r in pre])
           for part in ("palm","finger1","finger2","finger3","thumb")},
        opposed_pct_last100_before_tilt=100*np.mean([r["opposed"] for r in pre]),
        motion_50ms_to_tilt_mm=tilt_row["motion_from_50ms_mm"],
        first_5mm_from_50ms_release_ms=first_event(lambda r:r["motion_from_50ms_mm"]>=5),
        first_25mm_drift_release_ms=first_event(lambda r:r["drift_from_tilt_mm"] is not None and r["drift_from_tilt_mm"]>25),
        first_floor_release_ms=first_event(lambda r:r["floor_touch"]),
        first_hold_finger_gap_release_ms=first_event(lambda r:r["phase"]=="hold" and not r["finger_touch"]))
    # 为避免单步抖动，提取从倾腕开始算、持续至少 10 ms 的首个区间。
    for name in ("finger3_unloaded","opposing_support_absent","finger_slip_over_20mm_s"):
        meta["first_10ms_"+name+"_since_tilt_ms"] = next(
            (max(e["start_release_ms"]-tilt, 0) for e in events if e["event"]==name
             and e["start_release_ms"]+e["duration_ms"]-max(e["start_release_ms"], tilt)>=10), None)
    expected = None if schedule.first_excess is None else round(1000*schedule.first_excess)
    if meta["first_25mm_drift_release_ms"] != expected:
        raise ValueError("Drift threshold timing mismatch")
    return meta,late,events


def figure(path, samples):
    # 同一行内两种掌位共用纵轴；色带表示该步有对侧支撑。空白滑速表示没有承载的指面接触。
    import math
    from collections import defaultdict
    series = defaultdict(dict)
    for r in samples:
        series[r["seed"]].setdefault(r["palm_mm"], []).append(r)
    width = 225
    height = 180+118*len(series)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1130" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif" fill="#203542">',
           '<text x="25" y="32" font-size="23">Grasp stability through wrist tilt</text>',
           '<text x="25" y="57" font-size="14">Five matched gusts with lower early angular impulse at 0 mm. '
           'Shading marks wrist tilt (0.90 to 1.90 s). Traces start 50 ms after impact.</text>',
           '<text x="25" y="82" fill="#ab4928">Orange: 0 mm</text><text x="155" y="82" fill="#22658b">Blue: +10 mm</text>',
           '<text x="298" y="82" font-size="13">Slip gaps indicate no loaded finger contact. Drift starts at tilt onset; display capped at 60 mm.</text>']
    columns = (("finger3_normal_N","Finger 3 normal load (N)"),("opposed","Opposing contact present"),
               ("mean_finger_slip_m_s","Mean finger slip (m/s)"),("drift_from_tilt_mm","Palm-relative drift (mm)"))
    for col,(key,title) in enumerate(columns):
        x = 74+col*265
        svg.append(f'<text x="{x}" y="116" font-size="15" font-weight="bold">{title}</text>')
        for i,(seed,pair) in enumerate(sorted(series.items())):
            y = 142+i*118
            if col==0:
                svg.append(f'<text x="10" y="{y+37}" font-size="14">Gust {seed}</text>')
            svg.append(f'<rect x="{x+width*.9/2.4:.1f}" y="{y}" width="{width/2.4:.1f}" height="68" fill="#edf1f3"/>')
            maxval = max((r[key] for g in pair.values() for r in g if r[key] is not None),default=1.)
            limit = (60 if key=="drift_from_tilt_mm" else 1 if key=="opposed" else
                     max(.1, math.ceil(10*maxval)/10))
            if key != "opposed":
                svg.append(f'<text x="{x-4}" y="{y+6}" text-anchor="end" font-size="10">{limit:g}</text>')
            for t in (0,.9,1.9,2.4):
                tx = x+width*t/2.4
                svg.append(f'<text x="{tx:.1f}" y="{y+84}" font-size="10" text-anchor="middle">{t:g}</text>')
            svg.append(f'<path d="M{x} {y+68}h{width}" stroke="#a4adb3"/>')
            if key=="drift_from_tilt_mm":
                svg.append(f'<path d="M{x} {y+68-68*25/60:.1f}h{width}" stroke="#6a757b" stroke-dasharray="3 3"/>')
            for mm,color in ((10,"#22658b"),(0,"#ab4928")):
                segments = []
                last = None
                for r in pair[mm]:
                    v = r[key]
                    if v is None or (key=="opposed" and not v):
                        last = None
                        continue
                    px = x+width*r["release_ms"]/2400
                    py = y+(44 if mm==10 else 24) if key=="opposed" else y+68*(1-min(v,limit)/limit)
                    segments.append(f'{"M" if last is None else "L"}{px:.2f} {py:.2f}')
                    if key=="opposed":
                        segments.append(f'h{width*r["dt_ms"]/2400:.2f}')
                    last = px
                sw = 5 if key=="opposed" else 1.15
                svg.append(f'<path d="{" ".join(segments)}" fill="none" stroke="{color}" stroke-width="{sw}"/>')
    svg.append(f'<text x="565" y="{height-17}" font-size="14" text-anchor="middle">Time since release (s). Dashed line: 25 mm evaluation threshold.</text>')
    path.write_text("\n".join([*svg,"</g></svg>"]),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path(__file__).resolve().parents[1]/"results")
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    pairs = read_csv(args.results/"palm_contact_sources/pairs.csv")
    seeds = {int(r["seed"]) for r in pairs if r["outcome"]=="lost" and float(r["delta_mN_m_s"])<0}
    prior = [r for r in read_csv(args.results/"palm_offset_gusts/trials.csv")
             if r["split"]=="validation" and int(r["seed"]) in seeds]
    if not seeds or sorted((int(r["seed"]),int(r["palm_x_mm"])) for r in prior) != sorted((s,p) for s in seeds for p in (0,10)):
        parser.error("Expected complete matched pairs for the selected failures")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    outcomes, samples, events = [], [], []
    for row in prior:
        outcome,trace,intervals = replay(row)
        outcomes.append(outcome)
        samples.extend(trace)
        events.extend(intervals)
    write_rows(args.output_dir/"summary.csv",outcomes)
    write_rows(args.output_dir/"trace.csv",samples)
    write_rows(args.output_dir/"events.csv",events)
    figure(args.output_dir/"late_grasp_en.svg",samples)
