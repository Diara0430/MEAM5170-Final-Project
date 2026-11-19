"""配对实验：重放相同的拇指命令，单独改变第 1 指在接近阶段的屈曲目标。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.thumb_face import ThumbFaceStep
from controllers.thumb_face_feedback import ThumbFaceFeedback
from experiments.contact_impulse import write_rows
from experiments.link_angular_impulse import read_csv
from experiments.thumb_face_reach import trial


class RecordThumb(ThumbFaceFeedback):
    def __init__(self):
        super().__init__()
        self.tape = {}

    def __call__(self, model, data, hand, t, tilt):
        super().__call__(model, data, hand, t, tilt)
        self.tape[round(t/model.opt.timestep)] = data.ctrl[hand.ids].copy()


def release_index(data, hand, t, tilt, control, release):
    if control.face is not None and control.start <= t < tilt:
        # 100 ms 内张开，最后 100 ms 平滑回到原闭指目标。侧摆关节 0 保持原命令。
        u = min((t-control.start)/.100, (tilt-t)/.100, 1.)
        weight = u*u*(3-2*u)
        data.ctrl[hand.ids[1:4]] *= 1-release*weight


class LiveRelease(ThumbFaceFeedback):
    def __call__(self,model,data,hand,t,tilt):
        super().__call__(model,data,hand,t,tilt)
        release_index(data,hand,t,tilt,self,.30)


class ReplayThumb(ThumbFaceStep):
    def __init__(self, tape, release):
        super().__init__("fixed")
        self.tape = tape
        self.release = release
        self.n_commands = 0

    def __call__(self, model, data, hand, t, tilt):
        # 只在起点选一次同一个测量面。拇指命令来自同一种子的对照，与让位后的接触无关。
        super().__call__(model, data, hand, t, tilt)
        recorded = self.tape[round(t/model.opt.timestep)]
        data.ctrl[hand.ids[12:]] = recorded[12:]
        self.n_commands += 1
        release_index(data,hand,t,tilt,self,self.release)


def run_condition(seed, split, name, control, headless=True):
    plateau, trace = [], []
    ids = None

    def measure(model, data, t, step, gap, thumb_load, target_load):
        nonlocal ids
        if ids is None:
            tip = model.body("hand/fingertip").id
            index_geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g]==tip and model.geom_contype[g])
            ids = (index_geom, [model.joint(f"hand/{j}").qposadr[0] for j in (1,2,3)])
        force = 0.
        tip_geom, joints = ids
        for i,c in enumerate(data.contact):
            if c.efc_address >= 0 and {c.geom1,c.geom2} == {tip_geom,step.pad_geom}:
                wrench = np.empty(6)
                mujoco.mj_contactForce(model,data,i,wrench)
                force += max(0.,wrench[0])
        # 只在完全让位时比较机制，避开倾腕前回到原目标的过渡期。
        active = t >= step.start+.100-1e-10 and t < .800-1e-10
        sample = dict(seed=seed,mode=name,t_s=round(t,6),plateau=int(active),tip_tip_N=force,
                      tip_clearance_mm=1000*mujoco.mj_geomDistance(model,data,tip_geom,step.pad_geom,.5,None),
                      pad_face_gap_mm=gap,pad_face_N=target_load,thumb_box_N=thumb_load,
                      index_flexion_sum_rad=sum(data.qpos[joints]))
        if active:
            plateau.append(sample)
        if split=="five_cases":
            trace.append(sample)

    row = trial((seed,name,split),headless=headless,controller=control,observe=measure)
    if row is None:
        return None,[]
    if row["target_face"] and not plateau:
        raise ValueError("Empty clearance plateau")
    for key in ("tip_tip_N","tip_clearance_mm","pad_face_gap_mm","pad_face_N","index_flexion_sum_rad"):
        row["plateau_"+key] = median(s[key] for s in plateau) if plateau else None
    return row,trace


def matched(job, view=None):
    seed, split = job
    record = RecordThumb()
    reference, trace = run_condition(seed,split,"no_release",record,headless=view!="no_release")
    if reference is None:
        return [],[]
    rows = [reference]
    conditions = (("fixed",None),("release15",.15),("release30",.30))
    if split=="live_validation":
        conditions = (("fixed",None),)
    for name, release in conditions:
        ctrl = ThumbFaceStep("fixed") if release is None else ReplayThumb(record.tape,release)
        row, samples = run_condition(seed,split,name,ctrl,headless=view!=name)
        if row is None:
            return [],[]
        if release is not None and ctrl.n_commands != len(record.tape):
            raise ValueError("Incomplete command replay")
        for key in ("initial_mm","first_contact_s","spin50_rad_s","target_face","close_s"):
            if row[key]!=reference[key]:
                raise ValueError(f"Mismatch before intervention at {key}")
        rows.append(row)
        trace.extend(samples)
    if split in ("five_cases","live_validation") or view=="live30":
        # 机制复核：用同样的第 1 指让位曲线，允许拇指对改变后的接触重新规划。
        row,samples=run_condition(seed,split,"live30",LiveRelease(),headless=view!="live30")
        if row is not None:
            rows.append(row)
            trace.extend(samples)
    return rows,trace


def summarize(rows):
    stats=[]
    for split in ("five_cases","fresh_gusts","live_validation"):
        ref={r["seed"]:r for r in rows if r["split"]==split and r["mode"]=="no_release"}
        modes=("fixed","no_release","release15","release30")
        if split=="five_cases":
            modes=(*modes,"live30")
        elif split=="live_validation":
            modes=("fixed","no_release","live30")
        for mode in modes:
            group=[r for r in rows if r["split"]==split and r["mode"]==mode]
            planned=[r for r in group if r["target_face"]]
            delta=[r["passed"]-ref[r["seed"]]["passed"] for r in group]
            out=dict(split=split,mode=mode,n=len(group),planned=len(planned),
                     caught=sum(r["passed"] for r in group),gains_vs_no_release=delta.count(1),
                     losses_vs_no_release=delta.count(-1),
                     target_face_ever_loaded=sum(r["first_pad_face_load_s"] is not None for r in group),
                     within_2mm=sum(r["min_before_tilt_mm"]<=2 for r in planned))
            for key in ("plateau_tip_tip_N","plateau_tip_clearance_mm","plateau_pad_face_gap_mm",
                        "plateau_index_flexion_sum_rad","median_pretilt_mm"):
                out[key]=median(r[key] for r in planned) if planned else None
            stats.append(out)
    return stats


def plot(path,stats):
    panels=(("five_cases","plateau_tip_tip_N","Fingertip contact load (N)"),
            ("five_cases","plateau_pad_face_gap_mm","Thumb-to-face gap (mm)"),
            ("fresh_gusts","caught","Catches in 48 fresh gusts"))
    labels=("Fixed closure","No release","15% release","30% release","30% + replan")
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="455"><rect width="100%" height="100%" fill="white"/>',
         '<g font-family="Arial, sans-serif" fill="#233746"><text x="26" y="37" font-size="22">Does opening finger 1 clear the thumb?</text>',
         '<text x="26" y="64" font-size="14">No release, 15% release and 30% release share the same thumb commands within each gust.</text>']
    for col,(split,key,title) in enumerate(panels):
        x=26+col*340
        grp=[s for s in stats if s["split"]==split]
        limit=48 if key=="caught" else max(1,max(s[key] for s in grp))
        svg.append(f'<text x="{x}" y="115" font-size="16" font-weight="bold">{title}</text>')
        for i,(s,label) in enumerate(zip(grp,labels)):
            y=145+42*i
            value=s[key]
            v=f"{value}/48" if key=="caught" else f"{value:.1f}"
            svg.extend([f'<text x="{x}" y="{y+12}" font-size="14">{label}</text>',
                        f'<rect x="{x+103}" y="{y}" width="135" height="14" fill="#e6edef"/>',
                        f'<rect x="{x+103}" y="{y}" width="{135*value/limit:.2f}" height="14" fill="#24766e"/>',
                        f'<text x="{x+247}" y="{y+12}" font-size="14">{v}</text>'])
    validation={s["mode"]:s for s in stats if s["split"]=="live_validation"}
    svg.append('<text x="26" y="383" font-size="13">Left and center: five diagnostic gusts, median during the full release setting within each trial, then across trials.</text>')
    svg.append(f'<text x="26" y="422" font-size="15">Separate 48-gust validation: fixed closure {validation["fixed"]["caught"]}/48; '
               f'no release {validation["no_release"]["caught"]}/48; 30% release with replanning {validation["live30"]["caught"]}/48.</text></g></svg>')
    path.write_text("\n".join(svg),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--view-seed",type=int)
    parser.add_argument("--mode",choices=("fixed","no_release","release15","release30","live30"),default="release30")
    args=parser.parse_args()
    if args.view_seed is not None:
        matched((args.view_seed,"viewer"),args.mode)
        sys.exit()
    if args.output_dir is None or args.workers<1:
        parser.error("Supply --output-dir and a positive worker count")
    prior=read_csv(Path(__file__).resolve().parents[1]/"results/thumb_face_feedback/trials.csv")
    old={int(r["seed"]):r for r in prior if r["split"]=="five_cases" and r["mode"]=="feedback"}
    jobs=[(s,split) for split,seeds in (("five_cases",sorted(old)),("fresh_gusts",range(312,360)),
                                        ("live_validation",range(360,408))) for s in seeds]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(matched,jobs))
    rows=[r for group,_ in results for r in group]
    traces=[r for _,group in results for r in group]
    for r in rows:
        if r["split"]=="five_cases" and r["mode"]=="no_release":
            for key in ("passed","failure_stage","max_drift_mm","median_pretilt_mm"):
                if str(r[key])!=old[r["seed"]][key]:
                    raise ValueError(f"Historical replay mismatch at {key}")
    stats=summarize(rows)
    write_rows(args.output_dir/"trials.csv",rows)
    write_rows(args.output_dir/"summary.csv",stats)
    write_rows(args.output_dir/"diagnostic_trace.csv",traces)
    plot(args.output_dir/"index_clearance_en.svg",stats)
