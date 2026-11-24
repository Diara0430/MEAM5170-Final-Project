"""在已独立验证的长方体 45 度位姿上，检查现有防滑反馈的增益。

中速，新种子616--639；保持与倾腕独立测试，固定参数。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.catch_stability import SlipGrip
from controllers.wrist import WristProgress
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial,gust_for_seed


def trial(job):
    seed,evaluation,mode=job
    timing=TimingTrial(gust_for_seed(seed),False)
    grip=SlipGrip() if mode=="feedback" else None
    with redirect_stdout(StringIO()):
        run("early",.04,.19,True,shape="box",pitch_deg=40.,palm_x_mm=0.,arrival_speed="medium",
            evaluation=evaluation,hold_duration_s=2. if evaluation=="hold" else .5,
            wrist_control=WristProgress("slow") if evaluation=="tilt" else None,
            pretilt_deg=45.,timing_trial=timing,post_close_control=grip)
    return dict(seed=seed,evaluation=evaluation,mode=mode,**timing.outcome,
                max_extra_rad=float(max(grip.extra)) if grip else 0.)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    jobs=[(seed,evaluation,mode) for seed in range(616,640) for evaluation in ("hold","tilt") for mode in ("fixed","feedback")]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(trial,jobs))
    summary=[]
    for evaluation in ("hold","tilt"):
        base={r["seed"]:r for r in rows if r["evaluation"]==evaluation and r["mode"]=="fixed"}
        for mode in ("fixed","feedback"):
            group=[r for r in rows if r["evaluation"]==evaluation and r["mode"]==mode]
            summary.append(dict(evaluation=evaluation,mode=mode,n=len(group),succeeded=sum(r["passed"] for r in group),
                gains=sum(r["passed"] and not base[r["seed"]]["passed"] for r in group),
                losses=sum(not r["passed"] and base[r["seed"]]["passed"] for r in group)))
    write_rows(args.output_dir/"trials.csv",rows)
    write_rows(args.output_dir/"summary.csv",summary)
    for r in summary:
        print(r)
