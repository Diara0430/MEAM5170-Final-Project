"""Record the physical cuboid state, observed track, chosen intercept and palm for replay."""
import argparse
import gzip
import json
from pathlib import Path

from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.render_observed_arc import render
from experiments.shape_catch import METHODS, TRANSFER_METHODS, simulate
from scene_setup.scene import build_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", type=int, choices=range(8), required=True)
    parser.add_argument("--profile", choices=("normal", "noisy", "corrected"), required=True)
    parser.add_argument("--method", choices=METHODS+TRANSFER_METHODS, default="pose")
    parser.add_argument("--grasp", choices=("original","aperture","aperture_slip"), default="original")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = build_scene("box")
    frames = []
    center = capture_center(initial(model))
    def record(model, data, info, observer, tracker):
        t = float(data.time)
        onset = info["first_contact_s"]
        stride = .040 if onset is None or t < onset+.16 else .100
        if round(t/model.opt.timestep) % round(stride/model.opt.timestep):
            return
        last = observer.frames[-1] if observer.frames else None
        fit = observer.estimate if onset is None else None
        plan = info.get("applied_reference")
        frames.append(dict(time_s=t, ball=data.joint("flight").qpos[:3].tolist(),
                           hand=capture_center(data).tolist(), qpos=data.qpos.round(8).tolist(),
                           forecast=None if fit is None else [fit.time,fit.position.tolist(),fit.velocity.tolist()],
                           measurement_time=None if last is None else last[0],
                           measurement=None if last is None else last[1].tolist(),
                           intercept=None if plan is None else plan.point.tolist()))
    case, p, v, *_ = next(r for r in throws(center,model.opt.gravity,args.seed) if r[0]==args.case)
    row = simulate(model, args.seed, case, p, v, args.profile, args.method,
                   grasp=args.grasp, record_frame=record)
    sample = dict(shape="box", seed=args.seed, scenario=case, profile=args.profile,
                  method=args.method, grasp=args.grasp, held=int(row["held"]), qualified=int(row["qualified"]),
                  contact=row["first_contact_s"], first_estimate={"g":model.opt.gravity.tolist()}, path=frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output_dir/"replay.json.gz", "wt", encoding="utf-8") as f:
        json.dump(sample, f)
    render(sample, args.output_dir/"catch.gif")
    print("Replay outcome", {k: row[k] for k in ("held","qualified","peak_accel","drift_mm","spin_rad_s")})


if __name__ == "__main__":
    main()
