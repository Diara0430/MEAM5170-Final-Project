"""Paired online lock timing study with matched launches and delayed position frames."""
import argparse
import csv
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np

from controllers.ball_observer import BallObserver
from controllers.deadline_arm_filter import DeadlineArmFilter
from controllers.lock_timing import TimingMPC, RULES, crossing
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


FIXED = ("clock26", "clock44", "clock62", "lead35")
ADAPTIVE = ("stable4", "stable5", "confidence4")
PROFILES = ("normal", "noisy")


class StudyObserver(BallObserver):
    def __init__(self, gravity, noise_seed, *, profile, seed, case):
        noisy = profile != "normal"
        super().__init__(gravity, noise_seed, noise_m=.004 if noisy else .002,
                         latency_s=.040 if noisy else .020)
        self.correction = None
        self.bias = np.zeros(3)
        if profile == "corrected":
            rng = np.random.default_rng(np.random.SeedSequence([seed, case, 938]))
            angle = rng.uniform(0.,2*np.pi)
            self.bias[:2] = .032*np.array([np.cos(angle),np.sin(angle)])
            self.correction = rng.uniform(.46,.60)

    def sample(self, now, position):
        if self.correction is not None and now < self.correction:
            position = position+self.bias
        return super().sample(now,position)


def choose(paths):
    table = {}
    for path in paths:
        with path.open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["seed"]), int(row["scenario"]), row["profile"], row["policy"])
                if key in table:
                    raise ValueError("Duplicated development row")
                table[key] = row
    expected = set(product((931,932,933),range(8),PROFILES,("rolling",*FIXED,*ADAPTIVE)))
    if set(table) != expected:
        raise ValueError("Selection requires a complete development set")
    def rank(policy):
        groups = [[r for (*_, profile, name),r in table.items() if name==policy and profile==p]
                  for p in PROFILES]
        counts = [sum(int(r["qualified"]) for r in g) for g in groups]
        holds = sum(int(r["held"]) for g in groups for r in g)
        return -sum(counts), -min(counts), -holds, (*FIXED,*ADAPTIVE).index(policy)
    return min(FIXED,key=rank), min(ADAPTIVE,key=rank)


def run(model, seed, case, p, v, profile, label):
    reversible = label.endswith("_release")
    name = label.removesuffix("_release")
    observer = controller = None
    def make_observer(gravity, noise_seed):
        nonlocal observer
        observer = StudyObserver(gravity, noise_seed, profile=profile, seed=seed, case=case)
        return observer
    def make_planner(*args, **kwargs):
        nonlocal controller
        controller = TimingMPC(*args, observer=observer, rule=RULES[name], reversible=reversible, **kwargs)
        return controller

    latest = previous_estimate = previous_frame = truth_error = touch_error = None
    max_xy = max_abs_time = 0.
    n_after = 0
    def record(m,d,info,sensor,tracker):
        nonlocal latest,previous_estimate,previous_frame,truth_error,touch_error,max_xy,max_abs_time,n_after
        if controller.lock_s is None:
            return
        now = float(d.time)
        stamp, point = controller.lock_target_s, controller.lock_point
        if info["first_contact_s"] is not None:
            if touch_error is None:
                touch_error = 1000*float(np.linalg.norm(capture_center(d)-point))
            return
        # Truth is read only by the experiment logger after the physics step.
        ball = d.joint("flight").qpos[:3].copy()
        if previous_frame is not None and previous_frame[0] <= stamp <= now and now>previous_frame[0]:
            t0,p0 = previous_frame
            at = p0 + (ball-p0)*(stamp-t0)/(now-t0)
            truth_error = 1000*float(np.linalg.norm(at-point))
        previous_frame = now,ball
        estimate = sensor.estimate
        if estimate is None or estimate.time == previous_estimate or estimate.time < controller.lock_s-1e-8:
            return
        previous_estimate = estimate.time
        pred = crossing(estimate, point[2])
        if pred is None:
            return
        dt_ms = 1000*(pred[0]-stamp)
        xy_mm = 1000*float(np.linalg.norm(pred[1][:2]-point[:2]))
        max_xy = max(max_xy,xy_mm)
        max_abs_time = max(max_abs_time,abs(dt_ms))
        latest = xy_mm,dt_ms
        n_after += 1

    row = trial(model,seed,case,p,v,True,tracker_factory=partial(DeadlineArmFilter,fast=True,refine=True),
                observer_factory=make_observer,planner_factory=make_planner,record_frame=record)
    contact = row["first_contact_s"]
    lock = controller.lock_s
    metrics = controller.lock_metrics or (None,)*4
    row.update(profile=profile,policy=label,decision_s=controller.decision_s,lock_s=lock,
               lock_x=None if lock is None else controller.lock_point[0],
               lock_y=None if lock is None else controller.lock_point[1],
               lock_z=None if lock is None else controller.lock_point[2],
               locked_intercept_s=controller.lock_target_s,
               predicted_lead_s=None if lock is None else controller.lock_target_s-lock,
               contact_lead_s=None if lock is None or contact is None else contact-lock,
               spread_mm=None if metrics[0] is None else 1000*metrics[0],
               spread_ms=None if metrics[1] is None else 1000*metrics[1],
               confidence_mm=None if metrics[2] is None else 1000*metrics[2],
               confidence_ms=None if metrics[3] is None else 1000*metrics[3],
               postlock_frames=n_after,last_forecast_xy_mm=None if latest is None else latest[0],
               last_forecast_dt_ms=None if latest is None else latest[1],
               max_forecast_xy_mm=max_xy if n_after else None,max_forecast_dt_ms=max_abs_time if n_after else None,
               actual_ball_error_at_locked_time_mm=truth_error,palm_to_original_lock_at_touch_mm=touch_error,
               unlock_s=controller.unlock_s,replacement_applied_s=controller.replacement_s,
               correction_s=observer.correction)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--profile",choices=(*PROFILES,"corrected"),required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--selection",nargs="+",type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.selection:
        clock, adaptive = choose(args.selection)
        policies = ("rolling", clock, adaptive, adaptive+"_release")
    else:
        policies = ("rolling", *FIXED, *ADAPTIVE)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case,p,v,*_ in throws(center,model.opt.gravity,args.seed):
        shift = case%len(policies)
        for policy in policies[shift:]+policies[:shift]:
            row = run(model,args.seed,case,p,v,args.profile,policy)
            if row["lock_s"] is not None and abs(row["lock_s"]-row["decision_s"]-.020)>1e-7:
                raise AssertionError("Lock must start on application of a timely plan")
            if row["unlock_s"] is not None and row["unlock_s"]+1e-8 < row["lock_s"]:
                raise AssertionError("Release before lock")
            rows.append(row)
        print(args.seed,args.profile,case,flush=True)
    save_csv(args.output_dir/"trials.csv",rows)


if __name__ == "__main__":
    main()
