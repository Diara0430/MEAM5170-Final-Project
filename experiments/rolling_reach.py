"""Pair a rolling controller with the same controller plus a 16 mm recovery aim."""
import argparse
from dataclasses import replace
from pathlib import Path

from controllers.deadline_arm_filter import DeadlineArmFilter
from controllers.lock_timing import RULES, TimingMPC
from controllers.palm_aim import PalmAimRecovery
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.rolling_success import POLICIES, StudyTrack
from scene_setup.scene import build_scene


def compare(model, seed, case, start, velocity, profile, core, recovery, *, selector_factory=None):
    factors = set(core.split("_"))
    observer = planner = None

    def sensor(gravity, noise_seed):
        nonlocal observer
        observer = StudyTrack(gravity, noise_seed, seed=seed, case=case, profile=profile,
                              long_track="track" in factors)
        return observer

    def tracker(m, obs, enabled):
        out = DeadlineArmFilter(m, obs, enabled, fast=True, refine=True)
        if "reserve" in factors:
            out.config = replace(out.config,speed=.985)
        return out

    def controller(*args, **kwargs):
        nonlocal planner
        cls = PalmAimRecovery if recovery else TimingMPC
        if selector_factory is not None:
            kwargs["contact_selector"] = selector_factory(model, observer)
        planner = cls(*args, observer=observer, rule=RULES["rolling"], **kwargs,
                      planning_speed_limit=.97 if "reserve" in factors else 1.,
                      orientation_tolerance=.085 if "wrist" in factors else .03)
        return planner

    row = trial(model,seed,case,start,velocity,True,tracker_factory=tracker,
                observer_factory=sensor,planner_factory=controller)
    row.update(profile=profile, policy=core+("_edge" if recovery else ""),
               camera_max_ms=observer.sample_max_ms, offset_trials=getattr(planner,"offset_trials",0),
               offset_deliveries=getattr(planner,"offset_deliveries",0),
               first_offset_s=getattr(planner,"first_offset",None))
    if selector_factory is not None:
        row.update(planner.contact_selector.summary())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("normal","noisy","corrected"), required=True)
    parser.add_argument("--core", choices=POLICIES, required=True)
    parser.add_argument("--recovery-only", action="store_true", help="Run just the recovery policy")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows = []
    for case,p,v,*_ in throws(center,model.opt.gravity,args.seed):
        options = [("rolling",False), (args.core,False), (args.core,True)]
        if args.core == "rolling":
            options = options[1:]
        if args.recovery_only:
            options = [(args.core,True)]
        shift = (args.seed+case) % len(options)
        rows.extend(compare(model,args.seed,case,p,v,args.profile,core,on)
                    for core,on in options[shift:]+options[:shift])
    save_csv(args.output_dir/"trials.csv",rows)


if __name__ == "__main__":
    main()
