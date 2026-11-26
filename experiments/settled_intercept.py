"""Compare forecast agreement alone against agreement of executable targets."""
import argparse
from functools import partial
from pathlib import Path

from controllers.deadline_arm_filter import DeadlineArmFilter
from controllers.lock_timing import TimingMPC, RULES
from controllers.settled_intercept import SettledInterceptMPC
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.lock_timing import StudyObserver
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


def run(model,seed,case,p,v,profile,name):
    sensor = planner = None
    def make_sensor(g, noise_seed):
        nonlocal sensor
        sensor = StudyObserver(g,noise_seed,profile=profile,seed=seed,case=case)
        return sensor
    def make_planner(*args,**kwargs):
        nonlocal planner
        cls = SettledInterceptMPC if name=="confirmed" else TimingMPC
        rule = RULES["stable5" if name=="confirmed" else name]
        planner = cls(*args,observer=sensor,rule=rule,**kwargs)
        return planner
    row = trial(model,seed,case,p,v,True,tracker_factory=partial(DeadlineArmFilter,fast=True,refine=True),
                observer_factory=make_sensor,planner_factory=make_planner)
    row.update(profile=profile, policy=name, decision_s=planner.decision_s, lock_s=planner.lock_s,
               contact_lead_s=None if planner.lock_s is None or row["first_contact_s"] is None
               else row["first_contact_s"]-planner.lock_s,
               gate_checks=getattr(planner,"gate_checks",0),gate_rejections=getattr(planner,"gate_rejections",0))
    return row


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--profile",choices=("normal","noisy"),required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model=build_scene("ball")
    center=capture_center(initial(model))
    policies=("rolling","clock44","stable5","confirmed")
    rows=[]
    for case,p,v,*_ in throws(center,model.opt.gravity,args.seed):
        shift=case%len(policies)
        rows.extend(run(model,args.seed,case,p,v,args.profile,policy)
                    for policy in policies[shift:]+policies[:shift])
    save_csv(args.output_dir/"trials.csv",rows)
