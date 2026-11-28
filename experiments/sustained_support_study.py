"""Paired fixed-intercept cuboid trials: original, shadow and support assist."""
import argparse
import csv
from pathlib import Path
from unittest.mock import patch
from controllers.sustained_support_grasp import SustainedSupportGrasp
from experiments import fixed_angle_study
from experiments.box_contact_window import ContactWindow
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center,throws
from experiments.scoped_calibration import ScopedCalibrationMPC
from scene_setup.scene import build_scene


def run_one(model,seed,case,p,v,profile,policy):
    audit=ContactWindow(model); hand=None
    original=fixed_angle_study.trial
    def trial_wrapper(*args,**kwargs):
        nonlocal hand
        original_record=kwargs.get('record_frame')
        if original_record is not None:
            def record(m,d,info,observer,tracker):
                original_record(m,d,info,observer,tracker)
                audit.record(m,d,info,observer,tracker)
            kwargs['record_frame']=record
            if policy!='original':
                def factory(model,actuators):
                    nonlocal hand
                    hand=SustainedSupportGrasp(model,actuators,shadow=policy=='shadow')
                    return hand
                kwargs['hand_controller_factory']=factory
        return original(*args,**kwargs)
    with patch.object(fixed_angle_study,'trial',trial_wrapper),patch.object(fixed_angle_study,'CalibrationMPC',ScopedCalibrationMPC):
        row=fixed_angle_study.run(model,seed,case,p,v,profile,[0.])[0]
    row.update(policy=policy,**audit.summary())
    if hand is not None:row.update(hand.diagnostics())
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--profile',choices=('normal','noisy'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=8)
    args=parser.parse_args();m=build_scene('box');center=capture_center(initial(m));rows=[]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    policies=['original','shadow','assist']
    for case,p,v,*_ in throws(center,m.opt.gravity,args.seed):
        if case>=args.cases:continue
        shift=(args.seed+case)%len(policies)
        for policy in policies[shift:]+policies[:shift]:
            row=run_one(m,args.seed,case,p,v,args.profile,policy);rows.append(row)
            print(args.seed,args.profile,case,policy,'applied',row['accepted'],'held',row['held'],
                  'qualified',row['qualified'],'assist steps',row.get('support_assist_delivered_steps',0),flush=True)
        columns=list(dict.fromkeys(k for row in rows for k in row))
        with (args.output_dir/'trials.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
