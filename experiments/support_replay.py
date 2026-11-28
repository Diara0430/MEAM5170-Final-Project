"""Record original and support-assist catches for a selected cuboid throw."""
import argparse
import gzip
import json
from pathlib import Path
from unittest.mock import patch
from experiments.box_contact_window import ContactWindow
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center,throws
from experiments import sustained_support_study
from scene_setup.scene import build_scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--case',type=int,choices=range(8),required=True)
    p.add_argument('--profile',choices=('normal','noisy'),required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--render',action='store_true')
    args=p.parse_args();model=build_scene('box');center=capture_center(initial(model))
    _,pos,vel,*_=next(t for t in throws(center,model.opt.gravity,args.seed) if t[0]==args.case)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for policy in ('original','assist'):
        frames=[]
        class Recorder(ContactWindow):
            def record(self,m,d,info,observer,tracker):
                super().record(m,d,info,observer,tracker)
                t=float(d.time);dt=m.opt.timestep
                if round(t/dt)%round(.020/dt):return
                last=observer.frames[-1] if observer.frames else None
                fit=observer.estimate if info['first_contact_s'] is None else None
                plan=info.get('applied_reference')
                frames.append(dict(time_s=t,ball=d.joint('flight').qpos[:3].tolist(),hand=capture_center(d).tolist(),
                    qpos=d.qpos.round(8).tolist(),forecast=None if fit is None else [fit.time,fit.position.tolist(),fit.velocity.tolist()],
                    measurement_time=None if last is None else last[0],measurement=None if last is None else last[1].tolist(),
                    intercept=None if plan is None else plan.point.tolist()))
        with patch.object(sustained_support_study,'ContactWindow',Recorder):
            row=sustained_support_study.run_one(model,args.seed,args.case,pos,vel,args.profile,policy)
        sample=dict(shape='box',seed=args.seed,scenario=args.case,profile=args.profile,method='fixed',grasp=policy,
                    held=int(row['held']),qualified=int(row['qualified']),contact=row['first_contact_s'],
                    first_estimate={'g':model.opt.gravity.tolist()},path=frames)
        dest=args.output_dir/f'{policy}_replay.json.gz'
        with gzip.open(dest,'wt',encoding='utf-8') as f:json.dump(sample,f)
        if args.render:
            from experiments.render_observed_arc import render
            render(sample,args.output_dir/f'{policy}.gif')
        print(policy,'held',row['held'],'qualified',row['qualified'],'saved',dest,flush=True)

if __name__=='__main__':main()
