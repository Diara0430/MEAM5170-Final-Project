"""Paired full-grid fixture calibration with and without OSQP workspace reuse."""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch
from controllers.catching_mpc import CatchingMPC
from controllers.fixed_intercept import CalibrationMPC
from experiments.reuse_calibration import ReuseCalibrationMPC
from experiments.fixed_angle_study import run
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


def compare(model,seed,case,p,v,profile):
    methods=[('grid',CalibrationMPC),('reuse',ReuseCalibrationMPC)]
    if (seed+case)%2: methods.reverse()
    rows=[]
    for name,cls in methods:
        instances=[]
        class Captured(cls):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                instances.append(self)
        counters=defaultdict(float)
        original=CatchingMPC.solve
        def timed(self,*a,**kw):
            if not isinstance(self,CalibrationMPC): return original(self,*a,**kw)
            start=time.perf_counter()
            try: return original(self,*a,**kw)
            finally:
                label='speed' if kw.get('speed_request') is not None else 'anchor'
                counters[label+'_ms']+=(time.perf_counter()-start)*1000
                counters[label+'_calls']+=1
        with patch.object(CatchingMPC,'solve',timed), patch('experiments.fixed_angle_study.CalibrationMPC',Captured):
            row=run(model,seed,case,p,v,profile,[0.])[0]
        f=instances[0].fixture
        row.update(method=name, levels='|'.join(format(v,'.17g') for v in f.levels),
                   anchor_solve_ms=counters['anchor_ms'],speed_solve_ms=counters['speed_ms'],
                   speed_solve_calls=int(counters['speed_calls']),
                   reuse_hits=getattr(instances[0],'cache_hits',0),reuse_misses=getattr(instances[0],'cache_misses',0))
        rows.append(row)
        print(f'{seed} {profile} {case} {name} {f.reason} prep={f.preparation_ms:.1f}ms levels={len(f.levels)} hits={row["reuse_hits"]} held={row["held"]}',flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--cases',type=int,default=8)
    p.add_argument('--profile',choices=('normal','noisy'),required=True)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args()
    m=build_scene('box'); center=capture_center(initial(m)); rows=[]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for case,pos,vel,*_ in throws(center,m.opt.gravity,a.seed):
        if case>=a.cases: continue
        rows.extend(compare(m,a.seed,case,pos,vel,a.profile))
        columns=list(dict.fromkeys(k for row in rows for k in row))
        with (a.output_dir/'trials.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)

if __name__=='__main__': main()
