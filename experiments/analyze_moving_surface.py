"""Offline box-surface and fixed-arm first-order finger reach diagnostics."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

BOX_HALF_SIZE=np.array([.032,.021,.017])


def box_signed_distance(point,pose,half_size=BOX_HALF_SIZE):
    """Signed distance to an oriented box: negative inside, zero on surface."""
    rot=Rotation.from_quat(np.roll(np.asarray(pose[3:7],dtype=float),-1))
    local=rot.inv().apply(np.asarray(point,dtype=float)-np.asarray(pose[:3],dtype=float))
    q=np.abs(local)-np.asarray(half_size,dtype=float)
    return float(np.linalg.norm(np.maximum(q,0.))+min(float(np.max(q)),0.))


def local_finger_residual(jac,tip,target,q,lo,hi,horizon=.1,speed=2.):
    """Optimistic local surrogate at fixed arm; not a nonlinear feasibility certificate."""
    jac=np.asarray(jac,dtype=float);q=np.asarray(q,dtype=float)
    lo=np.maximum(np.asarray(lo,dtype=float)-q,-speed*horizon)
    hi=np.minimum(np.asarray(hi,dtype=float)-q,speed*horizon)
    if np.any(lo>hi+1e-10):return None,'joint_range_inconsistent'
    fixed=(hi-lo)<=1e-10;delta=np.zeros(4)
    delta[fixed]=(lo[fixed]+hi[fixed])/2
    error=np.asarray(target)-np.asarray(tip)-jac@delta
    if np.any(~fixed):
        result=lsq_linear(jac[:,~fixed],error,bounds=(lo[~fixed],hi[~fixed]),tol=1e-10,max_iter=300)
        if not result.success:return None,'solver_failed'
        delta[~fixed]=result.x
    residual=float(np.linalg.norm(jac@delta-(np.asarray(target)-np.asarray(tip))))
    return residual,'ok'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def pose(row,prefix='obj_'):
    if prefix=='episode_':return np.array([float(row[f'episode_obj{i}']) for i in range(7)])
    return np.array([float(row['obj_'+k]) for k in ('x','y','z','qw','qx','qy','qz')])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--horizon-ms',type=float,default=100.);p.add_argument('--speed-rad-s',type=float,default=2.)
    a=p.parse_args()
    if a.horizon_ms<=0 or a.speed_rad_s<=0:p.error('positive horizon and speed required')
    trials={};frames=defaultdict(list)
    for path in a.runs:
        for r in read(path/'trials.csv'):
            key=(int(r['seed']),int(r['scenario']),r['profile'],r['policy'])
            if key in trials:raise ValueError(f'duplicate {key}')
            trials[key]=r
        fpath=path/'frames.csv'
        if fpath.exists():
            for f in read(fpath):frames[(int(f['seed']),int(f['scenario']),f['profile'],f['policy'])].append(f)
    episodes=[];cases=[]
    for key,r in sorted(trials.items()):
        seq=sorted(frames.get(key,[]),key=lambda f:float(f['ms']))
        by_episode=defaultdict(list)
        for f in seq:
            if f['selected']!='':by_episode[int(f['episode'])].append(f)
        for e,fs in sorted(by_episode.items()):
            first=fs[0];digit=int(first['selected']);start=float(first['ms'])
            target=np.array([float(first['proxy_'+v]) for v in 'xyz'])
            if any(not np.allclose(target,[float(f['proxy_'+v]) for v in 'xyz'],atol=0,rtol=0) for f in fs):
                raise ValueError(f'frozen proxy changed: {key} {e}')
            tip=np.array([float(first[f'tip{i}']) for i in range(3)])
            J=np.array([[float(first[f'jac{i}{j}']) for j in range(4)] for i in range(3)])
            q=np.array([float(first[f'q{j}']) for j in range(4)])
            vel=np.array([float(first[f'v{j}']) for j in range(4)])
            low=np.array([float(first[f'lo{j}']) for j in range(4)])
            high=np.array([float(first[f'hi{j}']) for j in range(4)])
            residual,status=local_finger_residual(J,tip,target,q,low,high,a.horizon_ms/1000.,a.speed_rad_s)
            signed=box_signed_distance(target,pose(first,'episode_'))
            item=dict(seed=key[0],scenario=key[1],profile=key[2],policy=key[3],episode=e,finger=digit,
                start_ms=start,frames=len(fs),horizon_ms=a.horizon_ms,assumed_speed_rad_s=a.speed_rad_s,
                tip_proxy_start_mm=1000*np.linalg.norm(target-tip),
                proxy_surface_signed_start_mm=1000*signed,proxy_surface_abs_start_mm=1000*abs(signed),
                initial_joint_speed_max_rad_s=float(np.max(np.abs(vel))),
                initial_velocity_within_assumed_speed=int(np.max(np.abs(vel))<=a.speed_rad_s+1e-9),
                linear_residual_mm=None if residual is None else 1000*residual,linear_status=status,
                linear_within_5mm=None if residual is None else int(residual<=.005),
                digit_loaded_during=int(any(f['selected_loaded_after_step']=='1' for f in fs)),
                trial_held=r['held'],trial_qualified=r['qualified'])
            for future_ms in (50,100):
                future=next((f for f in seq if abs(float(f['ms'])-(start+future_ms))<1e-6),None)
                future_signed=None if future is None else box_signed_distance(target,pose(future))
                item[f'proxy_surface_signed_{future_ms}ms_mm']=None if future_signed is None else 1000*future_signed
                item[f'proxy_surface_abs_{future_ms}ms_mm']=None if future_signed is None else 1000*abs(future_signed)
            episodes.append(item)
        cases.append(dict(seed=key[0],scenario=key[1],profile=key[2],policy=key[3],accepted=r['accepted'],
                          held=r['held'],qualified=r['qualified'],episodes=len(by_episode),trace_frames=len(seq)))
    summary=[]
    for profile in ('normal','noisy'):
        for policy in ('shadow','geometry'):
            e=[r for r in episodes if r['profile']==profile and r['policy']==policy]
            c=[r for r in cases if r['profile']==profile and r['policy']==policy]
            valid=[r for r in e if r['linear_status']=='ok']
            eligible=[r for r in valid if r['initial_velocity_within_assumed_speed']]
            paired100=[r for r in e if r['proxy_surface_abs_100ms_mm'] is not None]
            def med(key):
                values=[r[key] for r in e if r[key] is not None]
                return median(values) if values else None
            summary.append(dict(profile=profile,policy=policy,horizon_ms=a.horizon_ms,assumed_speed_rad_s=a.speed_rad_s,trials=len(c),applied=sum(int(r['accepted']) for r in c),
                held=sum(int(r['held']) for r in c),qualified=sum(int(r['qualified']) for r in c),
                trials_with_selection=sum(r['episodes']>0 for r in c),episodes=len(e),
                surface_50ms_available=sum(r['proxy_surface_abs_50ms_mm'] is not None for r in e),
                surface_100ms_available=len(paired100),
                paired100_start_median_mm=median(r['proxy_surface_abs_start_mm'] for r in paired100) if paired100 else None,
                paired100_future_median_mm=median(r['proxy_surface_abs_100ms_mm'] for r in paired100) if paired100 else None,
                paired100_delta_median_mm=median(r['proxy_surface_abs_100ms_mm']-r['proxy_surface_abs_start_mm'] for r in paired100) if paired100 else None,
                median_surface_abs_start_mm=med('proxy_surface_abs_start_mm'),
                median_surface_abs_50ms_mm=med('proxy_surface_abs_50ms_mm'),
                median_surface_abs_100ms_mm=med('proxy_surface_abs_100ms_mm'),
                median_tip_proxy_start_mm=med('tip_proxy_start_mm'),
                linear_valid=len(valid),linear_within_5mm=sum(r['linear_within_5mm'] for r in valid),
                initial_speed_within_assumption=len(eligible),
                eligible_linear_within_5mm=sum(r['linear_within_5mm'] for r in eligible),
                median_linear_residual_mm=med('linear_residual_mm'),
                episodes_loaded_during=sum(r['digit_loaded_during'] for r in e)))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    save(a.output_dir/'episodes.csv',episodes);save(a.output_dir/'cases.csv',cases);save(a.output_dir/'summary.csv',summary)

if __name__=='__main__':main()
