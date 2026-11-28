"""Create paired summary of original, shadow and sustained-support trials."""
import argparse
import csv
from pathlib import Path
from statistics import median


def val(row,name):
    x=row.get(name,'')
    return None if x in ('','None') else float(x)


def same(a,b,name,tol=1e-9):
    x,y=val(a,name),val(b,name)
    return x is not None and y is not None and abs(x-y)<=tol


def save(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();groups={}
    for path in a.inputs:
        with path.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                key=(int(r['seed']),int(r['scenario']),r['profile'])
                g=groups.setdefault(key,{})
                if r['policy'] in g:raise ValueError(f'duplicate {key}')
                g[r['policy']]=r
    out=[]
    for (seed,case,profile),g in sorted(groups.items()):
        if set(g)!={'original','shadow','assist'}:raise ValueError(f'incomplete pair {seed} {case}')
        b=g['original']
        for policy in ('shadow','assist'):
            x=g[policy]
            same_fixture=all(same(b,x,k) for k in ('fixed_time','fixed_x','fixed_y','fixed_z','requested_speed_m_s'))
            pair=dict(seed=seed,scenario=case,profile=profile,policy=policy,
                same_fixture=int(same_fixture),baseline_applied=b['accepted'],candidate_applied=x['accepted'],
                baseline_held=b['held'],candidate_held=x['held'],baseline_qualified=b['qualified'],candidate_qualified=x['qualified'],
                rescued=int(b['held']=='0' and x['held']=='1'),lost=int(b['held']=='1' and x['held']=='0'),
                candidate_delivered_steps=x.get('support_assist_delivered_steps',''),
                candidate_torque_delta_Nm=x.get('support_assist_max_torque_delta_Nm',''))
            for field in ('impulse20_Ns','peak_contact20_accel','travel_mm_100','angle_deg_100'):
                bv,xv=val(b,field),val(x,field)
                pair['baseline_'+field]=bv;pair['candidate_'+field]=xv
                pair['delta_'+field]=None if bv is None or xv is None else xv-bv
            out.append(pair)
    summaries=[]
    for phase,seeds in [('development',{1261,1262}),('confirmation',{1271,1272})]:
        for profile in ('normal','noisy'):
            for policy in ('shadow','assist'):
                pairs=[r for r in out if r['seed'] in seeds and r['profile']==profile and r['policy']==policy]
                both=[r for r in pairs if r['baseline_held']=='1' and r['candidate_held']=='1']
                summary=dict(phase=phase,profile=profile,policy=policy,throws=len(pairs),
                    baseline_applied=sum(int(r['baseline_applied']) for r in pairs),candidate_applied=sum(int(r['candidate_applied']) for r in pairs),
                    baseline_held=sum(int(r['baseline_held']) for r in pairs),candidate_held=sum(int(r['candidate_held']) for r in pairs),
                    baseline_qualified=sum(int(r['baseline_qualified']) for r in pairs),candidate_qualified=sum(int(r['candidate_qualified']) for r in pairs),
                    rescued=sum(r['rescued'] for r in pairs),lost=sum(r['lost'] for r in pairs),same_fixture=sum(r['same_fixture'] for r in pairs),
                    both_held=len(both),delivered_cases=sum(int(r['candidate_delivered_steps'] or 0)>0 for r in pairs))
                for field in ('impulse20_Ns','peak_contact20_accel','travel_mm_100','angle_deg_100'):
                    ds=[r['delta_'+field] for r in both if r['delta_'+field] is not None]
                    summary['median_delta_'+field]=median(ds) if ds else None
                summaries.append(summary)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    save(a.output_dir/'pairs.csv',out);save(a.output_dir/'summary.csv',summaries)

if __name__=='__main__':main()
