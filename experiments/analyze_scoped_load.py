"""Summarize paired scoped-planner trials by synthetic-load condition."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def write_rows(path,rows):
    if not rows: raise ValueError('No rows to write')
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();groups={}
    for directory in args.runs:
        metadata=json.loads((directory/'load_metadata.json').read_text(encoding='utf-8'))
        if metadata['child_exit_code']!=0: raise ValueError(f'Failed run: {directory}')
        workers=metadata['load_workers']
        if workers and (len(metadata['worker_cpu_seconds'])!=workers or any(x is None or x<=0 for x in metadata['worker_cpu_seconds'])):
            raise ValueError(f'Unverified worker load: {directory}')
        for row in read_rows(directory/'trials.csv'):
            key=(workers,row['profile'],row['seed'],row['scenario'])
            group=groups.setdefault(key,{})
            if row['method'] in group: raise ValueError(f'Duplicate {key}')
            group[row['method']]=row
    pairs=[]
    for (workers,profile,seed,case),g in sorted(groups.items()):
        if set(g)!={'grid','reuse'}: raise ValueError(f'Incomplete pair {(workers,profile,seed,case)}')
        a,b=g['grid'],g['reuse']
        timed=all(int(g[m]['speed_solve_calls'])>0 for m in ('grid','reuse'))
        same_anchor=timed and all(a[k]==b[k] for k in ('fixed_time','fixed_x','fixed_y','fixed_z'))
        def levels(r): return [float(v) for v in r['levels'].split('|') if v]
        av,bv=levels(a),levels(b)
        approximate=same_anchor and len(av)==len(bv) and all(abs(x-y)<=1e-9 for x,y in zip(av,bv))
        pairs.append(dict(workers=workers,profile=profile,seed=seed,scenario=case,
            grid_status=a['fixture_status'],scoped_status=b['fixture_status'],
            grid_held=a['held'],scoped_held=b['held'],grid_qualified=a['qualified'],scoped_qualified=b['qualified'],
            grid_applied=a['accepted'],scoped_applied=b['accepted'],both_completed_grid=int(timed),
            same_anchor=int(same_anchor),levels_exact=int(same_anchor and a['levels']==b['levels']),
            levels_within_1e9=int(approximate),target_exact=int(same_anchor and a['requested_speed_m_s']==b['requested_speed_m_s']),
            grid_ms=a['calibration_ms'] if timed else '',scoped_ms=b['calibration_ms'] if timed else '',
            paired_saving_ms=float(a['calibration_ms'])-float(b['calibration_ms']) if timed else ''))
    summary=[]
    for workers,profile in sorted({(r['workers'],r['profile']) for r in pairs}):
        r=[x for x in pairs if (x['workers'],x['profile'])==(workers,profile)]
        matched=[x for x in r if x['same_anchor']]
        a=[float(x['grid_ms']) for x in r if x['both_completed_grid']]
        b=[float(x['scoped_ms']) for x in r if x['both_completed_grid']]
        summary.append(dict(workers=workers,profile=profile,throws=len(r),
            grid_ready=sum(x['grid_status']=='ready' for x in r),scoped_ready=sum(x['scoped_status']=='ready' for x in r),
            grid_timeout=sum(x['grid_status']=='calibration_deadline' for x in r),scoped_timeout=sum(x['scoped_status']=='calibration_deadline' for x in r),
            grid_no_anchor=sum(x['grid_status']=='no_feasible_anchor' for x in r),scoped_no_anchor=sum(x['scoped_status']=='no_feasible_anchor' for x in r),
            grid_no_speed=sum(x['grid_status']=='no_certified_speed' for x in r),scoped_no_speed=sum(x['scoped_status']=='no_certified_speed' for x in r),
            grid_held=sum(int(x['grid_held']) for x in r),scoped_held=sum(int(x['scoped_held']) for x in r),
            grid_qualified=sum(int(x['grid_qualified']) for x in r),scoped_qualified=sum(int(x['scoped_qualified']) for x in r),
            grid_applied_not_held=sum(int(x['grid_applied']) and not int(x['grid_held']) for x in r),
            scoped_applied_not_held=sum(int(x['scoped_applied']) and not int(x['scoped_held']) for x in r),
            completed_pairs=sum(x['both_completed_grid'] for x in r),same_anchor=len(matched),
            levels_exact=sum(x['levels_exact'] for x in matched),levels_within_1e9=sum(x['levels_within_1e9'] for x in matched),
            target_exact=sum(x['target_exact'] for x in matched),grid_median_ms=median(a) if a else None,
            scoped_median_ms=median(b) if b else None,paired_median_saving_ms=median(x['paired_saving_ms'] for x in matched) if matched else None))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_rows(args.output_dir/'pairs.csv',pairs);write_rows(args.output_dir/'summary.csv',summary)

if __name__=='__main__': main()
