"""Summarize selection episodes and subsequent finger loading from trace CSVs."""
import argparse
import csv
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def groups(value):return {int(x) for x in value.split('|') if x}


def followup_contact(seq,finger,start,end):
    """Return first load, first 6 ms sustained load, and whether 50 ms was recorded."""
    within=[r for r in seq if start<=float(r['ms'])<=end+50.+1e-8]
    touches=[r for r in within if finger in groups(r['post_groups'])]
    streak=0;last_ms=None;first_stable=None
    for r in within:
        t=float(r['ms']);loaded=finger in groups(r['post_groups'])
        streak=(streak+1 if last_ms is not None and abs(t-last_ms-2.)<1e-6 else 1) if loaded else 0
        last_ms=t
        if streak>=3:first_stable=t;break
    return (float(touches[0]['ms']) if touches else None,first_stable,
            bool(seq) and float(seq[-1]['ms'])>=end+50.-1e-6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();trials={};frames=defaultdict(list)
    for directory in a.runs:
        for r in read(directory/'trials.csv'):
            key=(int(r['seed']),int(r['scenario']),r['profile'],r['policy'])
            if key in trials:raise ValueError(f'Duplicate trial {key}')
            trials[key]=r
        path=directory/'frames.csv'
        if path.exists():
            for f in read(path):frames[(int(f['seed']),int(f['scenario']),f['profile'],f['policy'])].append(f)
    episodes=[];case_rows=[]
    for key,trial in sorted(trials.items()):
        seed,case,profile,policy=key
        if policy not in ('shadow','geometry'):continue
        seq=sorted(frames.get(key,[]),key=lambda r:float(r['ms']))
        if any(float(x['ms'])>=float(y['ms']) for x,y in zip(seq,seq[1:])):raise ValueError(f'Nonmonotonic frames {key}')
        if any(r['selected'] and r['reason']!='selected' for r in seq):raise ValueError(f'Inconsistent selected state {key}')
        reasons=Counter(r['reason'] for r in seq)
        by_episode=defaultdict(list)
        for f in seq:
            if f['selected']!='':by_episode[int(f['episode'])].append(f)
        trial_episodes=[]
        for episode,fs in sorted(by_episode.items()):
            finger=int(fs[0]['selected'])
            if any(int(f['selected'])!=finger for f in fs):raise ValueError(f'Finger changed within episode {key}')
            first,last=fs[0],fs[-1];end=float(last['ms']);start=float(first['ms'])
            # 指尖距离一个固定的、选中时冻结的启发式目标；仍可能因机械臂运动变化。
            start_dist=float(first['pre_proxy_distance_m']);distances=[float(r['post_proxy_distance_m']) for r in fs]
            gaps=[float(r['exact_digit_object_gap_m']) for r in fs if r['exact_digit_object_gap_m'] not in ('','None')]
            first_loaded,first_stable,full_followup=followup_contact(seq,finger,start,end)
            item=dict(seed=seed,scenario=case,profile=profile,policy=policy,episode=episode,finger=finger,
                start_ms=start,end_ms=end,frames=len(fs),start_fixed_proxy_m=start_dist,
                end_fixed_proxy_m=distances[-1],min_fixed_proxy_m=min(distances),
                max_approach_to_fixed_proxy_mm=1000*(start_dist-min(distances)),
                end_change_from_fixed_proxy_mm=1000*(distances[-1]-start_dist),
                min_digit_object_gap_mm=1000*min(gaps) if gaps else None,
                loaded_during_episode=int(any(r['selected_loaded_after_step']=='1' for r in fs)),
                loaded_by_50ms_followup=int(first_loaded is not None),first_loaded_ms=first_loaded,
                stable_6ms_by_followup=int(first_stable is not None),first_stable_ms=first_stable,
                full_50ms_followup=int(full_followup),
                trial_held=trial['held'],trial_qualified=trial['qualified'])
            episodes.append(item);trial_episodes.append(item)
        case_rows.append(dict(seed=seed,scenario=case,profile=profile,policy=policy,
            accepted=trial['accepted'],contacted=trial['contacted'],held=trial['held'],qualified=trial['qualified'],
            trace_frames=len(seq),episodes=len(trial_episodes),selected_frames=reasons['selected'],
            no_loaded_contact_frames=reasons['no_loaded_contact'],ambiguous_normal_frames=reasons['ambiguous_normal'],
            opposing_normals_frames=reasons['opposing_normals'],no_digit_selected_frames=reasons['no_digit_selected'],
            outside_window_frames=reasons['outside_window'],
            episodes_loaded_by_followup=sum(x['loaded_by_50ms_followup'] for x in trial_episodes),
            episodes_stable_by_followup=sum(x['stable_6ms_by_followup'] for x in trial_episodes),
            episodes_reached_5mm_proxy=sum(x['max_approach_to_fixed_proxy_mm']>=5. for x in trial_episodes),
            delivered_steps=trial.get('geometry_delivered_steps',''),
            selected_mask=trial.get('geometry_selected_mask','')))
    summary=[]
    for profile in ('normal','noisy'):
        for policy in ('shadow','geometry'):
            cases=[r for r in case_rows if r['profile']==profile and r['policy']==policy]
            eps=[r for r in episodes if r['profile']==profile and r['policy']==policy]
            summary.append(dict(profile=profile,policy=policy,trials=len(cases),applied=sum(int(r['accepted']) for r in cases),
                held=sum(int(r['held']) for r in cases),qualified=sum(int(r['qualified']) for r in cases),
                trials_with_episode=sum(r['episodes']>0 for r in cases),episodes=len(eps),
                episodes_loaded_during=sum(r['loaded_during_episode'] for r in eps),
                episodes_loaded_by_followup=sum(r['loaded_by_50ms_followup'] for r in eps),
                episodes_stable_by_followup=sum(r['stable_6ms_by_followup'] for r in eps),
                episodes_approach_5mm=sum(r['max_approach_to_fixed_proxy_mm']>=5 for r in eps),
                episodes_full_followup=sum(r['full_50ms_followup'] for r in eps),
                median_proxy_approach_mm=median(r['max_approach_to_fixed_proxy_mm'] for r in eps) if eps else None,
                median_min_object_gap_mm=median(r['min_digit_object_gap_mm'] for r in eps if r['min_digit_object_gap_mm'] is not None) if eps else None,
                selected_frames=sum(r['selected_frames'] for r in cases),
                no_loaded_contact_frames=sum(r['no_loaded_contact_frames'] for r in cases),
                ambiguous_normal_frames=sum(r['ambiguous_normal_frames'] for r in cases),
                opposing_normals_frames=sum(r['opposing_normals_frames'] for r in cases),
                no_digit_selected_frames=sum(r['no_digit_selected_frames'] for r in cases)))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    save(a.output_dir/'episodes.csv',episodes);save(a.output_dir/'cases.csv',case_rows);save(a.output_dir/'summary.csv',summary)

if __name__=='__main__':main()
