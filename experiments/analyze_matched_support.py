"""Verify matched arm commands and summarize contact-point audit cases."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cases',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in a.cases:
        trials={r['policy']:r for r in read(path/'trials.csv')}
        if set(trials)!={'original','shadow','assist'}:raise ValueError('Expected all three policies')
        commands=defaultdict(dict)
        for r in read(path/'arm_commands.csv'):
            step=int(r['step']);policy=r['policy']
            if step in commands[policy]:raise ValueError('Duplicate arm command')
            commands[policy][step]=tuple(float(r[f'arm_ctrl{i}']) for i in range(1,8))
        frames=defaultdict(dict)
        for r in read(path/'frames.csv'):
            frames[r['policy']][float(r['ms'])]=r
        for policy in ('shadow','assist'):
            ref=trials[policy]
            if int(ref['missing_arm_schedule_steps']) or int(ref['mismatched_written_arm_commands']) or int(ref['mpc_schedule_decision_mismatches']):
                raise ValueError(f'Schedule mismatch {path} {policy}')
            if not commands[policy] or any(commands['original'].get(k)!=v for k,v in commands[policy].items()):
                raise ValueError(f'Arm command mismatch {path} {policy}')
        common=set(frames['original']) & set(frames['shadow'])
        keys=['rel_x_mm','rel_y_mm','rel_z_mm']+[f'q{i}_rad' for i in range(16)]+[f'ctrl{i}_rad' for i in range(16)]
        if not common or any(frames['original'][t][k]!=frames['shadow'][t][k] for t in common for k in keys):
            raise ValueError(f'Shadow differs in captured state {path}')
        for policy in ('original','shadow','assist'):
            r=trials[policy]
            valid=[f for t,f in frames[policy].items() if 20<=t<=100 and f['max_contact_tangent_speed_m_s'] not in ('','None')]
            loads=[float(f['normal_force_sum_N']) for t,f in frames[policy].items() if 20<=t<=100]
            rows.append(dict(case=path.name,policy=policy,held=r['held'],qualified=r['qualified'],
                first_contact_s=r['first_contact_s'],peak_accel_rad_s2=r['peak_accel'],
                translation_100mm=r['travel_mm_100'],rotation_100deg=r['angle_deg_100'],
                opposition_in_100ms=r['first_opposition_ms'],floor_ms=r['floor_ms'],
                median_per_step_max_tangent_speed_m_s=median(float(f['max_contact_tangent_speed_m_s']) for f in valid) if valid else None,
                tangent_speed_frames=len(valid),median_normal_force_sum_N=median(loads) if loads else None,
                arm_override_steps=r['arm_overrides'],planner_extensions=r['mpc_schedule_extension_count']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Verified schedules and shadow state for {len(a.cases)} selected cases')

if __name__=='__main__':main()
