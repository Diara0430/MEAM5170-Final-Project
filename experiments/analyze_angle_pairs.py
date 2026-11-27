"""Create auditable same-throw contact comparisons against the zero-degree baseline."""
import argparse
import csv
from pathlib import Path
from statistics import median


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def value(row, key):
    raw = row.get(key, '')
    return None if raw in ('', 'None') else float(raw)


def save(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--speed-tolerance', type=float, default=.05)
    args = parser.parse_args()
    if args.speed_tolerance < 0:
        parser.error('speed tolerance must be nonnegative')
    groups = {}
    for path in args.inputs:
        for row in read(path):
            key = (row['seed'], row['scenario'], row['profile'])
            angle = float(row['angle_deg'])
            group = groups.setdefault(key, {})
            if angle in group:
                raise ValueError(f'duplicate angle for {key}')
            group[angle] = row
    pairs = []
    metrics = ('axis6_peak_contact20_rad_s2','peak_contact20_accel','impact20_peak_N',
               'impulse20_Ns','relative_travel_100ms_mm','relative_rotation_100ms_deg')
    for key, group in sorted(groups.items()):
        if 0. not in group:
            raise ValueError(f'missing 0-degree baseline for {key}')
        base = group[0.]
        for angle, candidate in sorted(group.items()):
            if angle == 0:
                continue
            for name in ('fixture_status','fixed_time','fixed_x','fixed_y','fixed_z','requested_speed_m_s'):
                if base[name] != candidate[name]:
                    raise ValueError(f'fixture differs: {key} {angle} {name}')
            b, c = value(base,'contact_palm_speed_m_s'), value(candidate,'contact_palm_speed_m_s')
            ready = base['fixture_status'] == 'ready'
            applied = int(base['accepted']) == int(candidate['accepted']) == 1
            contacted = int(base['contacted']) == int(candidate['contacted']) == 1
            speed_gap = None if b is None or c is None else abs(c-b)
            matched = ready and applied and contacted and speed_gap is not None and speed_gap <= args.speed_tolerance
            reason = ('fixture_not_ready' if not ready else 'plan_not_applied' if not applied else
                      'no_contact' if not contacted else 'speed_unavailable' if speed_gap is None else
                      'speed_mismatch' if speed_gap > args.speed_tolerance else 'matched')
            entry = dict(seed=key[0], scenario=key[1], profile=key[2], angle_deg=angle,
                         fixture_status=base['fixture_status'], base_applied=base['accepted'],
                         candidate_applied=candidate['accepted'], base_held=base['held'],
                         candidate_held=candidate['held'], base_qualified=base['qualified'],
                         candidate_qualified=candidate['qualified'], base_speed_m_s=b,
                         candidate_speed_m_s=c, speed_difference_m_s=speed_gap,
                         matched=int(matched), exclusion_reason=reason)
            for name in metrics:
                bv, cv = value(base,name), value(candidate,name)
                entry['base_'+name], entry['candidate_'+name] = bv, cv
                entry['delta_'+name] = None if bv is None or cv is None else cv-bv
            pairs.append(entry)
    summaries = []
    for profile in sorted({r['profile'] for r in pairs}):
        for angle in sorted({r['angle_deg'] for r in pairs}):
            p = [r for r in pairs if r['profile']==profile and r['angle_deg']==angle]
            m = [r for r in p if r['matched']]
            row = dict(profile=profile, angle_deg=angle, all_pairs=len(p), matched_pairs=len(m),
                       fixture_not_ready=sum(r['exclusion_reason']=='fixture_not_ready' for r in p),
                       plan_not_applied=sum(r['exclusion_reason']=='plan_not_applied' for r in p),
                       no_contact=sum(r['exclusion_reason']=='no_contact' for r in p),
                       speed_unavailable=sum(r['exclusion_reason']=='speed_unavailable' for r in p),
                       speed_mismatch=sum(r['exclusion_reason']=='speed_mismatch' for r in p),
                       base_held=sum(int(r['base_held']) for r in m),
                       candidate_held=sum(int(r['candidate_held']) for r in m))
            for name in metrics:
                values = [r['delta_'+name] for r in m if r['delta_'+name] is not None]
                row['n_'+name] = len(values)
                row['median_delta_'+name] = median(values) if values else None
            summaries.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(args.output_dir/'pairs.csv',pairs)
    save(args.output_dir/'summary.csv',summaries)

if __name__=='__main__':
    main()
