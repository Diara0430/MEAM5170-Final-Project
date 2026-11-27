"""Replay archived fixed-state QPs with independent fresh and scoped solvers."""
import argparse
import csv
from pathlib import Path
import numpy as np
import osqp
from scipy.sparse import csc_matrix
from experiments.scoped_calibration import CalibrationWorkspace
from experiments.solver_snapshot_audit import signature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for case_dir in sorted(args.input_dir.iterdir()):
        candidate_path=case_dir/'candidates.csv'
        if not candidate_path.exists(): continue
        workspace=CalibrationWorkspace()
        with candidate_path.open(encoding='utf-8-sig',newline='') as f:
            cases=list(csv.DictReader(f))
        for i,case in enumerate(cases):
            with np.load(case_dir/f'qp_{i:02d}.npz') as z:
                P=csc_matrix((z['P_data'],z['P_indices'],z['P_indptr']),shape=tuple(z['P_shape']))
                A=csc_matrix((z['A_data'],z['A_indices'],z['A_indptr']),shape=tuple(z['A_shape']))
                if signature(P,z['q'],A,z['l'],z['u'])!=case['fresh_qp_hash']:
                    raise ValueError(f'Archived QP hash mismatch: {case_dir.name} {i}')
                opts=dict(verbose=False,polishing=False,max_iter=2000,eps_abs=1e-5,eps_rel=1e-5)
                fresh=osqp.OSQP();fresh.setup(P=P,q=z['q'],A=A,l=z['l'],u=z['u'],**opts)
                a=fresh.solve(raise_error=False)
                b=workspace.setup(P,z['q'],A,z['l'],z['u'],**opts).solve(raise_error=False)
                diff=float(np.max(np.abs(a.x-b.x))) if a.info.status_val==b.info.status_val==1 else None
                rows.append(dict(case=case_dir.name,index=i,speed=case['speed_m_s'],
                                 fresh_status=a.info.status,scoped_status=b.info.status,
                                 same_status=int(a.info.status_val==b.info.status_val),solution_max_difference=diff,
                                 previous_fresh_accepted=case['fresh_accepted'],previous_reset_accepted=case['reset_rho_accepted'],
                                 cache_hits=workspace.hits,cache_misses=workspace.misses))
    if not rows: raise ValueError('No archived QP snapshots found')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'{len(rows)} QPs: {sum(r["same_status"] for r in rows)} equal statuses',flush=True)

if __name__=='__main__': main()
