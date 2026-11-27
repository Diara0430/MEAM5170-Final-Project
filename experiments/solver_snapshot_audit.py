"""Compare fresh and reused OSQP solves on exactly the same calibration state.

Diagnostic instrumentation runs both solvers sequentially for each speed; elapsed
simulation deadlines are intentionally not valid for this offline investigation.
"""
import argparse
import csv
import hashlib
import linecache
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch
import numpy as np
import osqp
from controllers.catching_mpc import CatchingMPC
from controllers.fixed_intercept import CalibrationMPC, FixedIntercept, PreparationFinished
from experiments.box_fixed_speeds import arm
from experiments.contact_preview import trial
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.shape_catch import BoxObserver, box_launch
from scene_setup.scene import build_scene


def signature(P,q,A,l,u):
    h=hashlib.sha256()
    for x in (P.indptr,P.indices,P.data,q,A.indptr,A.indices,A.data,l,u):
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def state_signature(data):
    h=hashlib.sha256()
    for x in (data.qpos,data.qvel,data.ctrl,data.qacc_warmstart):
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def investigate(model,seed,case,p,v,profile,output):
    sensor=partial(BoxObserver,seed=seed,case=case,profile=profile,long_track=True,use_pose=True)
    fixture=FixedIntercept(); rows=[]; caches={"reuse":{},"reset_rho":{}}; snapshots=[]
    real_solver=osqp.OSQP
    real_solve=CatchingMPC.solve
    def compare(self,data,*args,**kwargs):
        if not isinstance(self,CalibrationMPC) or kwargs.get('speed_request') is None:
            return real_solve(self,data,*args,**kwargs)
        requested=float(kwargs['speed_request'][1]); initial_state=state_signature(data)
        original_last=None if self.last_solution is None else self.last_solution.copy()
        traces={}
        for mode in ('fresh','reuse','reset_rho'):
            out={}; traces[mode]=out
            cache=caches.get(mode,{})
            class Solver:
                def setup(self,*,P,q,A,l,u,**options):
                    out['qp_hash']=signature(P,q,A,l,u)
                    key=(P.shape,P.indptr.tobytes(),P.indices.tobytes(),P.data.tobytes(),q.tobytes(),
                         A.shape,A.indptr.tobytes(),A.indices.tobytes(),A.data.tobytes(),tuple(sorted(options.items())))
                    out['hit']=int(mode!='fresh' and cache.get('key')==key)
                    if out['hit']:
                        self.solver=cache['solver']; self.solver.update(l=l,u=u)
                        if mode=='reset_rho': self.solver.update_settings(rho=0.1)
                        self.solver.warm_start(x=np.zeros(P.shape[0]),y=np.zeros(A.shape[0]))
                    else:
                        self.solver=real_solver();self.solver.setup(P=P,q=q,A=A,l=l,u=u,**options)
                        if mode!='fresh':cache.update(key=key,solver=self.solver)
                    if mode=='fresh':
                        # Store the exact sparse QP inputs for an independent replay.
                        snapshots.append(dict(speed=requested,P=P.copy(),q=q.copy(),A=A.copy(),l=l.copy(),u=u.copy(),options=options.copy(),
                                              qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),
                                              qacc_warmstart=data.qacc_warmstart.copy(),p=np.array(args[0]),v=np.array(args[1]),
                                              now=float(args[2]),previous_acc=np.array(args[3].reference(args[2])[2]),
                                              fixed_point=np.array(kwargs['fixed_target'][0]),incoming=np.array(kwargs['fixed_target'][1]),
                                              projection=np.array(kwargs['speed_request'][0])))
                def warm_start(self,*a,**k): return self.solver.warm_start(*a,**k)
                def solve(self,*a,**k):
                    result=self.solver.solve(*a,**k)
                    out['result']=result
                    return result
            self.last_solution=None if original_last is None else original_last.copy()
            events=[]
            def trace(frame,event,arg):
                if frame.f_code is real_solve.__code__ and event=='line':
                    line=linecache.getline(frame.f_code.co_filename,frame.f_lineno).strip()
                    if line=='continue' or 'valid=False;break' in line:
                        context=linecache.getline(frame.f_code.co_filename,frame.f_lineno-1).strip()
                        event_row={'line':frame.f_lineno,'condition':context}
                        if 'achievable' in frame.f_locals and 'requested' in frame.f_locals:
                            event_row['terminal_speed_error']=abs(frame.f_locals['achievable']-frame.f_locals['requested'])
                        if 'actual' in frame.f_locals and 'point' in frame.f_locals:
                            event_row['terminal_position_error']=float(np.linalg.norm(frame.f_locals['actual']-frame.f_locals['point']))
                        events.append(event_row)
                return trace
            previous_trace=sys.gettrace()
            try:
                with patch('controllers.catching_mpc.osqp.OSQP',Solver):
                    sys.settrace(trace)
                    plan,info=real_solve(self,data,*args,**kwargs)
            finally:
                sys.settrace(previous_trace)
            out['events']=events
            out['plan']=plan
            out['info']=info
            out['state_after']=state_signature(data)
            if mode=='fresh':
                fresh_plan=plan; fresh_info=info
                fresh_last=None if self.last_solution is None else self.last_solution.copy()
        if any(t['state_after']!=initial_state for t in traces.values()):
            raise AssertionError('MuJoCo input state changed between paired solves')
        self.last_solution=fresh_last
        f,r=traces['fresh'],traces['reuse']
        row=dict(seed=seed,scenario=case,profile=profile,speed_m_s=requested,state_hash=initial_state,
                 fresh_qp_hash=f.get('qp_hash',''),reuse_qp_hash=r.get('qp_hash',''),
                 identical_qp=int(f.get('qp_hash')==r.get('qp_hash') and bool(f.get('qp_hash'))),
                 reuse_hit=r.get('hit',0),fresh_accepted=int(f['plan'] is not None),reuse_accepted=int(r['plan'] is not None),
                 reset_rho_accepted=int(traces['reset_rho']['plan'] is not None),
                 reset_rho_qp_hash=traces['reset_rho'].get('qp_hash',''))
        for name,t in traces.items():
            res=t.get('result'); inf=None if res is None else res.info
            row.update({name+'_status':None if inf is None else inf.status,
                        name+'_status_val':None if inf is None else inf.status_val,
                        name+'_iterations':None if inf is None else inf.iter,
                        name+'_primal_residual':None if inf is None else inf.prim_res,
                        name+'_dual_residual':None if inf is None else inf.dual_res,
                        name+'_rho_updates':None if inf is None else inf.rho_updates,
                        name+'_rho_estimate':None if inf is None else inf.rho_estimate,
                        name+'_objective':None if inf is None else inf.obj_val,
                        name+'_solved_then_rejected':int(inf is not None and inf.status_val==1 and t['plan'] is None),
                        name+'_rejection_trace':' | '.join(str(x) for x in t['events'])})
        if f.get('result') is not None and r.get('result') is not None and f['result'].info.status_val==1 and r['result'].info.status_val==1:
            row['solution_max_difference']=float(np.max(np.abs(f['result'].x-r['result'].x)))
        z=traces['reset_rho'].get('result')
        if f.get('result') is not None and z is not None and f['result'].info.status_val==1 and z.info.status_val==1:
            row['reset_rho_solution_max_difference']=float(np.max(np.abs(f['result'].x-z.x)))
        rows.append(row)
        print(f'{seed} {case} {profile} speed={requested:.4f} fresh={row["fresh_accepted"]} reuse={row["reuse_accepted"]} statuses={row["fresh_status"]}/{row["reuse_status"]}',flush=True)
        return fresh_plan,fresh_info
    with patch.object(CatchingMPC,'solve',compare):
        try:
            trial(model,seed,case,p,v,True,planner_factory=partial(CalibrationMPC,fixture=fixture,planning_speed_limit=.97),
                  tracker_factory=arm,observer_factory=sensor,launch_rotation=box_launch(seed,case))
        except PreparationFinished:
            pass
    output.mkdir(parents=True,exist_ok=True)
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (output/'candidates.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        for i,s in enumerate(snapshots):
            P,A=s['P'],s['A']
            np.savez_compressed(output/f'qp_{i:02d}.npz',speed=s['speed'],P_data=P.data,P_indices=P.indices,P_indptr=P.indptr,P_shape=P.shape,
                                A_data=A.data,A_indices=A.indices,A_indptr=A.indptr,A_shape=A.shape,q=s['q'],l=s['l'],u=s['u'],
                                qpos=s['qpos'],qvel=s['qvel'],ctrl=s['ctrl'],qacc_warmstart=s['qacc_warmstart'],
                                p=s['p'],v=s['v'],now=s['now'],previous_acc=s['previous_acc'],fixed_point=s['fixed_point'],
                                incoming=s['incoming'],projection=s['projection'])
    return len(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--case',type=int,required=True)
    p.add_argument('--profile',choices=('normal','noisy'),required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();m=build_scene('box');center=capture_center(initial(m))
    for case,pos,vel,*_ in throws(center,m.opt.gravity,a.seed):
        if case==a.case:
            n=investigate(m,a.seed,case,pos,vel,a.profile,a.output_dir)
            if not n: print('No eligible anchor or candidate speed was found.',flush=True)
            return
    raise ValueError('Unknown case')

if __name__=='__main__':main()
