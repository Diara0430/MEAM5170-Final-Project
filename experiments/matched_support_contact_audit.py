"""Contact-point investigation with matched recorded arm commands for first 300 ms.

The baseline run supplies its MPC deadline decisions and arm actuator setpoints;
shadow and assist replay these until the baseline contact+300 ms, then revert to
normal control. Diagnostic, single-process use only; not a real-time controller.
"""
import argparse
import csv
import gzip
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from controllers.deadline_arm_filter import DeadlineArmFilter
from experiments.box_contact_window import ContactWindow
from experiments import contact_preview,sustained_support_study
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center,throws
from experiments.scoped_calibration import ScopedCalibrationMPC
from scene_setup.scene import build_scene


def contact_relative_velocity(model,data,point,object_body,hand_body,normal):
    """计算同一世界接触点的物体相对手部法向和切向速度。"""
    jo=np.empty((3,model.nv));jh=np.empty((3,model.nv))
    mujoco.mj_jac(model,data,jo,None,point,object_body)
    mujoco.mj_jac(model,data,jh,None,point,hand_body)
    relative=(jo-jh)@data.qvel
    normal_speed=float(normal@relative)
    tangent=relative-normal*normal_speed
    return normal_speed,float(np.linalg.norm(tangent))


class PointAudit(ContactWindow):
    def __init__(self,model):
        super().__init__(model)
        self.points=[];self.frames=[];self.replay=[]
        self.act=[model.actuator(f'hand/{i}').id for i in range(16)]
        joints=model.actuator_trnid[self.act,0]
        self.qadr=model.jnt_qposadr[joints];self.vadr=model.jnt_dofadr[joints]

    def record(self,m,d,info,observer,tracker):
        super().record(m,d,info,observer,tracker)
        onset=info['first_contact_s'];t=float(d.time)
        dt=m.opt.timestep
        if round(t/dt)%round(.020/dt)==0:
            last=observer.frames[-1] if observer.frames else None
            fit=observer.estimate if onset is None else None
            plan=info.get('applied_reference')
            self.replay.append(dict(time_s=t,ball=d.joint('flight').qpos[:3].tolist(),
                hand=capture_center(d).tolist(),qpos=d.qpos.round(8).tolist(),
                forecast=None if fit is None else [fit.time,fit.position.tolist(),fit.velocity.tolist()],
                measurement_time=None if last is None else last[0],measurement=None if last is None else last[1].tolist(),
                intercept=None if plan is None else plan.point.tolist()))
        if onset is None or t-onset>.3000001:return
        ms=round(1000*(t-onset),5)
        groups=set();sum_fn=0.; speeds=[];wrench=np.empty(6)
        projectile=self.map.projectile
        for i,c in enumerate(d.contact):
            if c.efc_address<0 or projectile not in (c.geom1,c.geom2):continue
            other=c.geom1 if c.geom2==projectile else c.geom2
            group=self.map.group.get(other)
            if group is None:continue
            mujoco.mj_contactForce(m,d,i,wrench)
            fn=max(0.,float(wrench[0]))
            if fn<.02:continue
            # 同一点上分别求物体和接触手部的线速度，计算相对切向速度。
            normal=np.asarray(c.frame[:3])
            normal_speed,slip=contact_relative_velocity(m,d,c.pos,m.geom_bodyid[projectile],m.geom_bodyid[other],normal)
            ft=float(np.linalg.norm(wrench[1:3]))
            groups.add(group);sum_fn+=fn;speeds.append(slip)
            self.points.append(dict(ms=ms,group=group,hand_geom=m.geom(other).name or str(other),
                world_x=float(c.pos[0]),world_y=float(c.pos[1]),world_z=float(c.pos[2]),
                normal_force_N=fn,tangent_force_N=ft,friction_1=float(c.friction[0]),
                friction_utilization=ft/(float(c.friction[0])*fn) if c.friction[0]>0 else None,
                tangent_relative_speed_m_s=slip,normal_relative_speed_m_s=normal_speed))
        palm=d.body('hand/palm_lower');obj=d.body('projectile');R=palm.xmat.reshape(3,3)
        relpos=R.T@(obj.xpos-palm.xpos)
        row=dict(ms=ms,normal_force_sum_N=sum_fn,groups='|'.join(map(str,sorted(groups))),
                 max_contact_tangent_speed_m_s=max(speeds) if speeds else None,
                 rel_x_mm=1000*relpos[0],rel_y_mm=1000*relpos[1],rel_z_mm=1000*relpos[2],
                 floor=int(bool(info['floor'])),finger_contact=int(bool(info['finger_contact'])))
        for j in range(16):
            row[f'q{j}_rad']=float(d.qpos[self.qadr[j]])
            row[f'v{j}_rad_s']=float(d.qvel[self.vadr[j]])
            row[f'ctrl{j}_rad']=float(d.ctrl[self.act[j]])
        self.frames.append(row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--case',type=int,choices=range(8),required=True)
    p.add_argument('--profile',choices=('normal','noisy'),required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();model=build_scene('box');center=capture_center(initial(model))
    _,pos,vel,*_=next(r for r in throws(center,model.opt.gravity,a.seed) if r[0]==a.case)
    baseline_commands={};solver_schedule={};cutoff=None;allrows=[];allpoints=[];allframes=[];command_rows=[];allsolves=[]
    raw_command=DeadlineArmFilter.command;raw_run=contact_preview.run_case
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for policy in ('original','shadow','assist'):
        state={'scoring':False,'missed_schedule':0,'overrides':0,'bad_commands':0,'raw':None,'audit':None}
        class Recorder(PointAudit):
            def __init__(self,m):super().__init__(m);state['audit']=self
        def command(self,data,plan,t):
            raw_command(self,data,plan,t)
            if not state['scoring']:return
            step=round(t/self.model.opt.timestep);natural=data.ctrl[self.ids].copy();override=False
            if policy=='original':baseline_commands[step]=natural.copy()
            elif cutoff is not None and t<=cutoff+1e-9:
                if step not in baseline_commands:
                    state['missed_schedule']+=1
                else:
                    data.ctrl[self.ids]=baseline_commands[step];override=True;state['overrides']+=1
                    if not np.array_equal(data.ctrl[self.ids],baseline_commands[step]):state['bad_commands']+=1
                    q=data.qpos[self.qadr];v=data.qvel[self.dofs]
                    torque=self.kp*(data.ctrl[self.ids]-q)-self.kd*v
                    limits=self.model.actuator_forcerange[self.ids]
                    self.last_torque=np.clip(torque,limits[:,0],limits[:,1])
            if policy=='original' or override:
                command_rows.append(dict(policy=policy,step=step,time_s=t,overridden=int(override),
                    natural_diff=float(np.max(np.abs(natural-data.ctrl[self.ids]))),
                    **{f'arm_ctrl{i+1}':float(x) for i,x in enumerate(data.ctrl[self.ids])}))
        def run_case(*args,**kwargs):
            planner=kwargs.get('planner_factory')
            is_calibration=isinstance(planner,partial) and planner.func is ScopedCalibrationMPC
            if is_calibration:return raw_run(*args,**kwargs)
            state['scoring']=True
            if policy!='original':
                kwargs['solver_schedule']=solver_schedule
                kwargs['allow_schedule_extension']=True
            try:
                ret=raw_run(*args,**kwargs);state['raw']=ret
                return ret
            finally:state['scoring']=False
        with patch.object(sustained_support_study,'ContactWindow',Recorder),\
             patch.object(DeadlineArmFilter,'command',command),\
             patch.object(contact_preview,'run_case',run_case):
            row=sustained_support_study.run_one(model,a.seed,a.case,pos,vel,a.profile,policy)
        if policy=='original':
            solves=state['raw'][1]
            solver_schedule={round(r['time_s']/model.opt.timestep):bool(r['overrun']) for r in solves}
            cutoff=None if row['first_contact_s'] is None else row['first_contact_s']+.300
        audit=state['audit']
        solves=state['raw'][1]
        extension_count=sum(r.get('decision_source')=='measured_extension' for r in solves)
        mismatched_decisions=sum(round(r['time_s']/model.opt.timestep) in solver_schedule and
            bool(r['overrun'])!=solver_schedule[round(r['time_s']/model.opt.timestep)] for r in solves)
        allsolves.extend(dict(policy=policy,**r) for r in solves)
        row.update(mpc_schedule_extension_count=extension_count,mpc_schedule_decision_mismatches=mismatched_decisions,
                   matched_arm_cutoff_s=cutoff,arm_overrides=state['overrides'],
                   missing_arm_schedule_steps=state['missed_schedule'],mismatched_written_arm_commands=state['bad_commands'],
                   baseline_solver_schedule_entries=len(solver_schedule))
        sample=dict(shape='box',seed=a.seed,scenario=a.case,profile=a.profile,method='matched_arm',grasp=policy,
                    held=int(row['held']),qualified=int(row['qualified']),contact=row['first_contact_s'],
                    first_estimate={'g':model.opt.gravity.tolist()},path=audit.replay)
        with gzip.open(a.output_dir/f'{policy}_replay.json.gz','wt',encoding='utf-8') as f:json.dump(sample,f)
        allrows.append(row)
        allpoints.extend(dict(seed=a.seed,scenario=a.case,profile=a.profile,policy=policy,**r) for r in audit.points)
        allframes.extend(dict(seed=a.seed,scenario=a.case,profile=a.profile,policy=policy,**r) for r in audit.frames)
        print(policy,'held',row['held'],'qualified',row['qualified'],'first_contact',row['first_contact_s'],
              'arm overrides',row['arm_overrides'],'missing',row['missing_arm_schedule_steps'],flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    def save(path,rows):
        if not rows:return
        columns=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    save(a.output_dir/'trials.csv',allrows);save(a.output_dir/'contacts.csv',allpoints)
    save(a.output_dir/'frames.csv',allframes);save(a.output_dir/'arm_commands.csv',command_rows);save(a.output_dir/'mpc_decisions.csv',allsolves)
    with (a.output_dir/'baseline_mpc_deadlines.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['step','missed']);w.writeheader();w.writerows(dict(step=k,missed=int(v)) for k,v in solver_schedule.items())

if __name__=='__main__':main()
