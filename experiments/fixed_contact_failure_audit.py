"""Replay selected fixed-intercept cuboid trials with contact and hand diagnostics."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from experiments.box_contact_window import ContactWindow
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.scoped_calibration import ScopedCalibrationMPC
from experiments import fixed_angle_study
from scene_setup.scene import build_scene


class DetailedContact:
    def __init__(self,model):
        self.window=ContactWindow(model)
        self.rows=[];self.replay=[];self.pre_velocity=None;self.incoming=None
        self.act=[model.actuator(f'hand/{i}').id for i in range(16)]
        joints=model.actuator_trnid[self.act,0]
        self.qadr=model.jnt_qposadr[joints];self.vadr=model.jnt_dofadr[joints]

    def record(self,m,d,info,observer,tracker):
        t=float(d.time);onset=info['first_contact_s'];dt=m.opt.timestep
        velocity=d.joint('flight').qvel[:3].copy()
        if onset is None:
            self.pre_velocity=velocity
        elif self.incoming is None and self.pre_velocity is not None:
            speed=np.linalg.norm(self.pre_velocity)
            self.incoming=self.pre_velocity/speed if speed>1e-9 else np.zeros(3)
        self.window.record(m,d,info,observer,tracker)
        # 每 20 ms 保存整个模型的位姿，可离线查看仿真回放。
        if round(t/dt)%round(.020/dt)==0:
            last=observer.frames[-1] if observer.frames else None
            fit=observer.estimate if onset is None else None
            plan=info.get('applied_reference')
            self.replay.append(dict(time_s=t,ball=d.joint('flight').qpos[:3].tolist(),
                hand=capture_center(d).tolist(),qpos=d.qpos.round(8).tolist(),
                forecast=None if fit is None else [fit.time,fit.position.tolist(),fit.velocity.tolist()],
                measurement_time=None if last is None else last[0],measurement=None if last is None else last[1].tolist(),
                intercept=None if plan is None else plan.point.tolist()))
        if onset is None or t-onset>.3000001:
            return
        # 法向/切向力是模拟接触求解器输出；摩擦利用率是逐接触诊断而非真实测量。
        normal=tangent=0.;ratios=[];wrench=np.empty(6);parts=set()
        for i,c in enumerate(d.contact):
            if c.efc_address<0 or self.window.map.projectile not in (c.geom1,c.geom2): continue
            other=c.geom1 if c.geom2==self.window.map.projectile else c.geom2
            group=self.window.map.group.get(other)
            if group is None:continue
            mujoco.mj_contactForce(m,d,i,wrench)
            fn=max(0.,float(wrench[0]));ft=float(np.linalg.norm(wrench[1:3]))
            normal+=fn;tangent+=ft;parts.add(str(group))
            if fn>=.02 and c.friction[0]>0:ratios.append(ft/(c.friction[0]*fn))
        obj=d.body('projectile');palm=d.body('hand/palm_lower');rot=palm.xmat.reshape(3,3)
        rel=rot.T@(obj.xpos-palm.xpos)
        row=dict(ms=round(1000*(t-onset),5),rel_x_mm=1000*rel[0],rel_y_mm=1000*rel[1],rel_z_mm=1000*rel[2],
                 normal_sum_N=normal,tangent_norm_sum_N=tangent,max_friction_utilization=max(ratios) if ratios else None,
                 contact_groups='|'.join(sorted(parts)),flight_speed_m_s=float(np.linalg.norm(velocity)),
                 incoming_projection_m_s=None if self.incoming is None else float(velocity@self.incoming),
                 floor=int(bool(info['floor'])),finger_contact=int(bool(info['finger_contact'])))
        for i in range(16):
            row[f'finger_q{i}_rad']=float(d.qpos[self.qadr[i]])
            row[f'finger_v{i}_rad_s']=float(d.qvel[self.vadr[i]])
            row[f'finger_target{i}_rad']=float(d.ctrl[self.act[i]])
        self.rows.append(row)


def read_selected(paths):
    out=[];seen=set()
    for path in paths:
        with path.open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if r['method']!='reuse' or r['accepted']!='1':continue
                key=(int(r['seed']),int(r['scenario']),r['profile'])
                if key in seen: raise ValueError(f'duplicate source trial {key}')
                seen.add(key);out.append((key,r))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs='+',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args();selected=read_selected(a.sources)
    if not selected: raise ValueError('No applied scoped trials')
    model=build_scene('box');center=capture_center(initial(model));rows=[];frames=[]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for (seed,case,profile),old in selected:
        _,p,v,*_=next(x for x in throws(center,model.opt.gravity,seed) if x[0]==case)
        audit=DetailedContact(model); original_trial=fixed_angle_study.trial
        def instrumented(*args,**kwargs):
            rec=kwargs.get('record_frame')
            if rec is not None:
                def combined(m,d,info,observer,tracker):
                    rec(m,d,info,observer,tracker);audit.record(m,d,info,observer,tracker)
                kwargs['record_frame']=combined
            return original_trial(*args,**kwargs)
        with patch.object(fixed_angle_study,'trial',instrumented), patch.object(fixed_angle_study,'CalibrationMPC',ScopedCalibrationMPC):
            result=fixed_angle_study.run(model,seed,case,p,v,profile,[0.])[0]
        result.update(**audit.window.summary(),source_held=old['held'],source_qualified=old['qualified'],
                      source_accepted=old['accepted'],source_condition='four_cpu_workers',
                      rerun_condition='no_added_workers_with_diagnostics',instrumented_frames=len(audit.rows))
        rows.append(result)
        frames.extend(dict(seed=seed,scenario=case,profile=profile,**f) for f in audit.rows)
        name=f'{seed}_{case}_{profile}'
        sample=dict(shape='box',seed=seed,scenario=case,profile=profile,method='fixed_pose',grasp='original',
                    held=int(result['held']),qualified=int(result['qualified']),contact=result['first_contact_s'],
                    first_estimate={'g':model.opt.gravity.tolist()},path=audit.replay)
        with gzip.open(a.output_dir/f'{name}_replay.json.gz','wt',encoding='utf-8') as f:json.dump(sample,f)
        print(name,'previous held',old['held'],'rerun',result['held'],'applied',result['accepted'],flush=True)
        cols=dict.fromkeys(k for r in rows for k in r)
        save_csv(a.output_dir/'trials.csv',[{**cols,**r} for r in rows])
        if frames:save_csv(a.output_dir/'frames.csv',frames)

if __name__=='__main__':main()
