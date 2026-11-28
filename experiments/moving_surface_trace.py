"""Record moving-box surface and finger-Jacobian inputs for offline diagnosis.

The exact box pose and geometry are read only by the recorder. They never enter
GeometrySupportGrasp or its commands. Benchmark timing can change due to tracing.
"""
import argparse
import csv
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from experiments import geometry_selection_trace as trace
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center,throws
from scene_setup.scene import build_scene


class MotionTracedGrasp(trace.TracedGeometryGrasp):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.episode_kinematics=None

    def command(self,model,data,elapsed):
        old=self.episode
        super().command(model,data,elapsed)
        s=self.snapshot;digit=s['selected']
        if digit is not None and self.episode!=old:
            # 局部模型在选中时只记录雅可比和关节限制，不参与闭指决策。
            actuator_ids=np.asarray(self.hand.ids)[4*digit:4*digit+4]
            joints=model.actuator_trnid[actuator_ids,0]
            dofs=model.jnt_dofadr[joints];qadrs=model.jnt_qposadr[joints]
            J=np.empty((3,model.nv));mujoco.mj_jacBody(model,data,J,None,self.tip_bodies[digit])
            ranges=model.jnt_range[joints].copy()
            limited=model.jnt_limited[joints].astype(bool)
            ranges[~limited,0]=-np.inf;ranges[~limited,1]=np.inf
            self.episode_kinematics=dict(tip=data.xpos[self.tip_bodies[digit]].copy(),
                jac=J[:,dofs].copy(),q=data.qpos[qadrs].copy(),v=data.qvel[dofs].copy(),ranges=ranges,
                diagnostic_object_pose=data.joint('flight').qpos.copy())
        if digit is None:self.episode_kinematics=None


class MotionTraceWindow(trace.TraceWindow):
    def record(self,m,d,info,observer,tracker):
        count=len(self.output)
        super().record(m,d,info,observer,tracker)
        if len(self.output)==count:return
        row=self.output[-1];h=self.hand_ref();s=h.snapshot
        # 保存自由关节的完成步真实姿态，用于事后比较移动的长方体表面。
        q=d.joint('flight').qpos.copy()
        row.update(obj_x=q[0],obj_y=q[1],obj_z=q[2],obj_qw=q[3],obj_qx=q[4],obj_qy=q[5],obj_qz=q[6])
        if s['selected'] is not None:
            proxy=s['fixed_proxy'];row.update(proxy_x=proxy[0],proxy_y=proxy[1],proxy_z=proxy[2])
            kin=h.episode_kinematics
            if kin is not None:
                for j in range(7):row[f'episode_obj{j}']=kin['diagnostic_object_pose'][j]
                for i in range(3):
                    row[f'tip{i}']=kin['tip'][i]
                    for j in range(4):row[f'jac{i}{j}']=kin['jac'][i,j]
                for j in range(4):
                    row[f'q{j}']=kin['q'][j];row[f'v{j}']=kin['v'][j]
                    row[f'lo{j}']=kin['ranges'][j,0];row[f'hi{j}']=kin['ranges'][j,1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--profile',choices=('normal','noisy'),required=True)
    p.add_argument('--cases',type=int,default=8);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();m=build_scene('box');center=capture_center(initial(m));rows=[];frames=[]
    # 解析盒面使用 qpos，要求碰撞几何在刚体本地无额外变换。
    gid=m.geom('projectile').id
    if (m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_BOX or
        not np.allclose(m.geom_size[gid],[.032,.021,.017]) or
        not np.allclose(m.geom_pos[gid],0) or not np.allclose(m.geom_quat[gid],[1,0,0,0])):
        raise ValueError('Recorder requires the documented aligned cuboid geometry')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for case,pos,vel,*_ in throws(center,m.opt.gravity,a.seed):
        if case>=a.cases:continue
        policies=['shadow','geometry']
        if (a.seed+case)%2:policies.reverse()
        for policy in policies:
            with patch.object(trace,'TracedGeometryGrasp',MotionTracedGrasp),patch.object(trace,'TraceWindow',MotionTraceWindow):
                row,samples=trace.run_one(m,a.seed,case,pos,vel,a.profile,policy)
            rows.append(row);frames.extend(dict(seed=a.seed,scenario=case,profile=a.profile,policy=policy,**f) for f in samples)
            print(a.seed,a.profile,case,policy,'applied',row['accepted'],'held',row['held'],
                  'selected',row.get('geometry_active_steps',0),flush=True)
        trace.save(a.output_dir/'trials.csv',rows);trace.save(a.output_dir/'frames.csv',frames)

if __name__=='__main__':main()
