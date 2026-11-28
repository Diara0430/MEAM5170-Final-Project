"""Diagnose why a geometry-selected finger does or does not contact a cuboid.

Only diagnostic logs use exact MuJoCo object geometry for distances. They are
not inputs to the experimental hand controller.
"""
import argparse
import csv
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
from controllers.geometry_support_grasp import GeometrySupportGrasp
from experiments import geometry_support_study
from experiments.box_contact_window import ContactWindow
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center,throws
from scene_setup.scene import build_scene


class TracedGeometryGrasp(GeometrySupportGrasp):
    def __init__(self,model,hand,*,shadow=False):
        super().__init__(model,hand,shadow=shadow)
        self.snapshot=None;self.episode=-1;self.previous_selected=None;self.fixed_proxy=None
        self.geom_groups={i:[g for g,group in self.support.group.items()
                             if group==i and model.geom_contype[g]] for i in range(4)}

    def command(self,model,data,elapsed):
        # 保存本次命令决策使用的上一完成步信息，避免事后新触点被误认为决策输入。
        groups=set(self.touch_groups);point=None if self.contact_point is None else self.contact_point.copy()
        normal=None if self.contact_normal is None else self.contact_normal.copy()
        opposed=self.opposed_streak*self.dt>=.006
        eligible=.020<=elapsed<=.240 and point is not None and not opposed
        super().command(model,data,elapsed)
        selected=self.selected if eligible else None
        start_distance=None
        if selected is not None:
            if selected!=self.previous_selected:
                self.episode+=1
                self.fixed_proxy=point+.045*normal
            tip=data.xpos[self.tip_bodies[selected]]
            start_distance=float(np.linalg.norm(tip-self.fixed_proxy))
        else:self.fixed_proxy=None
        self.previous_selected=selected
        if not (.020<=elapsed<=.240):reason='outside_window'
        elif not groups:reason='no_loaded_contact'
        elif point is None:reason='ambiguous_normal'
        elif opposed:reason='opposing_normals'
        elif selected is None:reason='no_digit_selected'
        else:reason='selected'
        self.snapshot=dict(command_time_s=float(data.time),elapsed_s=float(elapsed),reason=reason,
            pre_groups='|'.join(map(str,sorted(groups))),selected=selected,episode=self.episode if selected is not None else None,
            fixed_proxy=None if self.fixed_proxy is None else self.fixed_proxy.copy(),
            pre_proxy_distance_m=start_distance,opposed_streak_steps=int(self.opposed_streak))


class TraceWindow(ContactWindow):
    def __init__(self,model,hand_ref,output):
        super().__init__(model);self.hand_ref=hand_ref;self.output=output

    def record(self,m,d,info,observer,tracker):
        super().record(m,d,info,observer,tracker)
        h=self.hand_ref()
        if h is None or h.snapshot is None or info['first_contact_s'] is None:return
        ms=1000*(float(d.time)-info['first_contact_s'])
        if ms>300.000001:return
        s=h.snapshot;selected=s['selected'];gap=None;tip_proxy=None;touch=0
        if selected is not None:
            tip_proxy=float(np.linalg.norm(d.xpos[h.tip_bodies[selected]]-s['fixed_proxy']))
            touch=int(selected in h.touch_groups)
            distances=[]
            for geom in h.geom_groups[selected]:
                # 仅用于离线诊断：控制器本身不读取物体真实几何距离。
                segment=np.empty(6)
                distances.append(float(mujoco.mj_geomDistance(m,d,geom,h.support.projectile,.30,segment)))
            gap=min(distances) if distances else None
        self.output.append(dict(ms=round(ms,5),command_time_s=s['command_time_s'],elapsed_s=s['elapsed_s'],
            reason=s['reason'],pre_groups=s['pre_groups'],post_groups='|'.join(map(str,sorted(h.touch_groups))),
            selected=selected,episode=s['episode'],opposed_streak_steps=s['opposed_streak_steps'],
            pre_proxy_distance_m=s['pre_proxy_distance_m'],post_proxy_distance_m=tip_proxy,
            exact_digit_object_gap_m=gap,selected_loaded_after_step=touch,
            delivered_steps=h.delivered_steps,selected_mask=h.selected_mask))


def save(path,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def run_one(model,seed,case,p,v,profile,policy):
    hand=[None];frames=[]
    class Hand(TracedGeometryGrasp):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);hand[0]=self
    def window(model):return TraceWindow(model,lambda:hand[0],frames)
    with patch.object(geometry_support_study,'GeometrySupportGrasp',Hand),patch.object(geometry_support_study,'ContactWindow',window):
        row=geometry_support_study.run_one(model,seed,case,p,v,profile,policy)
    return row,frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--profile',choices=('normal','noisy'),required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cases',type=int,default=8)
    a=p.parse_args();m=build_scene('box');center=capture_center(initial(m));rows=[];frames=[]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    policies=['original','shadow','geometry']
    for case,pos,vel,*_ in throws(center,m.opt.gravity,a.seed):
        if case>=a.cases:continue
        k=(a.seed+case)%len(policies)
        for policy in policies[k:]+policies[:k]:
            row,samples=run_one(m,a.seed,case,pos,vel,a.profile,policy)
            rows.append(row);frames.extend(dict(seed=a.seed,scenario=case,profile=a.profile,policy=policy,**x) for x in samples)
            print(a.seed,a.profile,case,policy,'applied',row['accepted'],'held',row['held'],
                  'selected steps',row.get('geometry_active_steps',0),flush=True)
        save(a.output_dir/'trials.csv',rows);save(a.output_dir/'frames.csv',frames)

if __name__=='__main__':main()
