"""Experimental tactile contact geometry for selecting a single opposing digit.

The extra rule uses completed-step contact point/normal and robot fingertip pose.
The inherited nominal support detector still uses MuJoCo's object center as
in the existing project; this is a simulation assumption, not a tactile sensor.
"""
import numpy as np
import mujoco
from controllers.box_contact_grasp import ContactGrasp
from controllers.sphere_cage import SphereCageClosure


class GeometrySupportGrasp(ContactGrasp):
    def __init__(self,model,hand,*,shadow=False):
        super().__init__(model,hand,'original',shadow=shadow)
        self.tip_bodies=[model.body(name).id for name in
                         ('hand/fingertip','hand/fingertip_2','hand/fingertip_3','hand/thumb_fingertip')]
        self.contact_point=self.contact_normal=None
        self.touch_groups=set();self.opposed_streak=0
        self.selected=None;self.next_select=0.
        self.active_steps=self.selection_steps=0;self.selected_mask=0

    def observe(self,model,data,velocity=None):
        # 原控制器的观察仍照常更新；以下几何规则只使用触点、法向和承载。
        super().observe(model,data,velocity=velocity)
        contacts=[];wrench=np.empty(6)
        for i,c in enumerate(data.contact):
            if self.support.projectile not in (c.geom1,c.geom2) or c.efc_address<0:continue
            on_second=c.geom2==self.support.projectile
            other=c.geom1 if on_second else c.geom2
            group=self.support.group.get(other)
            if group is None:continue
            mujoco.mj_contactForce(model,data,i,wrench)
            fn=float(wrench[0])
            if fn<.02:continue
            normal=np.array(c.frame[:3])*(1. if on_second else -1.)
            contacts.append((group,c.pos.copy(),normal,min(fn,2.)))
        self.touch_groups={x[0] for x in contacts}
        opposed=any(a[0]!=b[0] and a[2]@b[2]<-.5 for i,a in enumerate(contacts) for b in contacts[i+1:])
        self.opposed_streak=self.opposed_streak+1 if opposed else 0
        if contacts:
            weights=np.array([x[3] for x in contacts]);total=weights.sum()
            point=sum((x[1]*x[3] for x in contacts),np.zeros(3))/total
            normal=sum((x[2]*x[3] for x in contacts),np.zeros(3))/total
            norm=float(np.linalg.norm(normal))
            self.contact_point=point if norm>=.35 else None
            self.contact_normal=normal/norm if norm>=.35 else None
        else:self.contact_point=self.contact_normal=None

    def command(self,model,data,elapsed):
        SphereCageClosure.command(self,model,data,elapsed)
        active=.020<=elapsed<=.240 and self.contact_point is not None and self.opposed_streak*self.dt<.006
        if active and elapsed>=self.next_select:
            # 沿观测接触法向推测对侧位置；4.5 cm 是启发式，并非物体质心真值。
            proxy=self.contact_point+.045*self.contact_normal
            candidates=[]
            for i,body in enumerate(self.tip_bodies):
                if i in self.touch_groups:continue
                tip=data.xpos[body]
                distance=float(np.linalg.norm(tip-proxy))
                if distance<=.14:candidates.append((distance,i))
            self.selected=min(candidates)[1] if candidates else None
            self.next_select=elapsed+.040
            self.selection_steps+=1
        if not active or self.selected in self.touch_groups:self.selected=None
        desired=np.zeros(4)
        if active and self.selected is not None:
            # 保持所有已经承载手指的原指令，只辅助一根未承载手指。
            desired[self.selected]=.060
            self.active_steps+=1;self.selected_mask |= 1<<self.selected
        self._move(desired,.6)
        ids=self.ids_flex;original=data.ctrl[ids].copy()
        delta=np.clip(self.bias[:,None]*self.weights,-.20/self.stiffness,.20/self.stiffness)
        # 若手指刚开始承载，立即撤去该指的额外推进；原闭指命令保持不变。
        for group in self.touch_groups:
            if group>=0:delta[group,:]=0.
        command=np.clip(original+delta,self.hand.limits[self.flex,0],self.hand.limits[self.flex,1])
        change=float(np.max(np.abs((command-original)*self.stiffness)))
        self.max_torque_change=max(self.max_torque_change,change)
        if not self.shadow:
            data.ctrl[ids]=command
            self.delivered_steps+=int(change>1e-7)

    def diagnostics(self):
        return dict(geometry_active_steps=self.active_steps,geometry_selection_steps=self.selection_steps,
                    geometry_selected_mask=self.selected_mask,geometry_delivered_steps=self.delivered_steps,
                    geometry_max_torque_delta_Nm=self.max_torque_change)
