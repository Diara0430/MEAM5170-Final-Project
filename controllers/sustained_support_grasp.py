"""Experimental bounded tactile support assist for cuboid capture.

Uses only completed-step hand contact, never the true object state or outcome.
"""
import numpy as np
from controllers.box_contact_grasp import ContactGrasp
from controllers.sphere_cage import SphereCageClosure


class SustainedSupportGrasp(ContactGrasp):
    def __init__(self,model,hand,*,shadow=False):
        super().__init__(model,hand,'original',shadow=shadow)
        self.last_load_s=None
        self.active_steps=self.recovery_steps=self.stable_steps=0
        self.first_stable_s=None

    def command(self,model,data,elapsed):
        # 保留原闭指轨迹，按上一个完成的物理步判断承载及两侧支撑。
        SphereCageClosure.command(self,model,data,elapsed)
        loaded=self.support.loaded
        if loaded:self.last_load_s=elapsed
        if self.support.holding:
            self.stable_steps+=1
            if self.first_stable_s is None:self.first_stable_s=elapsed
        recent=(self.last_load_s is not None and elapsed-self.last_load_s<=.030)
        active=.020<=elapsed<=.240 and not self.support.holding and (bool(loaded) or recent)
        if active:
            # 已承载侧小幅让位；尚未承载的指缓慢向内，失去接触时限时补抓。
            desired=np.array([-.025 if i in loaded else .065 for i in range(4)])
            self.active_steps+=1
            self.recovery_steps+=int(not loaded)
        else:
            desired=np.zeros(4)
        self._move(desired,1.0)
        ids=self.ids_flex;original=data.ctrl[ids].copy()
        # 控制的是额外位置指令对应的执行器力矩变化，不是接触力上限。
        delta=np.clip(self.bias[:,None]*self.weights,-.25/self.stiffness,.25/self.stiffness)
        command=np.clip(original+delta,self.hand.limits[self.flex,0],self.hand.limits[self.flex,1])
        change=float(np.max(np.abs((command-original)*self.stiffness)))
        self.max_torque_change=max(self.max_torque_change,change)
        if not self.shadow:
            data.ctrl[ids]=command
            self.delivered_steps+=int(change>1e-7)

    def diagnostics(self):
        return dict(support_assist_active_steps=self.active_steps,support_assist_recovery_steps=self.recovery_steps,
                    support_assist_stable_steps=self.stable_steps, support_assist_first_stable_s=self.first_stable_s,
                    support_assist_delivered_steps=self.delivered_steps,
                    support_assist_max_torque_delta_Nm=self.max_torque_change)
