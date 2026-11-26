"""延迟交付预测中推进当前手指控制器、承载反馈和首触事件；无机器学习。

复制小型控制状态并复用独立MjData，避免每个物理步深拷贝大型日志/探针。
只使用当前手指控制状态与已排队指令，不访问真实未来轨迹。
未知的新拇指IK重规划不在短时预测内执行；相关误差仍记录。
"""
import copy
import hashlib

import mujoco
import numpy as np

from controllers.catching_mpc import braking_reference
from controllers.discrete_arm_filter import ArmFilterConfig
from controllers.scheduled_arm_tracker import ScheduledArmTracker
from controllers.sphere_cage import SphereCageClosure


def fingerprint(control):
    """验证预测没有改变真实控制器的数组、反馈计时、日志长度或探针状态。"""
    digest=hashlib.sha256()
    for obj in (control,control.support):
        for key,value in sorted(vars(obj).items()):
            if isinstance(value,np.ndarray):digest.update(key.encode());digest.update(value.tobytes())
            elif isinstance(value,(float,int,str,type(None),bool,set)):digest.update(repr((key,value)).encode())
    digest.update(str(len(control.rows)).encode());digest.update(control.probe.qpos.tobytes());digest.update(control.probe.qvel.tobytes())
    return digest.hexdigest()


class CoupledHandForecast(ScheduledArmTracker):
    def __init__(self,model,delay_steps=1,hand_feedback=True,acceleration=3.8):
        if delay_steps<1:raise ValueError("Coupled forecast requires at least one delayed step")
        super().__init__(model,delay_steps,True,ArmFilterConfig(acceleration=acceleration))
        self.hand_feedback=hand_feedback;self.context=None;self.clone=None
        self.hand_probe=mujoco.MjData(model);self.verify=mujoco.MjData(model)
        self.hand_ids=[model.actuator(f"hand/{i}").id for i in range(16)]
        joints=[model.joint(f"hand/{i}") for i in range(16)]
        self.hand_adr=[j.qposadr[0] for j in joints];self.hand_dof=[j.dofadr[0] for j in joints]
        self.projectile=model.geom("projectile").id
        self.hand_geoms={g for g,b in enumerate(model.geom_bodyid) if (model.body(b).name or "").startswith("hand/")}

    def bind_hand_context(self,control,nominal,close,first,accepted,braking,pending):
        if not isinstance(control,SphereCageClosure):raise TypeError("Coupled forecast requires SphereCageClosure")
        self.context=dict(control=control,nominal=nominal,close=close,first=first,accepted=accepted,braking=braking,pending=pending)

    def begin_forecast(self,data,plan,t):
        if self.context is None:raise RuntimeError("Hand forecast context was not bound")
        context=self.context;control=context["control"]
        self.before=fingerprint(control);self.real_hand_ctrl=data.ctrl[self.hand_ids].copy()
        self.first=context["first"];self.close=context["close"];self.predicted_brake=None;self.predicted_onset=False
        if self.hand_feedback:
            # 每个可变数组独立复制。静态关节索引/参数只读共享，历史日志不复制。
            self.clone=copy.copy(control)
            for key,value in vars(control).items():
                if isinstance(value,np.ndarray):setattr(self.clone,key,value.copy())
            self.clone.rows=[];self.clone.probe=self.hand_probe
            self.clone.support=copy.copy(control.support)
            self.clone.support.loaded=set(control.support.loaded);self.clone.support.delay=control.support.delay.copy()
            # 当前还没算出的新IK计划不可作为已知未来信息。
            self.clone.last_plan=float("inf")

    def forecast_command(self,data,offset,t):
        if not self.hand_feedback:return
        if offset:
            if self.predicted_onset and not self.context["braking"] and self.predicted_brake is None:
                self.predicted_brake=braking_reference(self.model,data,t)
            if self.close is None and self.first is not None and self.context["accepted"]:self.close=t
            if self.close is None:self.clone.prepare(self.model,data,t,self.context["nominal"])
            else:self.clone.command(self.model,data,t-self.close)

    def forecast_observe(self,data,offset,t,velocity):
        if not self.hand_feedback:return
        if self.first is None:
            touching=any(c.dist<=0 and self.projectile in (c.geom1,c.geom2) and
                (c.geom1 if c.geom2==self.projectile else c.geom2) in self.hand_geoms for c in data.contact)
            if touching:self.first=t;self.predicted_onset=True
        self.clone.observe(self.model,data,velocity=velocity)
        # 与真实循环相同：步末更新运动学后才计算下一步手指命令。
        mujoco.mj_kinematics(self.model,data);mujoco.mj_comPos(self.model,data)

    def delivery_plan(self,plan,data,t):
        if self.hand_feedback:
            if self.predicted_brake is not None:return self.predicted_brake
            pending=self.context["pending"]
            if pending is not None and pending.observation_time<=t and not self.context["braking"]:plan=pending
        return super().delivery_plan(plan,data,t)

    def contact_signature(self,data):
        loaded=[]
        for i,c in enumerate(data.contact):
            if self.projectile not in (c.geom1,c.geom2) or c.efc_address<0:continue
            other=c.geom1 if c.geom2==self.projectile else c.geom2
            if other not in self.hand_geoms:continue
            wrench=np.empty(6);mujoco.mj_contactForce(self.model,data,i,wrench)
            if wrench[0]>=.02:loaded.append(other)
        return tuple(sorted(set(loaded)))

    def forecast_extra(self,data,t):
        assert fingerprint(self.context["control"])==self.before,"Prediction mutated the live hand controller"
        # 额外复核选中的最终命令，用相同输入记录生效步的预测接触模式。
        mujoco.mj_copyData(self.verify,self.model,data);mujoco.mj_step(self.model,self.verify)
        verified=(self.verify.qvel[self.dofs]-data.qvel[self.dofs])/self.model.opt.timestep
        expected=np.array([self.inner.rows[-1][f"predicted_accel_{i+1}"] for i in range(7)])
        assert np.allclose(verified,expected,atol=1e-7,rtol=0),"Final candidate prediction inconsistent"
        return dict(hand_ctrl=data.ctrl[self.hand_ids].copy(),hand_q=data.qpos[self.hand_adr].copy(),
            hand_v=data.qvel[self.hand_dof].copy(),contact_signature=self.contact_signature(self.verify),
            predicted_onset=int(self.predicted_onset),snapshot_unchanged=1)

    def command(self,data,plan,t):
        super().command(data,plan,t)
        assert np.array_equal(data.ctrl[self.hand_ids],self.real_hand_ctrl),"Forecast changed real finger controls"
        applied=self.applied;log=self.rows[-1]
        log.update(hand_feedback=int(self.hand_feedback),filter_acceleration_limit=self.inner.config.acceleration,
            snapshot_unchanged=1,predicted_contact_onset=self.pending[-1]["predicted_onset"] if self.pending else 0,
            applied_hand_target_error_rad=None if applied is None else float(max(abs(applied["hand_ctrl"]-data.ctrl[self.hand_ids]))),
            applied_hand_q_error_rad=None if applied is None else float(max(abs(applied["hand_q"]-data.qpos[self.hand_adr]))),
            applied_hand_v_error_rad_s=None if applied is None else float(max(abs(applied["hand_v"]-data.qvel[self.hand_dof]))))
