"""明确模拟机械臂指令延迟；可用已知模型预测指令真正生效时的状态。

延迟期间真实仿真持续前进，旧位置伺服目标继续执行。手指仍以2 ms更新。
预测只读当前状态，冻结尚未知的未来手指命令；不读取真实未来轨迹。
固定延迟压力测试不等于完整硬件/调度器实时认证。
"""
import time

import mujoco
import numpy as np

from controllers.discrete_arm_filter import DiscreteArmFilter


class ScheduledArmTracker:
    def __init__(self,model,delay_steps=1,predictive=False):
        if not isinstance(delay_steps,int) or delay_steps<0:raise ValueError("Nonnegative integer delay required")
        self.model=model;self.delay_steps=delay_steps;self.predictive=predictive
        self.inner=DiscreteArmFilter(model)
        self.ids=self.inner.ids;self.qadr=self.inner.qadr;self.dofs=self.inner.dofs
        self.preview=mujoco.MjData(model);self.pending=[];self.current=None;self.applied=None;self.rows=[]

    def command(self,data,plan,t):
        started=time.perf_counter();m=self.model;dt=m.opt.timestep;step=round(t/dt)
        if self.current is None:self.current=data.ctrl[self.ids].copy()
        # 先交付到期的旧指令，计算本轮新命令时不能把它提前作用于真实系统。
        due=[x for x in self.pending if x["due"]<=step]
        if due:
            self.applied=max(due,key=lambda x:x["generated"]);self.current=self.applied["ctrl"].copy()
        self.pending=[x for x in self.pending if x["due"]>step]
        predicted_state=None
        if self.predictive and self.delay_steps:
            # 用当前已知指令队列传播到交付时刻。未来手指ctrl未知，保持当前值。
            mujoco.mj_copyData(self.preview,m,data);ctrl=self.current.copy()
            for offset in range(self.delay_steps):
                available=[x for x in self.pending if x["due"]<=step+offset]
                if available:ctrl=max(available,key=lambda x:x["generated"])["ctrl"]
                self.preview.ctrl[self.ids]=ctrl;mujoco.mj_step(m,self.preview)
            future=t+self.delay_steps*dt
            predicted_state=np.r_[self.preview.qpos[self.qadr],self.preview.qvel[self.dofs]].copy()
            predicted_plan=plan.state_plan(m,self.preview,future) if hasattr(plan,"state_plan") else plan
            self.inner.command(self.preview,predicted_plan,future)
            candidate=self.preview.ctrl[self.ids].copy()
        else:
            self.inner.command(data,plan,t);candidate=data.ctrl[self.ids].copy()
        inner=self.inner.rows[-1]
        item=dict(generated=step,due=step+self.delay_steps,ctrl=candidate,
            prediction=np.array([inner[f"predicted_accel_{i+1}"] for i in range(7)]),predicted_state=predicted_state)
        self.pending.append(item)
        if not self.delay_steps:self.applied=item;self.current=candidate.copy();self.pending.clear()
        data.ctrl[self.ids]=self.current
        wall=time.perf_counter()-started
        age=-1 if self.applied is None else step-self.applied["generated"]
        error_q=error_v=None
        if self.applied is not None and self.applied["predicted_state"] is not None:
            difference=self.applied["predicted_state"]-np.r_[data.qpos[self.qadr],data.qvel[self.dofs]]
            error_q=float(max(abs(difference[:7])));error_v=float(max(abs(difference[7:])))
        self.rows.append(dict(time_s=t,delay_steps=self.delay_steps,predictive=int(self.predictive),
            applied_generation_step=-1 if self.applied is None else self.applied["generated"],applied_age_steps=age,
            applied_forecast_q_error_rad=error_q,applied_forecast_v_error_rad_s=error_v,
            wall_ms=1000.*wall,overrun=int(wall>dt),
            exceeds_injected_delay=int(wall>max(1,self.delay_steps)*dt),
            generated_prediction_peak=inner["predicted_peak_accel"],generated_violation=inner["predicted_violation"],
            qp_failed=inner["qp_failed"],probe_steps=inner["probe_steps"]+self.delay_steps*int(self.predictive)))

    @property
    def applied_acceleration_prediction(self):return None if self.applied is None else self.applied["prediction"]
