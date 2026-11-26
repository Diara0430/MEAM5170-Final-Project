"""从实测状态重建接触参考，避免独立虚拟关节轨迹与实际滤波结果分离。

只生成名义加速度；最终力矩约束由同一个DiscreteArmFilter处理。
顺来球方向采用质量-阻尼退让，无回弹弹簧；侧向和掌面朝向反馈保持。
"""
from dataclasses import dataclass
import time

import mujoco
import numpy as np


@dataclass
class StatePlan:
    time: float
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray

    def reference(self,t):
        h=max(0.,t-self.time)
        return self.q+self.v*h+.5*self.a*h*h,self.v+self.a*h,self.a


class MeasuredContactReference:
    def __init__(self,model,data,now,incoming_velocity,mass=.5,damping=12.,stroke=.025):
        if not np.all(np.isfinite([mass,damping,stroke])) or min(mass,damping,stroke)<=0:
            raise ValueError("Positive finite contact parameters required")
        self.mass,self.damping,self.stroke=mass,damping,stroke
        joints=[model.joint(f"joint{i}") for i in range(1,8)]
        self.qadr=[j.qposadr[0] for j in joints];self.dofs=[j.dofadr[0] for j in joints]
        self.palm=model.body("hand/palm_lower").id;self.projectile=model.geom("projectile").id
        self.hand_geoms={g for g,b in enumerate(model.geom_bodyid) if (model.body(b).name or "").startswith("hand/")}
        f=model.joint("flight").dofadr[0];incoming=incoming_velocity[f:f+3]
        self.direction=incoming/max(np.linalg.norm(incoming),1e-12)
        self.origin=data.body(self.palm).xpos.copy();self.rotation=data.body(self.palm).xmat.reshape(3,3).copy()
        self.start=now;self.probe=mujoco.MjData(model);self.rows=[];self.current=None

    def force_along(self,model,data):
        force=np.zeros(3)
        for i,c in enumerate(data.contact):
            if self.projectile not in (c.geom1,c.geom2) or c.efc_address<0:continue
            second=c.geom2==self.projectile;other=c.geom1 if second else c.geom2
            if other not in self.hand_geoms:continue
            wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
            force-=(1. if second else -1.)*c.frame.reshape(3,3).T@wrench[:3]
        return max(0.,float(force@self.direction))

    def state_plan(self,model,data,now,record=False):
        # 探针只重算实测状态的运动学，真实qpos/qvel不变；接触输入取已解出的上一物理步。
        force=self.force_along(model,data);probe=self.probe
        mujoco.mj_copyData(probe,model,data)
        mujoco.mj_kinematics(model,probe);mujoco.mj_comPos(model,probe);mujoco.mj_comVel(model,probe)
        palm=probe.body(self.palm);rot=palm.xmat.reshape(3,3)
        jp,jr,jpd,jrd=[np.empty((3,model.nv)) for _ in range(4)]
        mujoco.mj_jac(model,probe,jp,jr,palm.xpos,self.palm)
        mujoco.mj_jacDot(model,probe,jpd,jrd,palm.xpos,self.palm)
        cart_v=jp@probe.qvel;omega=jr@probe.qvel;shift=palm.xpos-self.origin
        s=float(shift@self.direction);sv=float(cart_v@self.direction)
        drive=force if now-self.start<.25 else 0.
        sa=float(np.clip((drive-self.damping*sv)/self.mass,-1.5,1.5))
        # 100 ms前视的软行程约束。它仅修正目标，并非实际行程不超限的保证。
        horizon=.1
        sa=min(sa,2.*(self.stroke-s-sv*horizon)/horizon**2)
        sa=max(sa,2.*(-max(0.,s)-sv*horizon)/horizon**2)
        sa=float(np.clip(sa,-1.5,1.5))
        lateral=shift-self.direction*s;lateral_v=cart_v-self.direction*sv
        orientation=.5*sum(np.cross(rot[:,i],self.rotation[:,i]) for i in range(3))
        command=np.r_[self.direction*sa-80.*lateral-16.*lateral_v,80.*orientation-16.*omega]
        j=np.vstack((jp[:,self.dofs],jr[:,self.dofs]))
        rhs=command-np.vstack((jpd,jrd))@probe.qvel;weight=np.diag([1.,1.,1.,.01,.01,.01])
        q=probe.qpos[self.qadr].copy();v=probe.qvel[self.dofs].copy()
        desired=np.linalg.solve(j.T@weight@j+2e-4*np.eye(7),j.T@weight@rhs-4e-4*v)
        low=np.maximum(-3.,(-1.-v)/model.opt.timestep);high=np.minimum(3.,(1.-v)/model.opt.timestep)
        infeasible=bool(np.any(low>high))
        a=np.clip(-v/model.opt.timestep,-3.,3.) if infeasible else np.clip(desired,low,high)
        plan=StatePlan(now,q,v,a)
        if record:self.rows.append(dict(time_s=now,contact_ms=1000.*(now-self.start),force_along_N=force,
            measured_retreat_mm=1000.*s,measured_speed_m_s=sv,stroke_excess_mm=1000.*max(0.,s-self.stroke),
            reference_peak_accel=float(max(abs(a))),reference_infeasible=int(infeasible)))
        return plan

    def update(self,model,data,now):
        start=time.perf_counter();self.current=self.state_plan(model,data,now,record=True)
        self.rows[-1]["wall_ms"]=1000.*(time.perf_counter()-start)

    def reference(self,t):return self.current.reference(t)
