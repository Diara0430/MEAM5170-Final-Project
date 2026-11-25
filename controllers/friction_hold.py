"""捕获后保持控制：实际摩擦接触、相对位移/滚动阻尼和小规模接触力QP。

继承已验证中段柔顺闭指，前300 ms的捕获命令保持相同。
normal模式仅对摩擦极限附近的实际滑动响应；wrench模式还抑制相对迁移/转动。
接触力经过手指雅可比映射成原执行器的力矩修正，不给物体直接施力。
"""
from dataclasses import dataclass
import time

import mujoco
import numpy as np
import osqp
from scipy import sparse

from controllers.joint_aware_closure import JointAwareClosure


@dataclass(frozen=True)
class HoldConfig:
    start_s: float=.30
    support_s: float=.04
    period_s: float=.01
    position_gain: float=15.      # N/m
    velocity_gain: float=1.8      # Ns/m
    angular_damping: float=.0008 # Nm s/rad；球体不要求固定绝对朝向。
    friction_margin: float=.8
    risk_utilization: float=.85
    risk_slip_speed: float=.020
    max_normal_N: float=12.
    max_delta_N: float=.75
    max_force_correction_N: float=.75
    torque_filter_s: float=.020
    regularization: float=.05
    integral_gain: float=2.       # rad/(Nm s)，用力误差缓慢调整原位置伺服的平衡点。
    offset_speed: float=.05      # rad/s，保持阶段额外参考变化上限。
    offset_limit: float=.08      # rad，约4.6°，不重新执行大幅闭指。


class FrictionHoldClosure(JointAwareClosure):
    def __init__(self,model,hand,hold_mode="wrench",hold_config=None):
        super().__init__(model,hand,"compliant")
        if hold_mode not in ("normal","wrench"):raise ValueError("Unknown holding mode")
        self.hold_mode=hold_mode;self.hold_config=HoldConfig() if hold_config is None else hold_config
        c=self.hold_config
        if not np.all(np.isfinite(list(vars(c).values()))) or min(vars(c).values())<=0:
            raise ValueError("Finite positive holding parameters required")
        if c.friction_margin>=1.:raise ValueError("Friction margin must be below one")
        self.all_ids=np.array(hand.ids)
        self.all_qadr=[model.joint(f"hand/{i}").qposadr[0] for i in range(16)]
        self.all_kp=model.actuator_gainprm[self.all_ids,0].copy()
        self.all_kd=-model.actuator_biasprm[self.all_ids,2].copy()
        self.contact_states=[];self.net_wrench=np.zeros(6)
        self.relative_position=np.zeros(3);self.relative_velocity=np.zeros(3);self.relative_omega=np.zeros(3)
        self.hold_origin=None;self.support_time=0.;self.last_solve=-np.inf
        self.delta=np.zeros(16);self.filtered_delta=np.zeros(16)
        self.offset=np.zeros(16)
        self.rows=[];self.qp_calls=0;self.qp_failures=0;self.enabled_steps=0
        self.object_mass=model.body_mass[model.body("projectile").id]
        self.radius=model.geom_size[self.support.projectile,0]

    def observe(self,model,data,velocity=None):
        super().observe(model,data,velocity)
        # run_case传入步初速度，使雅可比、接触位置、接触力和速度按同一物理步对齐。
        v=data.qvel if velocity is None else velocity
        obj=data.body("projectile");palm=data.body(self.support.palm)
        rotation=palm.xmat.reshape(3,3)
        jo,jp,jro,jrp=(np.empty((3,model.nv)) for _ in range(4))
        mujoco.mj_jac(model,data,jo,jro,obj.xipos,obj.id)
        mujoco.mj_jac(model,data,jp,jrp,obj.xipos,palm.id)
        self.relative_position=rotation.T@(obj.xipos-palm.xpos)
        self.relative_velocity=rotation.T@((jo-jp)@v)
        self.relative_omega=(jro-jrp)@v
        self.rotation=rotation.copy()
        self.net_wrench[:]=0.;strongest={}
        for i,contact in enumerate(data.contact):
            if self.support.projectile not in (contact.geom1,contact.geom2) or contact.efc_address<0:continue
            second=contact.geom2==self.support.projectile
            other=contact.geom1 if second else contact.geom2
            group=self.support.group.get(other)
            if group is None:continue
            wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
            basis=(1. if second else -1.)*contact.frame.reshape(3,3).T
            force=basis@wrench[:3];r=contact.pos-obj.xipos
            self.net_wrench+=np.r_[force,np.cross(r,force)]
            if wrench[0]<.02:continue
            if contact.dim!=3:raise ValueError("Holding QP expects 3D friction contacts")
            jh,jc=np.empty((3,model.nv)),np.empty((3,model.nv))
            mujoco.mj_jac(model,data,jh,None,contact.pos,model.geom_bodyid[other])
            mujoco.mj_jac(model,data,jc,None,contact.pos,obj.id)
            slip=float(np.linalg.norm(contact.frame.reshape(3,3)[1:]@((jc-jh)@v)))
            eta=float(np.linalg.norm(wrench[1:3]/contact.friction[:2])/wrench[0])
            cross=np.array([[0.,-r[2],r[1]],[r[2],0.,-r[0]],[-r[1],r[0],0.]])
            sample=dict(group=group,force=wrench[:3].copy(),mu=contact.friction[:2].copy(),eta=eta,slip=slip,
                grasp=np.vstack((basis,cross@basis)),torque=jh[:,self.all_dofs].T@basis)
            if group not in strongest or wrench[0]>strongest[group]["force"][0]:strongest[group]=sample
        # 每指及掌面各选一个主要承载点；掌面反力必须能随手指夹持变化。
        # 掌面对应的手指雅可比为零，因此不会虚构可独立控制的掌面执行器。
        self.contact_states=[strongest[k] for k in sorted(strongest)]
        self.support_time=self.support_time+model.opt.timestep if self.support.holding else 0.

    def solve_hold(self,model,data,base_torque):
        started=time.perf_counter();c=self.hold_config;contacts=self.contact_states
        risk=any(s["eta"]>=c.risk_utilization and s["slip"]>=c.risk_slip_speed for s in contacts)
        info=dict(qp_failed=0,qp_called=0,risk=int(risk),wrench_residual=0.,wall_ms=0.)
        if self.hold_mode=="normal" and not risk:return np.zeros(16),info
        if not contacts:return np.zeros(16),info
        info["qp_called"]=1;self.qp_calls+=1
        g=np.hstack([s["grasp"] for s in contacts]);j=np.hstack([s["torque"] for s in contacts])
        measured=np.concatenate([s["force"] for s in contacts]);n=len(measured)
        desired=self.net_wrench.copy()
        if self.hold_mode=="wrench":
            force_local=-c.position_gain*(self.relative_position-self.hold_origin)-c.velocity_gain*self.relative_velocity
            correction=self.rotation@force_local
            norm=np.linalg.norm(correction)
            if norm>c.max_force_correction_N:correction*=c.max_force_correction_N/norm
            desired[:3]=-self.object_mass*model.opt.gravity+correction
            desired[3:]=-c.angular_damping*self.relative_omega
        scale=np.diag([1.,1.,1.,1./self.radius,1./self.radius,1./self.radius])
        task=scale@g;target=scale@(desired-self.net_wrench)
        matrix=task.T@task+c.regularization*np.eye(n);linear=-task.T@target
        constraints=[np.eye(n),j]
        lower=[np.full(n,-c.max_delta_N),-.95-base_torque]
        upper=[np.full(n,c.max_delta_N),.95-base_torque]
        for k,s in enumerate(contacts):
            normal=np.zeros(n);normal[3*k]=1.
            required=0.
            if risk and s["eta"]>=c.risk_utilization:
                required=min(c.max_normal_N,np.linalg.norm(s["force"][1:]/s["mu"])/c.friction_margin)
            # 若基线已超过12 N，不要求一步骤降到12 N；只允许维持或减小，禁止再增加。
            ceiling=max(c.max_normal_N,float(s["force"][0]))
            constraints.append(normal[None,:]);lower.append(np.array([required-s["force"][0]]));upper.append(np.array([ceiling-s["force"][0]]))
            # 内接矩形摩擦锥，保证优化接触力位于原椭圆摩擦锥以内。
            for axis in (1,2):
                for sign in (-1.,1.):
                    line=np.zeros(n);line[3*k+axis]=sign
                    line[3*k]=-c.friction_margin*s["mu"][axis-1]/np.sqrt(2.)
                    constraints.append(line[None,:]);lower.append(np.array([-np.inf]));upper.append(np.array([-line@measured]))
        a=np.vstack(constraints);lo=np.concatenate(lower);hi=np.concatenate(upper)
        solver=osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(2.*matrix),q=2.*linear,A=sparse.csc_matrix(a),l=lo,u=hi,
                     verbose=False,polishing=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=1500)
        result=solver.solve(raise_error=False)
        solution=result.x if result.info.status_val==1 else None
        if solution is None or max(np.max(lo-a@solution),np.max(a@solution-hi))>2e-5:
            self.qp_failures+=1;info["qp_failed"]=1;delta=np.zeros(16)
        else:
            delta=j@solution
            info["wrench_residual"]=float(np.linalg.norm(task@solution-target))
        info["wall_ms"]=1000.*(time.perf_counter()-started)
        return delta,info

    def command(self,model,data,elapsed):
        super().command(model,data,elapsed)
        c=self.hold_config
        if elapsed<c.start_s:return
        # 对向支撑可由一根手指与掌面组成；不能误要求必须两根手指同时承载。
        # 此时可控扳手空间较小，QP残差必须保留，不能宣称六维力闭合。
        ready=self.support_time>=c.support_s and any(s["group"]>=0 for s in self.contact_states)
        if self.hold_origin is None and ready:self.hold_origin=self.relative_position.copy()
        info=dict(qp_failed=0,qp_called=0,risk=0,wrench_residual=0.,wall_ms=0.)
        q=data.qpos[self.all_qadr];v=data.qvel[self.all_dofs]
        base=self.all_kp*(data.ctrl[self.all_ids]-q)-self.all_kd*v
        base=np.clip(base,-.95,.95)
        if ready and self.hold_origin is not None:
            self.enabled_steps+=1
            if elapsed-self.last_solve>=c.period_s-1e-10:
                self.delta,info=self.solve_hold(model,data,base);self.last_solve=elapsed
        else:self.delta[:]=0.
        alpha=1.-np.exp(-model.opt.timestep/c.torque_filter_s)
        self.filtered_delta+=alpha*(self.delta-self.filtered_delta)
        # 仅比例力矩修正很快被高刚度位置伺服抵消；有界积分移动其平衡点。
        # 暂时失去支撑或本步QP失败时停止积分；旧修正滤波衰减，平衡点偏移始终限幅。
        if ready and not info["qp_failed"]:
            speed=np.clip(c.integral_gain*self.filtered_delta,-c.offset_speed,c.offset_speed)
            self.offset=np.clip(self.offset+speed*model.opt.timestep,-c.offset_limit,c.offset_limit)
        # 用u的增量实现力矩修正；零修正时精确保留原指令，避免重新编码饱和力矩改变基线。
        target=data.ctrl[self.all_ids]+self.filtered_delta/self.all_kp+self.offset
        data.ctrl[self.all_ids]=np.clip(target,self.hand.limits[:,0],self.hand.limits[:,1])
        self.rows.append(dict(elapsed_s=elapsed,ready=int(ready),holding=int(self.hold_origin is not None),
            position_error_mm=0. if self.hold_origin is None else 1000.*float(np.linalg.norm(self.relative_position-self.hold_origin)),
            omega_norm=float(np.linalg.norm(self.relative_omega)),delta_torque_norm=float(np.linalg.norm(self.filtered_delta)),
            max_offset_rad=float(max(abs(self.offset))),
            **info))
