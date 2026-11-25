"""有行程限制的接触后导纳缓冲；不使用学习，不修改物体或碰撞材料。

虚拟质量和阻尼把净手部接触力转换成顺来球方向的退让目标。
K=0 避免恢复弹簧主动把手掌弹回；缓冲结束后保持已经达到的位置。
七变量 QP 把掌部任务转换成有关节和模型力矩约束的加速度参考。
参考约束不等于碰撞中的实际状态约束，实验另行记录实际超限。
"""
import time

import mujoco
import numpy as np
import osqp
from scipy import sparse


class ContactAdmittance:
    def __init__(self,model,data,now,incoming_velocity,mass=.5,damping=12.,stroke=.025):
        if min(mass,damping,stroke)<=0:
            raise ValueError("Admittance parameters must be positive")
        self.mass,self.damping,self.stroke=mass,damping,stroke
        self.qadr=[model.joint(f"joint{i}").qposadr[0] for i in range(1,8)]
        self.dofs=[model.joint(f"joint{i}").dofadr[0] for i in range(1,8)]
        self.ids=[model.actuator(f"act{i}").id for i in range(1,8)]
        joints=[model.joint(f"joint{i}").id for i in range(1,8)]
        self.lower=np.maximum(model.jnt_range[joints,0],model.actuator_ctrlrange[self.ids,0])
        self.upper=np.minimum(model.jnt_range[joints,1],model.actuator_ctrlrange[self.ids,1])
        self.limits=model.actuator_forcerange[self.ids]
        self.palm=model.body("hand/palm_lower").id
        self.projectile=model.geom("projectile").id
        self.hand_geoms={g for g,b in enumerate(model.geom_bodyid) if
                         (model.body(b).name or "").startswith("hand/")}
        f=model.joint("flight").dofadr[0]
        incoming=incoming_velocity[f:f+3]
        self.direction=incoming/max(np.linalg.norm(incoming),1e-12)
        self.origin=data.body(self.palm).xpos.copy()
        self.rotation=data.body(self.palm).xmat.reshape(3,3).copy()
        self.q=data.qpos[self.qadr].copy();self.v=data.qvel[self.dofs].copy()
        self.a=np.zeros(7)
        self.start=self.last=now
        self.s=self.sv=0.
        self.probe=mujoco.MjData(model)
        self.previous_j=None
        self.solver=None
        self.p_rows=np.array([i for j in range(7) for i in range(j+1)])
        self.p_cols=np.array([j for j in range(7) for i in range(j+1)])
        self.a_rows=np.tile(np.arange(14),7);self.a_cols=np.repeat(np.arange(7),14)
        self.rows=[]

    def contact_force(self,model,data):
        # 采用手受到的净矢量力，不能把互相抵消的夹持法向力相加作为缓冲输入。
        force=np.zeros(3)
        for i,c in enumerate(data.contact):
            if self.projectile not in (c.geom1,c.geom2) or c.efc_address<0:continue
            on_second=c.geom2==self.projectile
            other=c.geom1 if on_second else c.geom2
            if other not in self.hand_geoms:continue
            wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
            on_object=(1. if on_second else -1.)*c.frame.reshape(3,3).T@wrench[:3]
            force-=on_object
        return force

    def update(self,model,data,now):
        started=time.perf_counter();dt=model.opt.timestep
        elapsed=now-self.last
        # 先把上一步参考积分到当前时刻，避免参考位置提前一个物理步。
        self.q+=self.v*elapsed+.5*self.a*elapsed**2
        self.v+=self.a*elapsed;self.last=now
        along=max(0.,float(self.contact_force(model,data)@self.direction))
        # 前250 ms允许顺势退让；之后停止接触力驱动，让阻尼耗散虚拟速度。
        drive=along if now-self.start<.25 else 0.
        sa=float(np.clip((drive-self.damping*self.sv)/self.mass,-2.,2.))
        next_v=float(np.clip(self.sv+sa*dt,0.,.20))
        # 到行程终点前按加速度2 m/s²刹车，避免目标速度在限位处突跳。
        next_v=min(next_v,np.sqrt(max(0.,4.*(self.stroke-self.s))))
        old_v=self.sv
        self.s=min(self.stroke,self.s+.5*(old_v+next_v)*dt)
        self.sv=next_v;sa=(next_v-old_v)/dt
        probe=self.probe
        probe.qpos[:]=data.qpos;probe.qvel[:]=data.qvel
        probe.qpos[self.qadr]=self.q;probe.qvel[self.dofs]=self.v
        probe.joint("flight").qpos[:3]=[3.,3.,3.]
        mujoco.mj_forward(model,probe)
        palm=probe.body(self.palm)
        jp,jr=np.empty((3,model.nv)),np.empty((3,model.nv))
        mujoco.mj_jac(model,probe,jp,jr,palm.xpos,self.palm)
        j=np.vstack((jp[:,self.dofs],jr[:,self.dofs]))
        jdot_v=np.zeros(6) if self.previous_j is None else ((j-self.previous_j)/dt)@self.v
        self.previous_j=j.copy()
        rotation=palm.xmat.reshape(3,3)
        orientation_error=.5*sum(np.cross(rotation[:,i],self.rotation[:,i]) for i in range(3))
        desired=self.origin+self.direction*self.s
        command=np.r_[self.direction*sa+100.*(desired-palm.xpos)+20.*(self.direction*self.sv-j[:3]@self.v),
                      100.*orientation_error-20.*(j[3:]@self.v)]-jdot_v
        # 转动误差权重0.1 m/rad，维持掌面姿态；不主动扣腕。
        weight=np.diag([1.,1.,1.,.1,.1,.1]);a_task=weight@j;b_task=weight@command
        matrix=a_task.T@a_task+1e-4*np.eye(7)
        linear=-a_task.T@b_task+1e-4*2.*self.v
        full=np.empty((model.nv,model.nv));mujoco.mj_fullM(model,probe,full)
        inertia=full[np.ix_(self.dofs,self.dofs)]
        bias=(probe.qfrc_bias-probe.qfrc_passive)[self.dofs]
        # 离散一步后仍须能以4 rad/s²刹停，不能仅检查当前距离的速度上界。
        # 停车位置关于本步加速度单调，二分得到两个线性加速度边界。
        def stopped(acc,upper):
            qnext=self.q+self.v*dt+.5*acc*dt**2
            vnext=self.v+acc*dt
            return qnext+np.maximum(vnext,0.)**2/8. if upper else qnext-np.maximum(-vnext,0.)**2/8.
        left=np.full(7,-4.);right=np.full(7,4.)
        for _ in range(24):
            mid=(left+right)/2
            too_low=stopped(mid,False)<self.lower+1e-7
            left=np.where(too_low,mid,left);right=np.where(too_low,right,mid)
        low=np.maximum(right,(-1.-self.v)/dt)
        left=np.full(7,-4.);right=np.full(7,4.)
        for _ in range(24):
            mid=(left+right)/2
            too_high=stopped(mid,True)>self.upper-1e-7
            right=np.where(too_high,mid,right);left=np.where(too_high,left,mid)
        high=np.minimum(left,(1.-self.v)/dt)
        valid=bool(np.all(low<=high))
        solution=None
        if valid:
            p_values=2.*matrix[self.p_rows,self.p_cols]
            constraint=np.vstack((np.eye(7),inertia))
            a_values=constraint[self.a_rows,self.a_cols]
            lo=np.r_[low,self.limits[:,0]-bias];hi=np.r_[high,self.limits[:,1]-bias]
            if self.solver is None:
                self.solver=osqp.OSQP()
                self.solver.setup(P=sparse.csc_matrix((p_values,(self.p_rows,self.p_cols)),shape=(7,7)),q=2.*linear,
                    A=sparse.csc_matrix((a_values,(self.a_rows,self.a_cols)),shape=(14,7)),l=lo,u=hi,
                    verbose=False,polishing=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=1500)
            else:
                self.solver.update(Px=p_values,q=2.*linear,Ax=a_values,l=lo,u=hi)
            result=self.solver.solve(raise_error=False)
            if result.info.status_val==1 and max(np.max(lo-constraint@result.x),np.max(constraint@result.x-hi))<=1e-6:
                solution=result.x
        # 若实测初始速度/位置已经使参考约束不可行，显式记录并退回限加速度刹车。
        # 实际力矩始终由原模型的有力矩限位的位置执行器提供。
        if solution is None:
            self.a=np.clip(-self.v/dt,low,high) if valid else -4.*np.sign(self.v)
        else:self.a=np.clip(solution,-4.,4.)
        self.rows.append(dict(time_s=now,contact_ms=1000.*(now-self.start),force_along_N=along,
            target_retreat_mm=1000.*self.s,target_speed_m_s=self.sv,qp_failed=int(solution is None),
            reference_speed=float(max(abs(self.v))),reference_accel=float(max(abs(self.a))),
            reference_margin=float(min(np.r_[self.q-self.lower,self.upper-self.q])),
            wall_ms=1000.*(time.perf_counter()-started)))

    def reference(self,t):
        tail=max(0.,t-self.last)
        return self.q+self.v*tail+.5*self.a*tail**2,self.v+self.a*tail,self.a
