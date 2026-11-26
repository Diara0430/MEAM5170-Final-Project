"""在真实力矩执行器前，用一步动力学预测限制机械臂的积分加速度。

不修改实际 qpos/qvel，不修改材料，不直接对球施力，无机器学习。
独立 MjData 只预测当前已知状态及当前手指命令的一步响应。
这是已知仿真模型下的控制实验，不是未知接触/实机的安全保证。
"""
from dataclasses import dataclass
import time

import mujoco
import numpy as np
import osqp
from scipy import sparse

from controllers.arm_reference_tracker import ArmReferenceTracker


@dataclass(frozen=True)
class ArmFilterConfig:
    acceleration: float=3.8       # 给原来的4 rad/s²判据留0.2余量，不改变判据。
    speed: float=1.0
    difference_torque: float=.05 # Nm，局部响应的有限差分步长。
    iterations: int=3
    slack_penalty: float=1e5


class DiscreteArmFilter(ArmReferenceTracker):
    def __init__(self,model,config=None):
        super().__init__(model)
        self.config=ArmFilterConfig() if config is None else config
        c=self.config
        if not np.all(np.isfinite([c.acceleration,c.speed,c.difference_torque,c.slack_penalty])) or min(c.acceleration,c.speed,c.difference_torque,c.slack_penalty)<=0 or c.iterations<1:
            raise ValueError("Positive finite filter settings required")
        self.probe=mujoco.MjData(model)
        joints=[model.joint(f"joint{i}").id for i in range(1,8)]
        self.lower=model.jnt_range[joints,0].copy();self.upper=model.jnt_range[joints,1].copy()
        self.rows=[];self.solver=None;self.last_torque=None
        # 固定QP稀疏结构并复用工作区，避免每2 ms重新分配求解器。
        a_rows=[];a_cols=[]
        for j in range(7):
            a_rows.extend([j]+list(range(14,28)));a_cols.extend([j]*15)
        for j in range(7):
            a_rows.extend([7+j,14+j,21+j]);a_cols.extend([7+j]*3)
        self.a_rows=np.array(a_rows);self.a_cols=np.array(a_cols)

    def command(self,data,plan,t):
        started=time.perf_counter();m=self.model;c=self.config;dt=m.opt.timestep
        # 第一步：保留原跟踪器作为名义命令，历史默认路径不变。
        super().command(data,plan,t)
        q=data.qpos[self.qadr].copy();v=data.qvel[self.dofs].copy()
        limits=m.actuator_ctrlrange[self.ids];forces=m.actuator_forcerange[self.ids]
        lo=np.maximum(forces[:,0],self.kp*(limits[:,0]-q)-self.kd*v)
        hi=np.minimum(forces[:,1],self.kp*(limits[:,1]-q)-self.kd*v)
        nominal=np.clip(self.kp*(data.ctrl[self.ids]-q)-self.kd*v,lo,hi)
        calls=0
        def predict(torque):
            nonlocal calls
            # 完整复制求解器热启动等状态，探针与真实下一步使用相同积分器。
            mujoco.mj_copyData(self.probe,m,data)
            self.probe.ctrl[self.ids]=q+(torque+self.kd*v)/self.kp
            mujoco.mj_step(m,self.probe);calls+=1
            return (self.probe.qvel[self.dofs]-v)/dt
        low=np.maximum(-c.acceleration,(-c.speed-v)/dt)
        high=np.minimum(c.acceleration,(c.speed-v)/dt)
        # implicitfast的铰链位置由步末速度积分，增加一步关节位置限制。
        low=np.maximum(low,(self.lower-q-v*dt)/dt**2)
        high=np.minimum(high,(self.upper-q-v*dt)/dt**2)
        def violation(a):return max(0.,float(np.max(low-a)),float(np.max(a-high)))
        nominal_acc=predict(nominal);torque=nominal.copy();acc=nominal_acc.copy()
        best=(violation(acc),torque.copy(),acc.copy());max_slack=0.;failed=0;iterations=0
        # 上一步命令也是可检验候选。名义命令突变时从较好的邻近点线性化，
        # 避免在饱和/静摩擦死区中得到无效的局部导数。
        if violation(acc)>1e-4 and self.last_torque is not None:
            previous=np.clip(self.last_torque,lo,hi);previous_acc=predict(previous)
            if violation(previous_acc)<best[0]:
                torque,acc=previous,previous_acc;best=(violation(acc),torque.copy(),acc.copy())
        # 第二步：仅在预测超限时求7变量力矩修正；接触切换时迭代重新线性化。
        for iteration in range(c.iterations):
            if violation(acc)<=1e-4 and (iteration or violation(nominal_acc)<=1e-4):break
            iterations+=1;jac=np.empty((7,7))
            for j in range(7):
                delta=min(c.difference_torque,hi[j]-torque[j])
                if delta<1e-6:delta=-min(c.difference_torque,torque[j]-lo[j])
                if abs(delta)<1e-9:jac[:,j]=0.;continue
                shifted=torque.copy();shifted[j]+=delta
                jac[:,j]=(predict(shifted)-acc)/delta
            offset=acc-jac@torque
            # 有限力矩可能使约束无解：显式松弛并报告，不能伪造实际加速度。
            a_values=np.r_[np.concatenate([np.r_[1.,jac[:,j],jac[:,j]] for j in range(7)]),
                np.tile([1.,1.,-1.],7)]
            # 接触非线性导致首次修正不足时，改求内部解，避免反复贴着边界振荡。
            inner=.8*c.acceleration if iteration and violation(acc)>.05 else c.acceleration
            solve_low=np.maximum(low,-inner);solve_high=np.minimum(high,inner)
            if np.any(solve_low>solve_high):solve_low,solve_high=low,high
            lower=np.r_[lo,np.zeros(7),solve_low-offset,np.full(7,-np.inf)]
            upper=np.r_[hi,np.full(7,np.inf),np.full(7,np.inf),solve_high-offset]
            if self.solver is None:
                matrix=sparse.diags(np.r_[np.ones(7),np.full(7,c.slack_penalty)],format="csc")
                constraint=sparse.csc_matrix((a_values,(self.a_rows,self.a_cols)),shape=(28,14))
                self.solver=osqp.OSQP();self.solver.setup(P=matrix,q=np.r_[-nominal,np.zeros(7)],A=constraint,l=lower,u=upper,
                    verbose=False,polishing=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=4000)
            else:self.solver.update(q=np.r_[-nominal,np.zeros(7)],Ax=a_values,l=lower,u=upper)
            self.solver.warm_start(x=np.r_[torque,np.zeros(7)],y=np.zeros(28))
            result=self.solver.solve(raise_error=False)
            if result.info.status_val not in (1,2) or result.x is None:
                failed+=1;break
            candidate=np.clip(result.x[:7],lo,hi);max_slack=max(max_slack,float(max(result.x[7:])))
            candidate_acc=predict(candidate)
            new_error=violation(candidate_acc)
            if new_error<best[0]-1e-8 or (new_error<=1e-4 and best[0]<=1e-4 and
                np.linalg.norm(candidate-nominal)<np.linalg.norm(best[1]-nominal)):
                best=(new_error,candidate.copy(),candidate_acc.copy())
            torque,acc=candidate,candidate_acc
        # 第三步：执行通过真实积分探针复核的最佳命令，仍经原执行器限力矩。
        _,torque,acc=best
        self.last_torque=torque.copy()
        data.ctrl[self.ids]=q+(torque+self.kd*v)/self.kp
        wall=time.perf_counter()-started
        self.rows.append(dict(time_s=t,nominal_peak_accel=float(max(abs(nominal_acc))),
            predicted_peak_accel=float(max(abs(acc))),predicted_violation=violation(acc),
            torque_correction_Nm=float(max(abs(torque-nominal))),max_slack=max_slack,
            qp_failed=failed,linearizations=iterations,probe_steps=calls,wall_ms=1000.*wall,
            overrun=int(wall>dt),**{f"predicted_accel_{i+1}":float(acc[i]) for i in range(7)}))
