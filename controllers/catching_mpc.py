"""两段加速度的线性化滚动时域 QP。

对应 Lecture 12 的有限时域状态约束和 Lecture 13 的雅可比任务跟踪。
每个候选优化14个变量，状态使用双积分模型；实际机器人由逆动力学反馈跟踪。
"""
from dataclasses import dataclass
import time

import mujoco
import numpy as np
import osqp
from scipy import sparse

from controllers.intercept_planner import InterceptPlanner


@dataclass
class AccelerationReference:
    observation_time: float
    intercept_time: float
    point: np.ndarray
    palm_rotation: np.ndarray
    q0: np.ndarray
    v0: np.ndarray
    first: np.ndarray
    second: np.ndarray
    match_fraction: float = 0.0
    incoming_speed: float = 0.0

    def reference(self,t):
        duration=self.intercept_time-self.observation_time
        s=float(np.clip(t-self.observation_time,0.,duration))
        h=duration/2
        if s<=h:
            q=self.q0+self.v0*s+.5*self.first*s*s
            v=self.v0+self.first*s
            a=self.first
        else:
            tail=s-h
            middle_q=self.q0+self.v0*h+.5*self.first*h*h
            middle_v=self.v0+self.first*h
            q=middle_q+middle_v*tail+.5*self.second*tail*tail
            v=middle_v+self.second*tail
            a=self.second
        if t>=self.intercept_time:
            q=q+v*max(0.,t-self.intercept_time)
            a=np.zeros(7)
        return q,v,a


class CatchingMPC:
    period=.020

    def __init__(self,model,data,velocity_match=.2,acceleration_limit=4.,follow_through=None,approach=None):
        if not np.isfinite(velocity_match) or not 0<=velocity_match<=1:
            raise ValueError("Velocity matching fraction must be in [0,1]")
        if not np.isfinite(acceleration_limit) or acceleration_limit<=0:
            raise ValueError("Positive finite planning acceleration limit required")
        self.model=model
        self.checker=InterceptPlanner(model,data)
        self.match=velocity_match
        self.acceleration_limit=acceleration_limit
        self.follow_through=follow_through
        self.approach=approach
        self.orientation=data.body(self.checker.palm).xmat.reshape(3,3).copy()
        self.ids=[model.actuator(f"act{i}").id for i in range(1,8)]
        self.force_limits=model.actuator_forcerange[self.ids]
        self.last_solution=None

    @staticmethod
    def maps(t,h):
        """从两段恒定关节加速度到时刻 t 的位置、速度增量。"""
        if t<=h:
            return np.hstack((.5*t*t*np.eye(7),np.zeros((7,7)))),np.hstack((t*np.eye(7),np.zeros((7,7))))
        return np.hstack((h*(t-.5*h)*np.eye(7),.5*(t-h)**2*np.eye(7))),np.hstack((h*np.eye(7),(t-h)*np.eye(7)))

    def solve(self,data,p,v,now,previous,durations=None):
        start=time.perf_counter()
        m=self.model
        ck=self.checker
        q_now=data.qpos[ck.qadr].copy()
        v_now=data.qvel[ck.dofs].copy()
        _,_,old_acc=previous.reference(now)
        # 新解在一个控制周期后可用，优化起点先预测到该时刻。
        lag=self.period
        q0=q_now+lag*v_now+.5*lag*lag*old_acc
        v0=v_now+lag*old_acc
        palm=data.body(ck.palm)
        capture=palm.xpos+palm.xmat.reshape(3,3)@ck.local_capture
        jp,jr=np.empty((3,m.nv)),np.empty((3,m.nv))
        mujoco.mj_jac(m,data,jp,jr,capture,ck.palm)
        j=jp[:,ck.dofs];r=jr[:,ck.dofs]
        capture0=capture+j@(q0-q_now)
        orientation_error=.5*np.cross(palm.xmat.reshape(3,3).T,self.orientation.T).sum(axis=0)-r@(q0-q_now)
        inertia=np.empty((m.nv,m.nv))
        mujoco.mj_fullM(m,data,inertia)
        mass=inertia[np.ix_(ck.dofs,ck.dofs)]
        bias=data.qfrc_bias[ck.dofs]-data.qfrc_passive[ck.dofs]
        torque_map=sparse.block_diag((mass,mass)).toarray()
        best=None;best_score=float("inf")
        solved=0
        if durations is None:
            durations = (.12,.18,.24,.30,.36,.42,.48,.54,.60,.66)
        for duration in durations:
            flight_time=lag+duration
            point=p+v*flight_time+.5*m.opt.gravity*flight_time**2
            incoming=v+m.opt.gravity*flight_time
            if point[2]<ck.radius+.005 or np.linalg.norm(point)>ck.reach_bound+.002:
                continue
            if incoming@self.orientation[:,2]<.05 or np.linalg.norm(point-capture0)>.25:
                continue
            target_rotation = self.orientation
            fraction = self.match
            incoming_speed = float(np.linalg.norm(incoming))
            rotation_error = orientation_error
            if self.approach is not None:
                fraction = self.approach.match_fraction(incoming_speed)
                if self.approach.tilt:
                    target_rotation = self.approach.palm_target(self.orientation, incoming)
                    rotation_error = .5 * np.cross(palm.xmat.reshape(3, 3).T, target_rotation.T).sum(axis=0) - r @ (q0 - q_now)
            h=duration/2
            qmap,vmap=self.maps(duration,h)
            a_pos=j@qmap
            b_pos=point-capture0-j@(v0*duration)
            a_vel=j@vmap
            b_vel=fraction*incoming-j@v0
            a_rot=r@qmap
            b_rot=rotation_error-r@(v0*duration)
            # 终点位置、相对速度、姿态、加速度幅度共同进入二次目标。
            weights=((a_pos,b_pos,2e4),(a_vel,b_vel,4.),(a_rot,b_rot,100.))
            matrix=.002*np.eye(14)
            linear=np.zeros(14)
            for a,b,w in weights:
                matrix+=w*a.T@a
                linear-=w*a.T@b
            constraint=[np.eye(14),torque_map]
            lower=[np.full(14,-self.acceleration_limit),np.tile(self.force_limits[:,0]-bias,2)]
            upper=[np.full(14,self.acceleration_limit),np.tile(self.force_limits[:,1]-bias,2)]
            for t in np.linspace(duration/4,duration,4):
                qm,vm=self.maps(t,h)
                constraint.extend((qm,vm))
                lower.extend((ck.lower-q0-v0*t,-np.ones(7)-v0))
                upper.extend((ck.upper-q0-v0*t,np.ones(7)-v0))
            constraint.extend((a_pos,a_rot))
            lower.extend((b_pos-.005,b_rot-.03))
            upper.extend((b_pos+.005,b_rot+.03))
            a=np.vstack(constraint);lo=np.concatenate(lower);hi=np.concatenate(upper)
            solver=osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(2*matrix),q=2*linear,A=sparse.csc_matrix(a),l=lo,u=hi,
                         verbose=False,polishing=False,max_iter=2000,eps_abs=1e-5,eps_rel=1e-5)
            if self.last_solution is not None:
                solver.warm_start(x=self.last_solution)
            result=solver.solve(raise_error=False)
            if result.info.status_val!=1:
                continue
            x=result.x
            if max(np.max(lo-a@x),np.max(a@x-hi))>2e-4:
                continue
            plan=AccelerationReference(now+lag,now+lag+duration,point,target_rotation.copy(),q0,v0,x[:7],x[7:],fraction,incoming_speed)
            if self.follow_through is not None:
                end_q, end_v, _ = plan.reference(plan.intercept_time)
                if not self.follow_through.fits(end_q, end_v, ck.lower, ck.upper):
                    continue
            probe=mujoco.MjData(m);probe.qpos[:]=data.qpos
            valid=True
            # 五点非线性运动学复核，不把线性化终点误差当作实际位置误差。
            for fraction in np.linspace(0,1,5):
                q,vel,_=plan.reference(plan.observation_time+duration*fraction)
                if min(np.r_[q-ck.lower,ck.upper-q])<0 or max(abs(vel))>1.0002:
                    valid=False;break
                probe.qpos[ck.qadr]=q
                mujoco.mj_forward(m,probe)
                if any(ck.object not in m.geom_bodyid[[c.geom1,c.geom2]] and c.dist<-.001 for c in probe.contact):
                    valid=False;break
            if not valid:
                continue
            end=probe.body(ck.palm)
            actual=end.xpos+end.xmat.reshape(3,3)@ck.local_capture
            if np.linalg.norm(actual-point)>.012:
                continue
            solved+=1
            score=sum(w*float(np.sum((a@x-b)**2)) for a,b,w in weights)+.002*float(x@x)+.1*duration
            if score<best_score:
                best_score=score;best=plan;self.last_solution=x.copy()
        return best,dict(wall_s=time.perf_counter()-start,feasible_candidates=solved)


def braking_reference(model,data,now):
    """接触后保留当前运动状态，限幅减速，再保持该关节位置。"""
    q=np.array([data.joint(f"joint{i}").qpos[0] for i in range(1,8)])
    v=np.array([data.joint(f"joint{i}").qvel[0] for i in range(1,8)])
    duration=max(.2,float(max(abs(v)))/4.)
    a=-v/duration
    palm=data.body("hand/palm_lower")
    return AccelerationReference(now,now+duration,palm.xpos.copy(),palm.xmat.reshape(3,3).copy(),q,v,a,a)
