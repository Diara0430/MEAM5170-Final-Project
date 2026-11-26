"""弹道预测、候选接取位姿和关节参考轨迹筛选。用于后续 MPC 的目标生成。

速度/加速度限制为本项目的实验设置。碰撞检查为离散采样，尚需动力学跟踪验证。
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.polynomial import Polynomial


@dataclass
class InterceptPlan:
    observation_time: float
    intercept_time: float
    point: np.ndarray
    palm_rotation: np.ndarray
    q_goal: np.ndarray
    coefficients: np.ndarray

    def reference(self, time):
        """返回世界时间对应的七轴位置、速度、加速度参考。"""
        duration=self.intercept_time-self.observation_time
        u=float(np.clip((time-self.observation_time)/duration,0.,1.))
        q=np.array([1.,u,u**2,u**3,u**4,u**5])@self.coefficients
        v=np.array([0.,1.,2*u,3*u**2,4*u**3,5*u**4])@self.coefficients/duration
        a=np.array([0.,0.,2.,6*u,12*u**2,20*u**3])@self.coefficients/duration**2
        return q,v,a


def extrema(poly):
    """包含端点和内部驻点，避免仅靠离散采样遗漏关节峰值。"""
    points=[0.,1.]
    points.extend(float(r.real) for r in poly.deriv().roots()
                  if abs(r.imag)<1e-8 and 0<r.real<1)
    values=poly(points)
    return float(min(values)),float(max(values))


class InterceptPlanner:
    def __init__(self,model,data,velocity_limit=1.,acceleration_limit=4.):
        if not np.isfinite([velocity_limit,acceleration_limit]).all() or min(velocity_limit,acceleration_limit)<=0:
            raise ValueError("Positive finite joint limits required")
        self.model=model
        self.state=data
        joints=[model.joint(f"joint{i}") for i in range(1,8)]
        self.qadr=[j.qposadr[0] for j in joints]
        self.dofs=[j.dofadr[0] for j in joints]
        ids=[model.actuator(f"act{i}").id for i in range(1,8)]
        ranges=model.jnt_range[[j.id for j in joints]]
        self.lower=np.maximum(ranges[:,0],model.actuator_ctrlrange[ids,0])
        self.upper=np.minimum(ranges[:,1],model.actuator_ctrlrange[ids,1])
        self.vmax=velocity_limit
        self.amax=acceleration_limit
        self.palm=model.body("hand/palm_lower").id
        self.object=model.body("projectile").id
        projectile = model.geom("projectile")
        if projectile.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
            # Orientation-independent clearance from the floor for a rotating box.
            self.radius = float(np.linalg.norm(projectile.size))
        elif projectile.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE:
            self.radius = float(projectile.size[0])
        else:
            raise ValueError("Expected a spherical or cuboid projectile")
        self.local_capture=np.array([0.,-.055,-.08])
        # 串联旋转关节链的保守几何上界，先过滤显然过远的目标。
        self.reach_bound=float(np.linalg.norm(self.local_capture))
        body=self.palm
        while body:
            self.reach_bound+=float(np.linalg.norm(model.body_pos[body]))
            for joint in range(model.body_jntadr[body],model.body_jntadr[body]+model.body_jntnum[body]):
                if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("Reach bound requires a fixed-base hinge-joint arm")
                self.reach_bound+=2*float(np.linalg.norm(model.jnt_pos[joint]))
            body=model.body_parentid[body]
        self.candidates=[]

    def solve_pose(self,point,rotation,seed_q=None):
        """在独立状态中用阻尼最小二乘求 IK，不移动正在运行的机械臂。"""
        m=self.model
        ik=mujoco.MjData(m)
        ik.qpos[:]=self.state.qpos
        if seed_q is not None:
            ik.qpos[self.qadr]=seed_q
        goal=point-rotation@self.local_capture
        jp,jr=np.empty((3,m.nv)),np.empty((3,m.nv))
        position_error=orientation_error=float("inf")
        for _ in range(140):
            # 雅可比只需要运动学与质心坐标，省去完整动力学和碰撞求解。
            mujoco.mj_kinematics(m,ik)
            mujoco.mj_comPos(m,ik)
            palm=ik.body(self.palm)
            current=palm.xmat.reshape(3,3)
            error_p=goal-palm.xpos
            error_r=.5*np.cross(current.T,rotation.T).sum(axis=0)
            position_error=float(np.linalg.norm(error_p))
            orientation_error=float(np.arccos(np.clip((np.trace(rotation.T@current)-1)/2,-1,1)))
            if position_error<.002 and orientation_error<np.deg2rad(2):
                return ik.qpos[self.qadr].copy(),position_error,orientation_error
            mujoco.mj_jacBody(m,ik,jp,jr,self.palm)
            jac=np.vstack((jp[:,self.dofs],jr[:,self.dofs]))
            delta=jac.T@np.linalg.solve(jac@jac.T+1e-4*np.eye(6),np.r_[error_p,error_r])
            ik.qpos[self.qadr]=np.clip(ik.qpos[self.qadr]+np.clip(delta,-.15,.15),self.lower,self.upper)
        return None,position_error,orientation_error

    def trajectory(self,q_goal,duration):
        """五次多项式连接当前状态与静止接取目标，终点加速度为零。"""
        q0=self.state.qpos[self.qadr]
        v0=self.state.qvel[self.dofs]
        a0=self.state.qacc[self.dofs]
        coefficients=np.zeros((6,7))
        coefficients[0]=q0
        coefficients[1]=v0*duration
        coefficients[2]=.5*a0*duration**2
        rhs=np.vstack((q_goal-coefficients[:3].sum(axis=0),
                       -coefficients[1]-2*coefficients[2],-2*coefficients[2]))
        coefficients[3:]=np.linalg.solve(np.array([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]]),rhs)
        peak_v=peak_a=0.
        margin=float("inf")
        for i,c in enumerate(coefficients.T):
            poly=Polynomial(c)
            lo,hi=extrema(poly)
            margin=min(margin,lo-self.lower[i],self.upper[i]-hi)
            peak_v=max(peak_v,max(abs(x) for x in extrema(poly.deriv()))/duration)
            peak_a=max(peak_a,max(abs(x) for x in extrema(poly.deriv(2)))/duration**2)
        return coefficients,peak_v,peak_a,margin

    def path_collision(self,coefficients):
        """沿张手参考轨迹取 41 个点，检查模型启用的自碰撞与地面穿透。"""
        probe=mujoco.MjData(self.model)
        probe.qpos[:]=self.state.qpos
        for u in np.linspace(0,1,41):
            probe.qpos[self.qadr]=np.array([Polynomial(c)(u) for c in coefficients.T])
            mujoco.mj_forward(self.model,probe)
            for c in probe.contact:
                bodies=self.model.geom_bodyid[[c.geom1,c.geom2]]
                if self.object not in bodies and c.dist<-.001:
                    return True
        return False

    def plan(self,position,velocity,now=0.,acceleration=None,
             times=None,angles=(0.,30.,45.,60.,90.),warm_start=False):
        p,v=np.asarray(position,dtype=float),np.asarray(velocity,dtype=float)
        acc=self.model.opt.gravity.copy() if acceleration is None else np.asarray(acceleration,dtype=float)
        if any(x.shape!=(3,) for x in (p,v,acc)) or not np.isfinite(np.r_[p,v,acc,now]).all():
            raise ValueError("Finite 3D object state required")
        times=np.arange(.08,.800001,.02) if times is None else np.asarray(times,dtype=float)
        angles=np.asarray(angles,dtype=float)
        if times.ndim!=1 or angles.ndim!=1 or not times.size or not angles.size or not np.isfinite(times).all() or min(times)<=0 or not np.isfinite(angles).all():
            raise ValueError("Positive candidate times and finite angles required")
        self.candidates=[]
        selected=None
        best=float("inf")
        seeds={}
        reference=self.state.body(self.palm).xmat.reshape(3,3).copy()
        for duration in times:
            point=p+v*duration+.5*acc*duration**2
            incoming=v+acc*duration
            for angle in angles:
                a=np.deg2rad(angle)
                turn=np.array([[np.cos(a),0.,np.sin(a)],[0.,1.,0.],[-np.sin(a),0.,np.cos(a)]])
                rotation=reference@turn
                inward_speed=float(incoming@rotation[:,2])
                incidence=float(np.rad2deg(np.arccos(np.clip(inward_speed/max(np.linalg.norm(incoming),1e-12),-1,1))))
                row=dict(candidate=len(self.candidates),intercept_time_s=now+duration,delay_s=duration,
                    angle_deg=angle,x_m=point[0],y_m=point[1],z_m=point[2],inward_speed_m_s=inward_speed,
                    incidence_deg=incidence,reason="",position_error_mm=None,orientation_error_deg=None,
                    peak_joint_speed_rad_s=None,peak_joint_accel_rad_s2=None,joint_margin_rad=None,score=None,
                    **{f"q{i}":None for i in range(1,8)})
                self.candidates.append(row)
                if point[2]<self.radius+.005:
                    row["reason"]="below_floor_clearance";continue
                if np.linalg.norm(point)>self.reach_bound+.002:
                    row["reason"]="outside_reach_bound";continue
                if inward_speed<.05 or incidence>75:
                    row["reason"]="unfavorable_approach";continue
                q,ep,er=self.solve_pose(point,rotation,seeds.get(float(angle)) if warm_start else None)
                row.update(position_error_mm=1000*ep,orientation_error_deg=np.rad2deg(er))
                if q is None:
                    row["reason"]="ik_no_solution";continue
                seeds[float(angle)]=q
                row.update({f"q{i}":q[i-1] for i in range(1,8)})
                c,pv,pa,margin=self.trajectory(q,duration)
                row.update(peak_joint_speed_rad_s=pv,peak_joint_accel_rad_s2=pa,joint_margin_rad=margin)
                if margin<0:
                    row["reason"]="joint_position_limit";continue
                if pv>self.vmax:
                    row["reason"]="insufficient_time_velocity";continue
                if pa>self.amax:
                    row["reason"]="insufficient_time_acceleration";continue
                if self.path_collision(c):
                    row["reason"]="sampled_path_collision";continue
                # 分数权重为预设实验选择；兼顾到达时间、关节运动和法向撞击速度。
                score=.5*duration+float(np.sum((q-self.state.qpos[self.qadr])**2))+.05*inward_speed**2
                row.update(reason="feasible_reference",score=score)
                if score<best:
                    best=score
                    selected=InterceptPlan(now,now+duration,point.copy(),rotation.copy(),q,c)
        # 热启动可能落入不同局部解；无方案时回退到当前关节状态重新搜索。
        if selected is None and warm_start:
            return self.plan(p,v,now=now,acceleration=acc,times=times,angles=angles,warm_start=False)
        return selected
