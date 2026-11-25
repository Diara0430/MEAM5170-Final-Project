"""针对从指尖往掌面数第二段指节的角度、速度及触球后的柔顺闭指。

LEAP XML中这段指节的父关节为2/6/10，拇指对应14，body名称是dip/thumb_dip。
limited模式仅修改这四个关节的形状参考，其他关节沿用140 ms基准闭指。
compliant模式还在该段/远端承载时降低刚度、提高阻尼；不使用机器学习。
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.contact_closure import ContactClosure


@dataclass(frozen=True)
class ClosureConfig:
    middle_angle: float=1.30      # rad，保留球形包围所需终点；用提前减速降低实际过冲角度。
    thumb_angle: float=1.10       # rad，保留拇指关闭逃逸通道所需的最终角度。
    middle_speed: float=8.5       # rad/s，形状参考速度；不能等同于实际速度硬约束。
    middle_accel: float=80.       # rad/s²，armature约0.01，惯性力矩约0.8 Nm，低于0.95 Nm上限。
    unloaded_duration: float=.14 # s，其余关节保留原始包围时序。
    contact_stiffness: float=6.  # Nm/rad，单侧接触时的等效刚度，原值20。
    opposed_stiffness: float=10. # 已形成对向支撑时渐进提高刚度。
    damping_scale: float=1.5     # 触球后提高等效速度阻尼，吸收指节动能。
    preshape_angle: float=.30   # 接触前中段仅轻弯0.30 rad，减少触球后所需行程。
    preshape_duration: float=.12
    preshape_lead: float=.20    # 在预测穿过捕获平面前200 ms开始，保留进入通道。


def trapezoid(elapsed,distance,speed,acceleration,initial_speed=0.):
    """给定初始正向速度并以零速结束的解析梯形/三角形轨迹。"""
    v0=max(0.,initial_speed)
    if distance+1e-12<v0*v0/(2.*acceleration):raise ValueError("Insufficient stopping distance")
    vp=min(speed,np.sqrt(acceleration*distance+.5*v0*v0))
    ta=max(0.,(vp-v0)/acceleration);td=vp/acceleration
    accel_distance=(vp*vp-v0*v0)/(2.*acceleration)
    tc=max(0.,(distance-accel_distance-vp*vp/(2.*acceleration))/max(vp,1e-12))
    t=max(0.,elapsed)
    if t<ta:return v0*t+.5*acceleration*t*t,v0+acceleration*t,acceleration
    if t<ta+tc:return accel_distance+vp*(t-ta),vp,0.
    tail=t-ta-tc
    if tail<td:return distance-.5*acceleration*(td-tail)**2,acceleration*(td-tail),-acceleration
    return distance,0.,0.


class JointAwareClosure:
    def __init__(self,model,hand,mode="compliant",config=None):
        if mode not in ("limited","preshape","compliant"):raise ValueError("Unknown joint-aware closure mode")
        self.mode=mode;self.config=ClosureConfig() if config is None else config
        c=self.config
        values=list(vars(c).values())
        if not np.all(np.isfinite(values)) or min(values)<=0:raise ValueError("Finite positive closure parameters required")
        if c.preshape_angle>=min(c.middle_angle,c.thumb_angle) or c.preshape_lead<c.preshape_duration:
            raise ValueError("Preshape must preserve closure travel and finish before the nominal arrival")
        self.hand=hand;self.support=ContactClosure(model,hand)
        self.middle=np.array([2,6,10,14])
        self.ids=np.array(hand.ids)[self.middle]
        joints=[model.joint(f"hand/{i}") for i in self.middle]
        self.qadr=[j.qposadr[0] for j in joints];self.dofs=[j.dofadr[0] for j in joints]
        self.bodies=[int(model.jnt_bodyid[j.id]) for j in joints]
        self.kp=model.actuator_gainprm[self.ids,0].copy()
        self.kd=-model.actuator_biasprm[self.ids,2].copy()
        self.all_dofs=[model.joint(f"hand/{i}").dofadr[0] for i in range(16)]
        self.inertia=np.empty((model.nv,model.nv))
        self.goals=np.array([c.middle_angle]*3+[c.thumb_angle])
        if np.any(self.goals>hand.limits[self.middle,1]):raise ValueError("Middle target exceeds model joint limits")
        self.loads=np.zeros(4);self.blend=np.zeros(4)
        self.shape=hand.open_pose.copy();self.shape_velocity=np.zeros(16)
        self.shape_acceleration=np.zeros(16)
        self.effective_stiffness=self.kp.copy()
        self.close_start=None;self.close_speed=None
        self.affected={}
        # 中段或其远端碰球都会对该关节施加负载；近端/掌部负载不冒充中段负载。
        for geom,body in enumerate(model.geom_bodyid):
            current=int(body)
            while current:
                if current in self.bodies:
                    self.affected[geom]=self.bodies.index(current);break
                current=int(model.body_parentid[current])

    def prepare(self,model,data,now,nominal):
        """接触前先轻弯中段，其他关节保持张开；不是提前夹紧球体。"""
        if self.mode=="limited":return
        c=self.config
        fraction=float(np.clip((now-(nominal-c.preshape_lead))/c.preshape_duration,0.,1.))
        angle=c.preshape_angle
        self.shape=self.hand.open_pose.copy();self.shape_velocity=np.zeros(16)
        self.shape[self.middle]=angle*fraction*fraction*(3.-2.*fraction)
        self.shape_velocity[self.middle]=angle*6.*fraction*(1.-fraction)/c.preshape_duration
        data.ctrl[self.hand.ids]=np.clip(self.shape,self.hand.limits[:,0],self.hand.limits[:,1])

    def observe(self,model,data,velocity=None):
        self.support.observe(model,data);self.loads[:]=0.
        for i,contact in enumerate(data.contact):
            if self.support.projectile not in (contact.geom1,contact.geom2) or contact.efc_address<0:continue
            other=contact.geom1 if contact.geom2==self.support.projectile else contact.geom2
            part=self.affected.get(other)
            if part is None:continue
            wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
            self.loads[part]+=max(0.,wrench[0])
        target=(self.loads>=.10).astype(float)
        # 10 ms进入柔顺，40 ms退出，避免接触开闭产生增益跳变。
        tau=np.where(target>self.blend,.010,.040)
        self.blend+=(target-self.blend)*(1.-np.exp(-model.opt.timestep/tau))

    def command(self,model,data,elapsed):
        c=self.config
        if self.close_start is None:
            self.close_start=self.shape[self.middle].copy()
            self.close_speed=self.shape_velocity[self.middle].copy()
        fraction=float(np.clip(elapsed/c.unloaded_duration,0,1))
        smooth=fraction*fraction*(3.-2.*fraction)
        self.shape=self.hand.open_pose+(self.hand.closed-self.hand.open_pose)*smooth
        self.shape_velocity=(self.hand.closed-self.hand.open_pose)*(6.*fraction*(1.-fraction)/c.unloaded_duration)
        self.shape_acceleration=(self.hand.closed-self.hand.open_pose)*(6.*(1.-2.*fraction)/c.unloaded_duration**2)
        if fraction>=1.:self.shape_acceleration[:]=0.
        for i,joint in enumerate(self.middle):
            displacement,speed,accel=trapezoid(elapsed,self.goals[i]-self.close_start[i],c.middle_speed,c.middle_accel,self.close_speed[i])
            self.shape[joint]=self.close_start[i]+displacement;self.shape_velocity[joint]=speed
            self.shape_acceleration[joint]=accel
        target=self.shape.copy()
        self.effective_stiffness=self.kp.copy()
        blend=np.zeros(4)
        damping=self.kd.copy()
        if self.mode=="compliant":
            soft=c.opposed_stiffness if self.support.holding else c.contact_stiffness
            # 只在捕获早期柔顺；120-240 ms平滑恢复保持刚度，避免长期过软而滑落。
            recovery=float(np.clip((elapsed-.12)/.12,0.,1.))
            phase=.5*(1.+np.cos(np.pi*recovery))
            blend=self.blend*phase
            self.effective_stiffness=self.kp+(soft-self.kp)*blend
            damping=self.kd*(1.+(c.damping_scale-1.)*blend)
        # 模型惯性/重力前馈+速度跟踪，避免原位置伺服滞后后再高速追赶而过冲。
        # 只补偿手指参考加速度；手臂耦合加速度误差留给实际反馈，不假定完全逆动力学。
        mujoco.mj_fullM(model,data,self.inertia)
        feedforward=self.inertia[np.ix_(self.dofs,self.all_dofs)]@self.shape_acceleration
        feedforward+=(data.qfrc_bias-data.qfrc_passive)[self.dofs]
        q=data.qpos[self.qadr];v=data.qvel[self.dofs]
        # 承载时减弱运动前馈，防止仍沿时间轨迹强行推进；恢复阶段再平滑接回保持。
        torque=(1.-blend)*feedforward+self.effective_stiffness*(self.shape[self.middle]-q)
        torque+=damping*((1.-blend)*self.shape_velocity[self.middle]-v)
        # 原位置执行器tau=kp*(u-q)-kd*v；反解u，仍受原ctrlrange和±0.95 Nm限制。
        target[self.middle]=q+(torque+self.kd*v)/self.kp
        data.ctrl[self.hand.ids]=np.clip(target,self.hand.limits[:,0],self.hand.limits[:,1])
