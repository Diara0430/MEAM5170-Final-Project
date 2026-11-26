"""球体包围：食指让出拇指通道、提前整形、碰撞约束IK与触球后轻度保持。

只使用已知球几何、雅可比、已解接触力和有力矩限制的真实执行器；无机器学习。
食指避碰及预整形有意改变捕获动作；晚接近组与避碰组前300 ms保持一致。
静态探针候选与实际动力学结果分别记录。
"""
from dataclasses import dataclass
import time

import mujoco
import numpy as np
import osqp
from scipy import sparse

from controllers.joint_aware_closure import JointAwareClosure


@dataclass(frozen=True)
class CageConfig:
    start_s: float=.30
    support_s: float=.04
    period_s: float=.02
    speed: float=2.             # rad/s，拇指运动参考。
    acceleration: float=20.     # rad/s²。
    max_step: float=.25         # 每次局部IK的关节变化范围。
    max_offset: float=.70       # 相对启动姿态的总参考偏移上限。
    approach_step: float=.004   # m，每次局部目标最多接近4 mm。
    contact_load: float=.10     # N，触球识别。
    desired_load: float=.30     # N，轻度承载，不把球大力推出。
    load_limit: float=1.0       # N，超过此值撤去额外法向目标。
    tracking_stiffness: float=12.
    contact_stiffness: float=6.
    transition_s: float=.08
    allowed_penetration: float=.0005
    preshape_lead: float=.50
    preshape_duration: float=.40
    index_duration: float=.29


class SphereCageClosure(JointAwareClosure):
    def __init__(self,model,hand,cage_mode="preshape",cage_config=None,plan_schedule=None):
        super().__init__(model,hand,"compliant")
        if cage_mode not in ("position","force","preshape","clearance"):raise ValueError("Unknown cage mode")
        self.cage_mode=cage_mode;self.cage_config=CageConfig() if cage_config is None else cage_config
        self.plan_schedule=plan_schedule # 可选：严格重放真实实验记录的晚期拇指规划超时决策。
        c=self.cage_config
        if not np.all(np.isfinite(list(vars(c).values()))) or min(vars(c).values())<=0:raise ValueError("Positive finite cage parameters required")
        self.thumb_qadr=[model.joint(f"hand/{i}").qposadr[0] for i in range(12,16)]
        self.thumb_dofs=[model.joint(f"hand/{i}").dofadr[0] for i in range(12,16)]
        self.thumb_ids=np.array(hand.ids[12:]);self.thumb_kp=model.actuator_gainprm[self.thumb_ids,0].copy()
        self.thumb_kd=-model.actuator_biasprm[self.thumb_ids,2].copy()
        joints=[model.joint(f"hand/{i}").id for i in range(12,16)]
        self.lo=np.maximum(model.jnt_range[joints,0],hand.limits[12:,0]);self.hi=np.minimum(model.jnt_range[joints,1],hand.limits[12:,1])
        self.thumb_geoms=[g for g,p in self.support.group.items() if p==3 and model.geom_contype[g]]
        self.probe=mujoco.MjData(model);self.started=None;self.support_time=0.;self.last_plan=-np.inf
        self.loaded_time=0.;self.unloaded_time=0.;self.thumb_load=0.;self.contact_torque=np.zeros(4)
        self.qref=self.vref=self.goal=self.anchor=None
        self.touch_anchor=None
        self.segment_start=None;self.segment_q=None;self.segment_target=None;self.segment_duration=.1
        self.prepose=np.array([.8,.5,1.,.65]);self.pre_start=None
        self.index_start=None;self.index_goal=np.array([0.,.8,.9,.5])
        self.index_adr=[model.joint(f"hand/{i}").qposadr[0] for i in range(4)]
        self.index_dofs=[model.joint(f"hand/{i}").dofadr[0] for i in range(4)]
        self.index_ids=np.array(hand.ids[:4]);self.index_kp=model.actuator_gainprm[self.index_ids,0]
        self.index_kd=-model.actuator_biasprm[self.index_ids,2]
        self.reference_inertia=np.empty((model.nv,model.nv));self.rows=[]

    def preshape_reference(self,now):
        duration=self.cage_config.preshape_duration
        s=float(np.clip((now-self.pre_start)/duration,0.,1.))
        h=s**3*(10.-15.*s+6.*s*s);hd=30.*s*s*(1.-s)**2/duration
        hdd=60.*s*(1.-s)*(1.-2.*s)/duration**2
        return h*self.prepose,hd*self.prepose,hdd*self.prepose

    def prepare(self,model,data,now,nominal):
        super().prepare(model,data,now,nominal)
        if self.cage_mode!="preshape":return
        if self.pre_start is None:self.pre_start=max(0.,nominal-self.cage_config.preshape_lead)
        qref,vref,acc=self.preshape_reference(now)
        self.shape[12:]=qref;self.shape_velocity[12:]=vref;self.shape_acceleration[12:]=acc
        mujoco.mj_fullM(model,data,self.reference_inertia)
        q=data.qpos[self.thumb_qadr];v=data.qvel[self.thumb_dofs]
        torque=self.reference_inertia[np.ix_(self.thumb_dofs,self.thumb_dofs)]@acc
        torque+=(data.qfrc_bias-data.qfrc_passive)[self.thumb_dofs]
        torque+=self.cage_config.tracking_stiffness*(qref-q)+self.thumb_kd*(vref-v)
        target=q+(torque+self.thumb_kd*v)/self.thumb_kp
        data.ctrl[self.thumb_ids]=np.clip(target,self.hand.limits[12:,0],self.hand.limits[12:,1])

    def observe(self,model,data,velocity=None):
        super().observe(model,data,velocity)
        self.support_time=self.support_time+model.opt.timestep if self.support.holding else 0.
        self.thumb_load=0.;self.contact_torque[:]=0.
        for i,contact in enumerate(data.contact):
            if self.support.projectile not in (contact.geom1,contact.geom2) or contact.efc_address<0:continue
            second=contact.geom2==self.support.projectile
            other=contact.geom1 if second else contact.geom2
            if other not in self.thumb_geoms:continue
            force=np.empty(6);mujoco.mj_contactForce(model,data,i,force)
            self.thumb_load+=max(0.,force[0])
            if force[0]<.02:continue
            normal=(1. if second else -1.)*contact.frame[:3]
            jac=np.empty((3,model.nv));mujoco.mj_jac(model,data,jac,None,contact.pos,model.geom_bodyid[other])
            # 每个已承载点按当前法向载荷分配轻度法向保持目标。
            self.contact_torque+=jac[:,self.thumb_dofs].T@normal*force[0]
        self.loaded_time=self.loaded_time+model.opt.timestep if self.thumb_load>=self.cage_config.contact_load else 0.
        self.unloaded_time=self.unloaded_time+model.opt.timestep if self.thumb_load<.02 else 0.
        if self.thumb_load>.02:self.contact_torque/=self.thumb_load

    def self_depths(self,model,data):
        depths={}
        for contact in data.contact:
            if contact.geom1 not in self.support.group or contact.geom2 not in self.support.group:continue
            if contact.geom1 not in self.thumb_geoms and contact.geom2 not in self.thumb_geoms:continue
            key=tuple(sorted((contact.geom1,contact.geom2)))
            depths[key]=min(depths.get(key,0.),float(contact.dist))
        return depths

    def plan(self,model,data):
        start=time.perf_counter();c=self.cage_config;probe=self.probe
        probe.qpos[:]=data.qpos;probe.qvel[:]=0.;mujoco.mj_forward(model,probe)
        measured=probe.qpos[self.thumb_qadr].copy();base_depth=self.self_depths(model,probe)
        constraints=[];limits=[]
        # 已有自接触的线性化距离约束允许关节协同退让，避免只增大拇指根部角度。
        for contact in probe.contact:
            key=tuple(sorted((contact.geom1,contact.geom2)))
            if key not in base_depth or contact.dist>.002:continue
            j1,j2=np.empty((3,model.nv)),np.empty((3,model.nv))
            mujoco.mj_jac(model,probe,j1,None,contact.pos,model.geom_bodyid[contact.geom1])
            mujoco.mj_jac(model,probe,j2,None,contact.pos,model.geom_bodyid[contact.geom2])
            constraints.append(contact.frame[:3]@(j2-j1)[:,self.thumb_dofs])
            limits.append(-c.allowed_penetration-contact.dist)
        low=np.maximum.reduce((self.lo-measured,np.full(4,-c.max_step),self.anchor-c.max_offset-measured))
        high=np.minimum.reduce((self.hi-measured,np.full(4,c.max_step),self.anchor+c.max_offset-measured))
        best=None;smallest_gap=.5;rejected=0
        for geom in self.thumb_geoms:
            probe.qpos[:]=data.qpos;mujoco.mj_kinematics(model,probe);mujoco.mj_comPos(model,probe)
            segment=np.empty(6)
            gap=mujoco.mj_geomDistance(model,probe,geom,self.support.projectile,.5,segment)
            smallest_gap=min(smallest_gap,float(gap))
            if gap<.0002:continue
            normal=(segment[3:]-segment[:3])/max(np.linalg.norm(segment[3:]-segment[:3]),1e-12)
            jac=np.empty((3,model.nv));mujoco.mj_jac(model,probe,jac,None,segment[:3],model.geom_bodyid[geom])
            j=jac[:,self.thumb_dofs];error=normal*min(gap+.0002,c.approach_step)
            matrix=j.T@j+.004**2*np.eye(4);linear=-j.T@error
            a=np.vstack([np.eye(4)]+([np.array(constraints)] if constraints else []))
            lo=np.r_[low,limits];hi=np.r_[high,np.full(len(limits),np.inf)]
            if np.any(lo>hi):continue
            solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(2.*matrix),q=2.*linear,A=sparse.csc_matrix(a),l=lo,u=hi,
                verbose=False,polishing=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=1500)
            result=solver.solve(raise_error=False)
            if result.info.status_val!=1:continue
            for scale in (1.,.5,.25):
                proposed=measured+scale*result.x;valid=True;end_gap=gap
                # 用真实网格碰撞管线复核整段离散路径；不把线性化无碰撞当作充分条件。
                for fraction in (.25,.5,.75,1.):
                    probe.qpos[:]=data.qpos;probe.qpos[self.thumb_qadr]=measured+fraction*(proposed-measured)
                    mujoco.mj_forward(model,probe)
                    depths=self.self_depths(model,probe)
                    if any(depth<min(-c.allowed_penetration,base_depth.get(key,0.)-.0002) for key,depth in depths.items()):
                        valid=False;break
                if not valid:rejected+=1;continue
                end_gap=mujoco.mj_geomDistance(model,probe,geom,self.support.projectile,.5,None)
                improvement=gap-end_gap
                if improvement<.00005:continue
                score=end_gap+.0005*float(np.linalg.norm(proposed-measured))
                if best is None or score<best[0]:best=(score,proposed.copy())
                break
        wall=time.perf_counter()-start
        step=round(data.time/model.opt.timestep);measured_overrun=wall>c.period_s
        if self.plan_schedule is not None and step not in self.plan_schedule:raise ValueError("Missing recorded thumb-planner decision")
        overrun=measured_overrun if self.plan_schedule is None else self.plan_schedule[step]
        info=dict(plan_called=1,plan_failed=int(best is None),plan_overrun=int(overrun),plan_step=step,
            plan_measured_overrun=int(measured_overrun),plan_decision_source="measured" if self.plan_schedule is None else "recorded",
            plan_ms=1000.*wall,gap_mm=1000.*smallest_gap,collision_rejections=rejected)
        return None if best is None or overrun else best[1],info

    def command(self,model,data,elapsed):
        if self.index_start is None:self.index_start=self.shape[:4].copy()
        super().command(model,data,elapsed);c=self.cage_config
        # 食指原本几乎不承载，却与拇指指尖相撞；缩短其闭合行程并统一减速。
        # 该姿态经两种拇指终点下的网格路径筛查，保持至少约3.8 mm静态间隙。
        s=float(np.clip(elapsed/c.index_duration,0.,1.))
        h=s**3*(10.-15.*s+6.*s*s);hd=30.*s*s*(1.-s)**2/c.index_duration
        hdd=60.*s*(1.-s)*(1.-2.*s)/c.index_duration**2
        delta=self.index_goal-self.index_start
        index_q=self.index_start+h*delta;index_v=hd*delta;index_a=hdd*delta
        mujoco.mj_fullM(model,data,self.reference_inertia)
        q_index=data.qpos[self.index_adr];v_index=data.qvel[self.index_dofs]
        tau_index=self.reference_inertia[np.ix_(self.index_dofs,self.index_dofs)]@index_a
        tau_index+=(data.qfrc_bias-data.qfrc_passive)[self.index_dofs]
        tau_index+=self.index_kp*(index_q-q_index)+self.index_kd*(index_v-v_index)
        u_index=q_index+(tau_index+self.index_kd*v_index)/self.index_kp
        data.ctrl[self.index_ids]=np.clip(u_index,self.hand.limits[:4,0],self.hand.limits[:4,1])
        if self.cage_mode=="clearance":return
        early_preshape=self.cage_mode=="preshape" and elapsed<c.start_s
        if elapsed<c.start_s and not early_preshape:return
        q=data.qpos[self.thumb_qadr];v=data.qvel[self.thumb_dofs]
        if self.started is None:
            if not early_preshape and self.support_time<c.support_s:return
            self.started=elapsed;self.qref=q.copy();self.vref=np.zeros(4);self.goal=q.copy();self.anchor=q.copy()
        info=dict(plan_called=0,plan_failed=0,plan_overrun=0,plan_step=-1,plan_measured_overrun=0,
            plan_decision_source="none",plan_ms=0.,gap_mm=0.,collision_rejections=0)
        loaded=self.loaded_time>=.006
        segment_done=self.segment_start is None or elapsed>=self.segment_start+self.segment_duration-1e-10
        if not early_preshape and not loaded and self.unloaded_time>=.02 and segment_done and elapsed-self.last_plan>=c.period_s-1e-10:
            goal,info=self.plan(model,data);self.last_plan=elapsed
            if goal is not None:
                self.goal=goal;self.segment_q=q.copy();self.segment_target=goal.copy();self.segment_start=elapsed
                distance=float(max(abs(goal-q)))
                self.segment_duration=max(.10,1.875*distance/c.speed,np.sqrt(5.774*distance/c.acceleration))
        dt=model.opt.timestep
        if loaded:
            # 触球后以真实姿态为平衡点，撤去推进速度，避免残余运动参考继续挤球。
            if self.touch_anchor is None:self.touch_anchor=q.copy()
            self.qref=self.touch_anchor.copy();self.vref[:]=0.;acc=np.zeros(4)
        else:
            if self.unloaded_time>=.02:self.touch_anchor=None
            # 四关节共用同一段五次进度，保持已检查的协同路径，避免各自追赶造成自碰。
            if early_preshape:
                self.qref,self.vref,acc=self.preshape_reference(data.time)
            elif self.segment_start is not None:
                s=float(np.clip((elapsed-self.segment_start)/self.segment_duration,0.,1.))
                h=s**3*(10.-15.*s+6.*s*s);hd=30.*s*s*(1.-s)**2/self.segment_duration
                hdd=60.*s*(1.-s)*(1.-2.*s)/self.segment_duration**2
                delta=self.segment_target-self.segment_q
                self.qref=self.segment_q+h*delta;self.vref=hd*delta;acc=hdd*delta
            else:self.vref[:]=0.;acc=np.zeros(4)
        mujoco.mj_fullM(model,data,self.reference_inertia)
        feedforward=self.reference_inertia[np.ix_(self.thumb_dofs,self.thumb_dofs)]@acc
        feedforward+=(data.qfrc_bias-data.qfrc_passive)[self.thumb_dofs]
        stiffness=c.contact_stiffness if loaded else c.tracking_stiffness
        torque=feedforward+stiffness*(self.qref-q)+self.thumb_kd*(self.vref-v)
        if loaded and self.cage_mode in ("force","preshape"):
            force_target=0. if self.thumb_load>c.load_limit else c.desired_load
            torque+=self.contact_torque*force_target
        # 过渡的是力矩，不是跳变的关节位置；实际驱动仍经原位置执行器和±0.95 Nm限位。
        old=self.thumb_kp*(data.ctrl[self.thumb_ids]-q)-self.thumb_kd*v
        fraction=float(np.clip((elapsed-self.started)/c.transition_s,0.,1.));blend=fraction*fraction*(3.-2.*fraction)
        if self.cage_mode=="preshape":blend=1. # 接触前已运行同一参考跟踪器，无需重新混入原拇指闭合目标。
        torque=(1.-blend)*old+blend*torque
        target=q+(torque+self.thumb_kd*v)/self.thumb_kp
        data.ctrl[self.thumb_ids]=np.clip(target,self.hand.limits[12:,0],self.hand.limits[12:,1])
        self.rows.append(dict(elapsed_s=elapsed,loaded=int(loaded),thumb_load_N=self.thumb_load,
            reference_speed=float(max(abs(self.vref))),reference_accel=float(max(abs(acc))),
            reference_offset=float(max(abs(self.qref-self.anchor))),tracking_error=float(max(abs(self.qref-q))),
            **{f"actual_q_{i+12}":float(value) for i,value in enumerate(q)},
            **{f"reference_q_{i+12}":float(value) for i,value in enumerate(self.qref)},**info))
