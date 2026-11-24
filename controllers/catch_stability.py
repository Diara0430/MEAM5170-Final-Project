"""速度相关的顺势缓冲与有承载限制的小幅防滑反馈。未使用机器学习。"""
import mujoco
import numpy as np

from controllers.contact_closure import ContactClosure


class ArmBuffer:
    """用七轴机械臂实现手掌平移，保持接触阶段的手掌朝向。

    固定到达时刻 0.20 s 的本轮实验，提前 40 ms 启动。
    这是假设到达时间已知的缓冲实验，后续由在线到达预测器提供启动时刻。
    """
    def __init__(self, distance_scale=1.):
        if not np.isfinite(distance_scale) or not 0 < distance_scale <= 1:
            raise ValueError("Buffer distance scale must be within (0, 1]")
        self.distance_scale = distance_scale
        self.start = .160
        self.duration = .200
        self.ik = None
        self.infeasible = False
        self.offset = np.zeros(7)
        self.planned_mm = 0.
        self.actual_mm = 0.
        self.max_command_speed_rad_s = 0.

    def command(self, model, data, t, tilt):
        if t < self.start:
            return
        ids = [model.actuator(f"act{i}").id for i in range(1, 8)]
        joints = [model.joint(f"joint{i}") for i in range(1, 8)]
        qadr = [j.qposadr[0] for j in joints]
        dofs = [j.dofadr[0] for j in joints]
        palm = data.body("hand/palm_lower")
        if self.ik is None:
            self.ik = mujoco.MjData(model)
            self.ik.qpos[:] = data.qpos
            self.base_q = data.qpos[qadr].copy()
            self.base_ctrl = data.ctrl[ids].copy()
            self.origin = palm.xpos.copy()
            self.rotation = palm.xmat.reshape(3, 3).copy()
            velocity = data.joint("flight").qvel[:3].copy()
            speed = np.linalg.norm(velocity)
            # 峰值后移速度为来速的 15%，上限 0.15 m/s；总行程不超过 20 mm。
            peak = min(.15*speed, .15)
            self.distance = self.distance_scale*min(.020, peak*self.duration/1.875)
            self.direction = velocity/max(speed, 1e-12)
            self.planned_mm = 1000*self.distance
        if t < tilt:
            self.actual_mm = 1000*np.dot(palm.xpos-self.origin, self.direction)
        if t < tilt and not self.infeasible:
            # 五次平滑曲线使缓冲起止速度、加速度均为零，最后保持后移位置。
            u = float(np.clip((t-self.start)/self.duration, 0., 1.))
            goal = self.origin + self.direction*self.distance*(10*u**3-15*u**4+6*u**5)
            mujoco.mj_forward(model, self.ik)
            ik_palm = self.ik.body("hand/palm_lower")
            angular = .5*np.cross(ik_palm.xmat.reshape(3, 3).T, self.rotation.T).sum(axis=0)
            error = np.r_[goal-ik_palm.xpos, angular]
            jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
            mujoco.mj_jacBody(model, self.ik, jp, jr, ik_palm.id)
            jac = np.vstack((jp[:, dofs], jr[:, dofs]))
            delta = jac.T@np.linalg.solve(jac@jac.T+1e-4*np.eye(6), error)
            # 命令关节速度上限为 1 rad/s；这也是用于本实验的保守设定。
            delta = np.clip(delta, -model.opt.timestep, model.opt.timestep)
            proposed = self.ik.qpos[qadr]+delta
            joint_limits = model.jnt_range[[j.id for j in joints]]
            ctrl_limits = model.actuator_ctrlrange[ids]
            lower = np.maximum(joint_limits[:, 0], ctrl_limits[:, 0])
            upper = np.minimum(joint_limits[:, 1], ctrl_limits[:, 1])
            self.infeasible = bool(np.any(proposed < lower) or np.any(proposed > upper))
            if not self.infeasible:
                self.ik.qpos[qadr] = proposed
                self.offset = proposed-self.base_q
                self.max_command_speed_rad_s = max(self.max_command_speed_rad_s,
                    float(np.max(np.abs(delta))/model.opt.timestep))
        # 缓冲结束保留七轴偏移，之后原有腕部倾转在此基础上进行。
        command = self.base_ctrl.copy()
        command[5] = data.ctrl[ids[5]]
        command += self.offset
        limits = model.actuator_ctrlrange[ids]
        self.infeasible |= bool(np.any(command < limits[:, 0]) or np.any(command > limits[:, 1]))
        data.ctrl[ids] = np.clip(command, limits[:, 0], limits[:, 1])


class SlipGrip:
    """漂移持续增长且存在对向支撑时，小幅增加承载手指屈曲目标。"""
    def __init__(self):
        self.contact = None
        self.reference = None
        self.previous = 0.
        self.speed = 0.
        self.risk_time = 0.
        self.extra = np.zeros(4)
        self.active_steps = 0

    def __call__(self, model, data, hand, t, tilt):
        if self.contact is None:
            self.contact = ContactClosure(model, hand)
        self.contact.observe(model, data)
        if t < tilt:
            return
        dt = model.opt.timestep
        palm = data.body("hand/palm_lower")
        relative = palm.xmat.reshape(3, 3).T@(data.body("projectile").xpos-palm.xpos)
        if self.reference is None:
            self.reference = relative.copy()
        radius = np.linalg.norm(relative-self.reference)
        self.speed += dt/(.020+dt)*((radius-self.previous)/dt-self.speed)
        self.previous = radius
        risk = radius > .003 and self.speed > .005 and self.contact.holding
        self.risk_time = self.risk_time+dt if risk else 0.
        loads = np.zeros(4)
        for i, c in enumerate(data.contact):
            if self.contact.projectile not in (c.geom1, c.geom2) or c.efc_address < 0:
                continue
            other = c.geom2 if c.geom1 == self.contact.projectile else c.geom1
            group = self.contact.group.get(other, -1)
            if group < 0:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            loads[group] += max(0., wrench[0])
        if self.risk_time >= .010:
            self.active_steps += 1
            # 只调已有接触的手指。法向力超过 12 N 时停止增加目标。
            mask = (loads >= .02) & (loads < 12.)
            self.extra[mask] = np.minimum(.12, self.extra[mask]+.4*dt)
        for group in range(4):
            flex = slice(4*group+1, 4*group+4)
            ids = hand.ids[flex]
            data.ctrl[ids] = np.clip(data.ctrl[ids]+self.extra[group],
                                     hand.limits[flex, 0], hand.limits[flex, 1])
