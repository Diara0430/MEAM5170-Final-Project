"""Bounded positive wrist acceleration shortly before the measured negative onset shock."""
import time

import mujoco
import numpy as np

from controllers.deadline_arm_filter import DeadlineArmFilter


class WristPreload(DeadlineArmFilter):
    def __init__(self, *args, shadow=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shadow = shadow
        self.free = mujoco.MjData(self.model)
        self.check = mujoco.MjData(self.model)
        self.projectile = self.model.geom("projectile").id
        self.contact_geoms = [
            i for i, b in enumerate(self.model.geom_bodyid)
            if self.model.geom_contype[i] and
            self.model.body(b).name in ("hand/palm_lower","hand/mcp_joint","hand/mcp_joint_2","hand/mcp_joint_3")]
        self.armed = self.proposed = self.applied = self.late = 0
        self.changed = False

    def command(self, data, plan, t):
        self.changed = False
        super().command(data, plan, t)
        self.applied += int(self.changed and not self.last_missed)

    def apply_filter(self, data, plan, t):
        super().apply_filter(data, plan, t)
        if self.touched or data is not self.robot_with_estimate:
            return
        m = self.model
        dt = m.opt.timestep
        deadline = self.command_start + .88*dt
        if time.perf_counter() >= deadline:
            return
        # Contact distance is computed on the DELAYED estimate in this private view.
        if not any(mujoco.mj_geomDistance(m,data,self.projectile,g,.030,None) < .027
                   for g in self.contact_geoms):
            return
        self.armed += 1
        mujoco.mj_copyData(self.free, m, data)
        self.free.joint("flight").qpos[:3] = [4., 4., 4.]
        self.free.joint("flight").qvel[:] = 0.
        mujoco.mj_forward(m, self.free)
        wrist = 5
        j = self.ids[wrist]
        v = data.qvel[self.dofs]
        q = data.qpos[self.qadr]
        nominal = data.ctrl[j]
        torque = self.kp[wrist]*(nominal-q[wrist])-self.kd[wrist]*v[wrist]
        upper = min(.7, m.actuator_forcerange[j,1]-torque,
                    self.kp[wrist]*(m.actuator_ctrlrange[j,1]-nominal))
        if upper < .01:
            return

        def predict(scene, delta):
            mujoco.mj_copyData(self.check, m, scene)
            self.check.ctrl[j] = nominal + delta/self.kp[wrist]
            mujoco.mj_step(m, self.check)
            return (self.check.qvel[self.dofs]-v)/dt

        free = predict(self.free, 0.)
        target = min(2.8, free[wrist]+2.)
        if target <= free[wrist]+.2:
            return
        step = min(.05, upper)
        shifted = predict(self.free, step)
        response = (shifted[wrist]-free[wrist])/step
        if response <= 1e-4:
            return
        delta = min(upper, (target-free[wrist])/response)
        estimated = predict(data, 0.)
        for scale in (1., .5):
            if time.perf_counter() >= deadline:
                self.late += 1
                return
            move = scale*delta
            free_new, estimated_new = predict(self.free, move), predict(data, move)
            if time.perf_counter() >= deadline:
                self.late += 1
                return
            if np.any(abs(free_new)>self.config.acceleration):
                continue
            if np.any(abs(estimated_new)>np.maximum(abs(estimated),self.config.acceleration)+1e-5):
                continue
            speeds = v + dt*np.array([free_new, estimated_new])
            positions = q + dt*speeds
            if (np.any(abs(speeds)>self.config.speed) or np.any(positions<self.lower)
                    or np.any(positions>self.upper)):
                continue
            self.proposed += 1
            if not self.shadow:
                data.ctrl[j] = nominal + move/self.kp[wrist]
                self.last_torque[wrist] += move
                self.changed = True
            return

    def summary(self):
        return dict(preload_armed=self.armed, preload_proposed=self.proposed,
                    preload_applied=self.applied, preload_late=self.late)
