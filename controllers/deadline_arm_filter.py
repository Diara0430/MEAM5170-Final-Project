"""Compare arm torque filters under a measured two-millisecond command deadline."""
import time

import mujoco
import numpy as np

from controllers.arm_reference_tracker import ArmReferenceTracker
from controllers.contact_preview import ContactPreviewFilter


class DeadlineArmFilter(ContactPreviewFilter):
    def __init__(self, model, observer, enabled, fast=False):
        super().__init__(model, observer, enabled)
        self.fast = fast
        self.deadline_misses = self.command_count = 0
        self.last_missed = False
        self.onset_missed = None
        self.mass = np.empty((model.nv, model.nv))
        self.input_axes = np.eye(model.nv)[:, self.dofs]

    def command(self, data, plan, t):
        start = time.perf_counter()
        last = data.ctrl[self.ids].copy()
        super().command(data, plan, t)
        self.last_missed = time.perf_counter()-start > self.model.opt.timestep
        self.command_count += 1
        if self.last_missed:
            # Withhold the late position-actuator command. The physical simulation
            # still integrates exactly one step using the previous setpoint.
            self.deadline_misses += 1
            data.ctrl[self.ids] = last
            actual = self.kp*(last-data.qpos[self.qadr]) - self.kd*data.qvel[self.dofs]
            limits = self.model.actuator_forcerange[self.ids]
            self.last_torque = np.clip(actual, limits[:, 0], limits[:, 1])

    def apply_filter(self, data, plan, t):
        if not self.fast:
            return super().apply_filter(data, plan, t)
        model = self.model
        dt = model.opt.timestep
        c = self.config
        ArmReferenceTracker.command(self, data, plan, t)
        q, v = data.qpos[self.qadr], data.qvel[self.dofs]
        ctrl = model.actuator_ctrlrange[self.ids]
        force = model.actuator_forcerange[self.ids]
        lo = np.maximum(force[:, 0], self.kp*(ctrl[:, 0]-q)-self.kd*v)
        hi = np.minimum(force[:, 1], self.kp*(ctrl[:, 1]-q)-self.kd*v)
        torque = np.clip(self.kp*(data.ctrl[self.ids]-q)-self.kd*v, lo, hi)
        low = np.maximum.reduce([np.full(7, -c.acceleration), (-c.speed-v)/dt,
                                 (self.lower-q-v*dt)/dt**2])
        high = np.minimum.reduce([np.full(7, c.acceleration), (c.speed-v)/dt,
                                  (self.upper-q-v*dt)/dt**2])

        def predicted(candidate):
            mujoco.mj_copyData(self.probe, model, data)
            self.probe.ctrl[self.ids] = q+(candidate+self.kd*v)/self.kp
            mujoco.mj_step(model, self.probe)
            acceleration = (self.probe.qvel[self.dofs]-v)/dt
            error = max(0., float(np.max(low-acceleration)), float(np.max(acceleration-high)))
            return acceleration, error

        acceleration, error = predicted(torque)
        if error > 1e-4:
            # Approximate torque sensitivity using articulated inertia. Each candidate
            # must improve the full one-step simulation before it replaces the nominal.
            mujoco.mj_fullM(model, data, self.mass)
            self.mass.flat[::model.nv+1] += dt*model.dof_damping
            self.mass[self.dofs, self.dofs] += dt*self.kd
            response = np.linalg.solve(self.mass, self.input_axes)[self.dofs]
            for _ in range(3):
                desired = np.clip(np.clip(acceleration, -3., 3.), low, high)
                correction = np.linalg.solve(response, desired-acceleration)
                candidate = np.clip(torque+correction, lo, hi)
                new_acc, new_error = predicted(candidate)
                if new_error >= error:
                    break
                torque, acceleration, error = candidate, new_acc, new_error
                if error <= 1e-4:
                    break
        self.last_torque = torque.copy()
        data.ctrl[self.ids] = q+(torque+self.kd*v)/self.kp
