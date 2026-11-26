"""Compare arm torque filters under a measured two-millisecond command deadline."""
import time

import mujoco
import numpy as np

from controllers.arm_reference_tracker import ArmReferenceTracker
from controllers.contact_preview import ContactPreviewFilter


class DeadlineArmFilter(ContactPreviewFilter):
    def __init__(self, model, observer, enabled, fast=False, refine=False):
        super().__init__(model, observer, enabled)
        self.fast = fast
        self.refine = refine
        self.refinements = self.improvements = 0
        self.deadline_misses = self.command_count = 0
        self.last_missed = False
        self.onset_missed = None
        self.mass = np.empty((model.nv, model.nv))
        self.input_axes = np.eye(model.nv)[:, self.dofs]

    def command(self, data, plan, t):
        start = time.perf_counter()
        self.command_start = start
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
        if self.refine and error > 1e-4:
            self.refinements += 1
            old_error = error
            # Preserve the best already verified torque while time remains. A change
            # of friction regime can invalidate the inertia-only approximation.
            deadline = self.command_start + .80 * dt
            if self.last_torque is not None and time.perf_counter() < deadline:
                previous = np.clip(self.last_torque, lo, hi)
                prev_acc, prev_error = predicted(previous)
                if prev_error < error:
                    torque, acceleration, error = previous, prev_acc, prev_error
            if error > 1e-4 and time.perf_counter() < deadline:
                jac = np.zeros((7, 7))
                for joint in range(7):
                    if time.perf_counter() >= deadline:
                        break
                    delta = min(c.difference_torque, hi[joint]-torque[joint])
                    if delta < 1e-6:
                        delta = -min(c.difference_torque, torque[joint]-lo[joint])
                    if abs(delta) > 1e-9:
                        shifted = torque.copy()
                        shifted[joint] += delta
                        jac[:, joint] = (predicted(shifted)[0] - acceleration) / delta
                else:
                    if time.perf_counter() < deadline:
                        target = np.clip(np.clip(acceleration, -3., 3.), low, high) - acceleration
                        correction = np.linalg.lstsq(jac, target, rcond=1e-4)[0]
                        candidate = np.clip(torque+correction, lo, hi)
                        candidate_acc, candidate_error = predicted(candidate)
                        if candidate_error < error:
                            torque, acceleration, error = candidate, candidate_acc, candidate_error
            self.improvements += int(error < old_error)
        self.last_torque = torque.copy()
        data.ctrl[self.ids] = q+(torque+self.kd*v)/self.kp
