"""Test one-axis contact uncertainty compensation inside the existing 2 ms deadline."""
import time

import mujoco
import numpy as np

from controllers.deadline_arm_filter import DeadlineArmFilter


class WristImpactFilter(DeadlineArmFilter):
    def __init__(self, model, observer, enabled):
        super().__init__(model, observer, enabled, fast=True, refine=True)
        self.wrist = 5  # xArm joint 6 in the seven arm actuator channels.
        self.scored_steps = self.changed_steps = 0

    def apply_filter(self, data, plan, t):
        super().apply_filter(data, plan, t)
        if data is not self.robot_with_estimate or self.touched:
            return
        model = self.model
        dt = model.opt.timestep
        deadline = self.command_start + .90*dt
        if time.perf_counter() > self.command_start + .60*dt:
            return
        idx = self.ids[self.wrist]
        q, v = data.qpos[self.qadr], data.qvel[self.dofs]
        nominal = data.ctrl[idx]
        kp, kd = self.kp[self.wrist], self.kd[self.wrist]
        lo, hi = model.actuator_forcerange[idx]
        limits = model.actuator_ctrlrange[idx]
        torque = kp*(nominal-q[self.wrist])-kd*v[self.wrist]
        lo, hi = max(lo-torque, kp*(limits[0]-nominal)), min(hi-torque, kp*(limits[1]-nominal))
        phases = (0., -.001, .001)

        def predict(shift, delta):
            probe = self.probe
            mujoco.mj_copyData(probe, model, data)
            flight = probe.joint("flight")
            flight.qpos[:3] += shift*flight.qvel[:3] + .5*shift**2*model.opt.gravity
            flight.qvel[:3] += shift*model.opt.gravity
            probe.ctrl[idx] = nominal + delta/kp
            mujoco.mj_step(model, probe)
            return (probe.qvel[self.dofs] - v)/dt

        initial = np.array([predict(shift, 0.) for shift in phases])
        wrist = initial[:,self.wrist]
        worst = float(np.max(np.abs(wrist)))
        if worst <= 3.8 or time.perf_counter() >= deadline:
            return
        self.scored_steps += 1
        perturb = min(.08, hi)
        if perturb < 1e-5:
            perturb = max(-.08, lo)
        if abs(perturb) < 1e-5:
            return
        sensitivity = (predict(0., perturb)[self.wrist]-wrist[0])/perturb
        if abs(sensitivity) < 1e-5:
            return
        delta = float(np.clip(-.5*(max(wrist)+min(wrist))/sensitivity, max(lo,-1.), min(hi,1.)))
        for fraction in (1., .45):
            if time.perf_counter() >= deadline:
                break
            shift_torque = fraction*delta
            checked = np.array([predict(shift, shift_torque) for shift in phases])
            if (np.max(np.abs(checked[:,self.wrist])) < worst-.05
                and np.max(np.abs(checked[0])) <= max(3.8, np.max(np.abs(initial[0])))
                and np.max(np.abs(checked)) <= max(3.8, np.max(np.abs(initial)))
                and np.all(np.abs(v + dt*checked) <= np.maximum(1., np.abs(v+dt*initial))+1e-8)):
                data.ctrl[idx] = nominal + shift_torque/kp
                self.last_torque[self.wrist] += shift_torque
                self.changed_steps += 1
                break
