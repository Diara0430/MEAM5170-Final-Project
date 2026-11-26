"""Before first contact, test axial wrist torque against an early and a late arrival."""
import time

import mujoco
import numpy as np

from controllers.deadline_arm_filter import DeadlineArmFilter


class WristOnsetFilter(DeadlineArmFilter):
    def __init__(self, *args, shadow=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shadow = shadow
        self.scenes = [mujoco.MjData(self.model) for _ in range(2)]
        self.roll = mujoco.MjData(self.model)
        self.attempts = self.proposals = self.deliveries = self.timeouts = 0
        self.pending_change = False

    def command(self, data, plan, t):
        self.pending_change = False
        super().command(data, plan, t)
        if self.pending_change and not self.last_missed:
            self.deliveries += 1

    def apply_filter(self, data, plan, t):
        super().apply_filter(data, plan, t)
        if self.touched or data is not self.robot_with_estimate:
            return
        m = self.model
        dt = m.opt.timestep
        deadline = self.command_start + .86*dt
        if time.perf_counter() >= deadline:
            return
        p, v = self.observer.estimate.at(t)
        speed = np.linalg.norm(v)
        if speed < .1:
            return
        palm = data.body(self.palm)
        center = palm.xpos + palm.xmat.reshape(3, 3) @ self.offset
        if np.linalg.norm(p-center) > .065:
            return
        self.attempts += 1
        spread = np.clip(1.5*self.observer.noise_m, .003, .007)
        for scene, sign in zip(self.scenes, (-1., 1.)):
            mujoco.mj_copyData(scene, m, data)
            scene.joint("flight").qpos[:3] = p + sign*spread*v/speed
            mujoco.mj_forward(m, scene)
        joint = 5
        j = self.ids[joint]
        velocity = data.qvel[self.dofs].copy()
        q = data.qpos[self.qadr].copy()
        nominal = data.ctrl[j]
        torque = self.kp[joint]*(nominal-q[joint])-self.kd[joint]*velocity[joint]
        lo, hi = m.actuator_forcerange[j] - torque
        slo, shi = self.kp[joint]*(m.actuator_ctrlrange[j]-nominal)
        lo, hi = max(lo, slo, -.8), min(hi, shi, .8)
        if lo >= hi:
            return

        def predict(delta):
            values = []
            for scene in self.scenes:
                if time.perf_counter() >= deadline:
                    return None
                mujoco.mj_copyData(self.roll, m, scene)
                self.roll.ctrl[j] = nominal + delta/self.kp[joint]
                mujoco.mj_step(m, self.roll)
                values.append((self.roll.qvel[self.dofs]-velocity)/dt)
            return np.array(values)

        baseline = predict(0.)
        if baseline is None:
            self.timeouts += 1
            return
        axis = baseline[:, joint]
        worst = np.max(abs(axis))
        if worst <= self.config.acceleration:
            return
        probe = float(np.clip(.05 if hi>=.05 else -.05, lo, hi))
        if abs(probe) < 1e-6:
            return
        test = predict(probe)
        if test is None:
            self.timeouts += 1
            return
        slope = (test[:,joint]-axis)/probe
        if np.min(slope) <= 1e-4:
            return
        # Two affine responses with positive slopes are minimax when opposite in sign.
        delta = -axis.sum()/slope.sum()
        # Never intentionally move a previously compliant scenario across the limit.
        bound = np.maximum(abs(axis), self.config.acceleration)
        lo = max(lo, np.max((-bound-axis)/slope))
        hi = min(hi, np.min((bound-axis)/slope))
        if hi < lo:
            return
        delta = float(np.clip(delta, lo, hi))
        if abs(delta) < .001:
            return
        candidate = predict(delta)
        if candidate is None or time.perf_counter() >= deadline:
            self.timeouts += 1
            return
        speed_after = velocity + dt*candidate
        q_after = q + dt*speed_after
        if (np.max(abs(candidate[:,joint])) > worst-.10 or
            np.any(abs(candidate) > np.maximum(abs(baseline), self.config.acceleration)+1e-5) or
            np.any(abs(speed_after) > self.config.speed) or
            np.any(q_after < self.lower) or np.any(q_after > self.upper)):
            return
        self.proposals += 1
        if not self.shadow:
            data.ctrl[j] = nominal + delta/self.kp[joint]
            self.last_torque[joint] += delta
            self.pending_change = True

    def summary(self):
        return dict(wrist_attempts=self.attempts, wrist_proposals=self.proposals,
                    wrist_deliveries=self.deliveries, wrist_timeouts=self.timeouts)


class WristLeverAim:
    def __init__(self, model, observer, *, shadow=False):
        self.model = model
        self.probe = mujoco.MjData(model)
        self.palm = model.body("hand/palm_lower").id
        self.joint = model.joint("joint6").id
        self.arm = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]
        self.shadow = shadow
        self.aim = None
        self.decided = False
        self.deliveries = 0

    def contact_offset(self, data, previous, now):
        stamp = previous.intercept_time
        if not self.decided and .18 <= stamp-now <= .55:
            self.decided = True
            d = self.probe
            mujoco.mj_copyData(d, self.model, data)
            d.qpos[self.arm] = previous.reference(stamp)[0]
            mujoco.mj_fwdPosition(self.model, d)
            palm = d.body(self.palm)
            rotation = palm.xmat.reshape(3, 3)
            center = palm.xpos + rotation @ np.array([0., -.055, -.08])
            gradient = np.cross(rotation[:,2], d.xaxis[self.joint])
            norm = np.linalg.norm(gradient)
            lever = (center-d.xanchor[self.joint]) @ gradient
            if norm > .1 and abs(lever) > .003:
                self.aim = min(.004, abs(lever)/norm) * np.sign(lever)*gradient/norm
        return None if self.shadow else self.aim

    def choose(self, data, now, first, second, deadline):
        return first

    def applied(self, plan):
        self.deliveries += int(getattr(plan,"impact_offset",False))

    def summary(self):
        return dict(lever_chosen=int(self.aim is not None), lever_deliveries=self.deliveries)
