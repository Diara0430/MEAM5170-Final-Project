"""Locate first contact geometrically before spending the rollout budget on impact."""
import time

import mujoco
import numpy as np

from controllers.intercept_impact import Impact, InterceptImpact


class ShortImpact(InterceptImpact):
    def __init__(self, model, observer, *, shadow=False):
        super().__init__(model, observer, method="preview")
        self.shadow = shadow

    def choose(self, data, now, first, second, deadline):
        selected = super().choose(data,now,first,second,deadline)
        return first if self.shadow else selected

    def _state(self, plan, stamp):
        d = self.probe
        q, v, _ = plan.reference(stamp)
        d.time = stamp
        d.qpos[self.qadr], d.qvel[self.dofs] = q, v
        pos, vel = self.observer.estimate.at(stamp)
        d.joint("flight").qpos[:] = [*pos,1.,0.,0.,0.]
        d.joint("flight").qvel[:] = [*vel,0.,0.,0.]
        pose = getattr(self.observer, "preview_rotation", None)
        if pose is not None:
            pose.write(d, stamp)
        return q, v

    def _hand_sign(self, c):
        if c.geom2==self.ball and c.geom1 in self.hand:
            return 1
        if c.geom1==self.ball and c.geom2 in self.hand:
            return -1
        return 0

    def preview(self, plan, deadline):
        m, d = self.model, self.probe
        dt, stamp = m.opt.timestep, plan.intercept_time
        for k in range(-12,19,2):
            if time.perf_counter() >= deadline:
                return None
            when = stamp + k*dt
            self._state(plan,when)
            mujoco.mj_fwdPosition(m,d)
            if any(self._hand_sign(c) and c.dist<=0. for c in d.contact):
                break
        else:
            return None
        start = when-2*dt
        q, v = self._state(plan,start)
        d.qacc_warmstart[:] = 0.
        mujoco.mj_forward(m,d)
        # Keep the local reference velocity and bias torque fixed over this short
        # forecast. The actual plant retains its original 2 ms feedback controller.
        tr = self.tracker
        torque = d.qfrc_bias[self.dofs] - d.qfrc_passive[self.dofs]
        target = q + (torque + tr.kd*v)/tr.kp
        limits = m.actuator_ctrlrange[tr.ids]
        flight = d.joint("flight")
        onset = direction = None
        peak_a = peak_f = impulse = 0.
        wrench = np.empty(6)
        for _ in range(round(.036/dt)):
            if time.perf_counter() >= deadline:
                return None
            t = float(d.time)
            d.ctrl[tr.ids] = np.clip(target+(t-start)*v,limits[:,0],limits[:,1])
            before = d.qvel[self.dofs].copy()
            ball_before = flight.qvel[:3].copy()
            mujoco.mj_step(m,d)
            force = np.zeros(3)
            touched = False
            for index,c in enumerate(d.contact):
                sign = self._hand_sign(c)
                if sign and c.efc_address>=0:
                    touched = True
                    mujoco.mj_contactForce(m,d,index,wrench)
                    force += sign*(c.frame.reshape(3,3).T@wrench[:3])
            if touched and onset is None:
                onset = t
                direction = ball_before/max(np.linalg.norm(ball_before),1e-9)
            if onset is not None:
                peak_a = max(peak_a,np.max(np.abs(d.qvel[self.dofs]-before))/dt)
                peak_f = max(peak_f,np.linalg.norm(force))
                impulse -= (force@direction)*dt
                if d.time >= onset+.020-1e-8:
                    return Impact(float(peak_a),float(peak_f),float(impulse))
        return None
