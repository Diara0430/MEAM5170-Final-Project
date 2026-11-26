"""Compare the two best feasible rolling intercepts using only delivered observations."""
from dataclasses import dataclass
import time

import mujoco
import numpy as np

from controllers.arm_reference_tracker import ArmReferenceTracker
from controllers.follow_through import FollowThrough


@dataclass(frozen=True)
class Impact:
    acceleration: float
    force: float
    impulse: float


def prefer_alternative(first, second):
    if second.impulse > first.impulse * 1.02 or second.force > first.force * 1.01:
        return False
    if first.acceleration <= 4.:
        return second.acceleration <= 4. and second.force < .98 * first.force
    return second.acceleration <= 4. or second.acceleration < .95 * first.acceleration


class InterceptImpact:
    def __init__(self, model, observer, *, method):
        if method not in ("inertia", "preview"):
            raise ValueError("Unknown contact selector")
        self.method = method
        self.model, self.observer = model, observer
        self.probe = mujoco.MjData(model)
        self.tracker = ArmReferenceTracker(model)
        self.dofs, self.qadr = self.tracker.dofs, self.tracker.qadr
        self.stop = FollowThrough()
        self.ball = model.geom("projectile").id
        self.ball_mass = model.body_mass[model.geom_bodyid[self.ball]]
        self.palm = model.body("hand/palm_lower").id
        self.offset = np.array([0.,-.055,-.08])
        self.hand = {g for g,b in enumerate(model.geom_bodyid)
                     if (model.body(b).name or "").startswith("hand/")}
        self.mass = np.empty((model.nv,model.nv))
        self.jac = np.empty((3,model.nv))
        self.attempts = self.pairs = self.timeouts = self.unresolved = 0
        self.swaps = self.applied_swaps = 0
        self.pending = self.last_applied = None

    def inertia(self, plan, deadline):
        m, d = self.model, self.probe
        stamp = plan.intercept_time
        q, v, _ = plan.reference(stamp)
        d.qpos[self.qadr] = q
        mujoco.mj_fwdPosition(m,d)
        palm = d.body(self.palm)
        rotation = palm.xmat.reshape(3,3)
        normal = rotation[:,2]
        center = palm.xpos + rotation @ self.offset
        mujoco.mj_jac(m,d,self.jac,None,center,self.palm)
        jn = normal @ self.jac[:,self.dofs]
        mujoco.mj_fullM(m,d,self.mass)
        response = np.linalg.solve(self.mass[np.ix_(self.dofs,self.dofs)], jn)
        _, ball_v = self.observer.estimate.at(stamp)
        closing = max(0.,float(normal @ ball_v - jn @ v))
        impulse = closing / (1/self.ball_mass + jn @ response)
        # Inelastic normal impulse spread uniformly over 20 ms, other joints locked.
        # Include the planned acceleration just before arrival in the joint proxy.
        acceleration = np.max(np.abs(plan.second + response * impulse / .020))
        return Impact(float(acceleration), float(impulse/.020), float(impulse))

    def preview(self, plan, deadline):
        m, d = self.model, self.probe
        dt, stamp = m.opt.timestep, plan.intercept_time
        start = stamp-.024
        d.time = start
        q, v, _ = plan.reference(start)
        d.qpos[self.qadr], d.qvel[self.dofs] = q, v
        pos, velocity = self.observer.estimate.at(start)
        flight = d.joint("flight")
        flight.qpos[:] = [*pos,1.,0.,0.,0.]
        flight.qvel[:] = [*velocity,0.,0.,0.]
        d.qacc_warmstart[:] = 0
        mujoco.mj_forward(m,d)
        onset = direction = None
        reference = plan
        peak_a = peak_f = impulse = 0.
        for _ in range(round(.082/dt)):
            if time.perf_counter() >= deadline:
                return None
            t = float(d.time)
            if onset is None and t > stamp+.036:
                return None
            arm_before = d.qvel[self.dofs].copy()
            ball_before = flight.qvel[:3].copy()
            self.tracker.command(d,reference,t)
            mujoco.mj_step(m,d)
            force = np.zeros(3)
            touched = False
            for index, c in enumerate(d.contact):
                if c.efc_address < 0:
                    continue
                if c.geom2==self.ball and c.geom1 in self.hand:
                    sign = 1.
                elif c.geom1==self.ball and c.geom2 in self.hand:
                    sign = -1.
                else:
                    continue
                wrench = np.empty(6)
                mujoco.mj_contactForce(m,d,index,wrench)
                force += sign * (c.frame.reshape(3,3).T @ wrench[:3])
                touched = True
            if touched and onset is None:
                onset = t
                direction = ball_before / max(np.linalg.norm(ball_before),1e-9)
                reference = self.stop.reference_from_state(d.time,d.qpos[self.qadr],d.qvel[self.dofs])
            if onset is not None:
                peak_a = max(peak_a, np.max(np.abs(d.qvel[self.dofs]-arm_before))/dt)
                peak_f = max(peak_f, np.linalg.norm(force))
                impulse -= (force @ direction)*dt
                if d.time >= onset+.020-1e-8:
                    return Impact(float(peak_a),float(peak_f),float(impulse))
        return None

    def choose(self, data, now, first, second, deadline):
        self.pending = None
        if max(first.intercept_time,second.intercept_time)-now > .55:
            return first
        self.attempts += 1
        deadline -= .0006
        predictions = []
        for plan in (first,second):
            if time.perf_counter() >= deadline:
                self.timeouts += 1
                return first
            mujoco.mj_copyData(self.probe,self.model,data)
            outcome = (self.preview(plan,deadline) if self.method=="preview"
                       else self.inertia(plan,deadline))
            if time.perf_counter() >= deadline:
                self.timeouts += 1
                return first
            if outcome is None:
                self.unresolved += 1
                return first
            predictions.append(outcome)
        self.pairs += 1
        swap = prefer_alternative(*predictions)
        chosen = second if swap else first
        self.pending = (chosen, predictions[int(swap)], swap)
        self.swaps += int(swap)
        return chosen

    def applied(self, plan):
        self.last_applied = None
        if self.pending is not None and self.pending[0] is plan:
            _, self.last_applied, swapped = self.pending
            self.applied_swaps += int(swapped)
        self.pending = None

    def summary(self):
        p = self.last_applied
        return dict(impact_attempts=self.attempts, impact_pairs=self.pairs,
                    budget_fallbacks=self.timeouts, unresolved_pairs=self.unresolved,
                    proposed_swaps=self.swaps, applied_swaps=self.applied_swaps,
                    last_pred_accel=None if p is None else p.acceleration,
                    last_pred_force=None if p is None else p.force,
                    last_pred_impulse=None if p is None else p.impulse)
