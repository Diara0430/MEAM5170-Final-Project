"""Use remaining joint headroom to increase incoming-direction speed at the same endpoint."""
from dataclasses import replace
import time

import mujoco
import numpy as np
from scipy.optimize import lsq_linear


class FixedPointTrim:
    def __init__(self, model, observer, *, shadow=False):
        self.model, self.observer, self.shadow = model, observer, shadow
        self.probe = mujoco.MjData(model)
        self.jac = np.empty((3,model.nv))
        self.attempts = self.budget = self.proposed = self.delivered = 0
        self.speed_gains = 0.

    def choose(self, data, now, first, second, deadline):
        return first

    def refine(self, data, plan, mpc, now, deadline):
        duration = plan.intercept_time-plan.observation_time
        if duration > .65:
            return plan
        self.attempts += 1
        cutoff = deadline-.0006
        if time.perf_counter() >= cutoff-.0005:
            self.budget += 1
            return plan
        ck, m, d = mpc.checker, self.model, self.probe
        mid_q, mid_v, _ = plan.reference(plan.observation_time+duration/2)
        end_q, end_v, _ = plan.reference(plan.intercept_time)
        a, s = mpc.acceleration_limit, mpc.planning_speed_limit
        # y is the change in terminal joint velocity. These two acceleration changes
        # preserve all seven terminal joint positions exactly for equal-length phases.
        low = np.maximum.reduce([(plan.first-a)*duration, (-a-plan.second)*duration/3,
                                 -s-end_v, 2*(mid_v-s), 8*(mid_q-ck.upper)/duration])
        high = np.minimum.reduce([(plan.first+a)*duration, (a-plan.second)*duration/3,
                                  s-end_v, 2*(mid_v+s), 8*(mid_q-ck.lower)/duration])
        if np.any(low>1e-6) or np.any(high < -1e-6):
            return plan
        free = high-low > 1e-6
        if not np.any(free):
            return plan
        mujoco.mj_copyData(d,m,data)
        d.qpos[ck.qadr] = end_q
        mujoco.mj_fwdPosition(m,d)
        palm = d.body(ck.palm)
        center = palm.xpos+palm.xmat.reshape(3,3)@ck.local_capture
        mujoco.mj_jac(m,d,self.jac,None,center,ck.palm)
        J = self.jac[:,ck.dofs].copy()
        _, incoming = self.observer.estimate.at(plan.intercept_time)
        norm = np.linalg.norm(incoming)
        if norm < .1:
            return plan
        direction = incoming/norm
        # Damp the least-squares solution and penalize lateral velocity changes.
        matrix = np.vstack([J[:,free],.035*np.eye(sum(free))])
        target = np.r_[.06*direction,np.zeros(sum(free))]
        result = lsq_linear(matrix,target,bounds=(low[free],high[free]),method="bvls")
        if not result.success:
            return plan
        y = np.zeros(7)
        y[free] = result.x
        for scale in (1.,.5):
            delta = scale*y
            gain = float(direction @ J @ delta)
            if gain < .008:
                break
            alt = replace(plan,first=plan.first-delta/duration,second=plan.second+3*delta/duration)
            if not mpc.follow_through.fits(end_q,end_v+delta,ck.lower,ck.upper):
                continue
            if not self._check(alt,mpc,cutoff):
                if time.perf_counter()>=cutoff:
                    self.budget += 1
                    break
                continue
            if time.perf_counter()>=cutoff:
                self.budget += 1
                break
            q_final,v_final,_ = alt.reference(alt.intercept_time)
            if not np.allclose(q_final,end_q,atol=1e-9,rtol=0) or not np.allclose(
                    v_final,end_v+delta,atol=1e-9,rtol=0):
                raise RuntimeError("Fixed-endpoint identity failed")
            self.proposed += 1
            if self.shadow:
                return plan
            alt.trim_gain = gain
            mpc.last_solution = np.r_[alt.first,alt.second]
            return alt
        return plan

    def _check(self, plan, mpc, cutoff):
        d, m, ck = self.probe, self.model, mpc.checker
        h = (plan.intercept_time-plan.observation_time)/2
        q, v = plan.q0.copy(), plan.v0.copy()
        for a in (plan.first,plan.second):
            # A quadratic joint trajectory can also reach a position extremum inside a phase.
            root = np.divide(-v,a,out=np.zeros_like(v),where=np.abs(a)>1e-12)
            u = np.clip(root,0.,h)
            turn = q+v*u+.5*a*u*u
            q = q+h*v+.5*h*h*a
            v = v+h*a
            if (np.any(turn<ck.lower) or np.any(turn>ck.upper) or np.any(q<ck.lower)
                or np.any(q>ck.upper) or max(abs(v))>mpc.planning_speed_limit+1e-6
                or max(abs(a))>mpc.acceleration_limit+1e-6):
                return False
        for fraction in np.linspace(0,1,7):
            if time.perf_counter()>=cutoff:
                return False
            t = plan.observation_time+2*h*fraction
            q,v,a = plan.reference(t)
            # At the terminal instant reference() reports zero acceleration. Also check
            # the incoming phase's actual acceleration for inverse-dynamics feasibility.
            d.qpos[ck.qadr], d.qvel[ck.dofs] = q, v
            d.qacc[:] = 0.
            d.qacc[ck.dofs] = plan.second if fraction==1 else a
            mujoco.mj_inverse(m,d)
            tau = d.qfrc_inverse[ck.dofs]
            if np.any(tau<mpc.force_limits[:,0]) or np.any(tau>mpc.force_limits[:,1]):
                return False
            if any(ck.object not in m.geom_bodyid[[c.geom1,c.geom2]] and c.dist<-.001 for c in d.contact):
                return False
        return True

    def applied(self, plan):
        if hasattr(plan,"trim_gain"):
            self.delivered += 1
            self.speed_gains += plan.trim_gain

    def summary(self):
        return dict(trim_attempts=self.attempts,trim_budget=self.budget,
                    trim_proposed=self.proposed,trim_delivered=self.delivered,
                    mean_delivered_gain=self.speed_gains/self.delivered if self.delivered else None)
