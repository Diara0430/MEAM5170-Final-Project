"""Decide whether an applied intercept should keep a fixed world point and time."""
from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from controllers.catching_mpc import CatchingMPC


@dataclass(frozen=True)
class LockRule:
    clock: float | None = None
    lead: float | None = None
    frames: int = 0
    spread_m: float = 0.
    spread_s: float = 0.
    confidence: bool = False


RULES = {"rolling": LockRule(), "clock26": LockRule(clock=.26), "clock44": LockRule(clock=.44),
         "clock62": LockRule(clock=.62), "lead35": LockRule(lead=.35),
         "stable4": LockRule(frames=4, spread_m=.010, spread_s=.004),
         "stable5": LockRule(frames=5, spread_m=.006, spread_s=.002),
         "confidence4": LockRule(frames=4, spread_m=.010, spread_s=.004, confidence=True)}


def crossing(estimate, height):
    """Descending plane crossing, including a recently passed crossing for audit."""
    vz = estimate.velocity[2]
    gz = estimate.gravity[2]
    discriminant = vz*vz - 2*gz*(estimate.position[2]-height)
    if discriminant <= 0:
        return None
    delta = (-vz-np.sqrt(discriminant))/gz
    stamp = estimate.time+delta
    position, velocity = estimate.at(stamp)
    return stamp, position, velocity


def agreement(history, height, count, period):
    if len(history) < count:
        return None
    recent = list(history)[-count:]
    if np.max(np.diff([e.time for e in recent])) > period+1e-7:
        return None
    predictions = [crossing(e,height) for e in recent]
    if any(x is None for x in predictions):
        return None
    times = [x[0] for x in predictions]
    points = np.array([x[1][:2] for x in predictions])
    span = np.max(np.linalg.norm(points[:,None]-points[None,:],axis=2))
    return float(span), float(np.ptp(times))


def uncertainty(observer, when, incoming):
    """Residual-based position and time widths for the fitted mean crossing."""
    samples = list(observer.frames)
    times = np.array([t for t, _ in samples])
    centered = times-times.mean()
    sxx = centered@centered
    if len(samples) < 4 or sxx <= 0 or abs(incoming[2]) < .1:
        return float("inf"), float("inf")
    estimate = observer.estimate
    residual = np.array([point-estimate.at(t)[0] for t, point in samples])
    covariance = residual.T@residual/(len(samples)-2) + np.eye(3)*1e-6
    covariance *= 1./len(samples)+(when-times.mean())**2/sxx
    plane = np.column_stack((np.eye(2), -incoming[:2]/incoming[2]))
    radius = 2.45*np.sqrt(np.linalg.eigvalsh(plane@covariance@plane.T)[-1])
    width = 1.96*np.sqrt(covariance[2,2])/abs(incoming[2])
    return float(radius), float(width)


class TimingMPC(CatchingMPC):
    def __init__(self, *args, observer, rule, reversible=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer = observer
        self.rule = rule
        self.reversible = reversible
        self.history = deque(maxlen=5)
        self.proposal = self.anchor = None
        self.lock_s = self.decision_s = self.unlock_s = None
        self.lock_point = self.lock_target_s = self.lock_metrics = None
        self.unlock_streak = 0
        self.unlock_tick = None
        self.replacement_s = None

    def on_applied(self, plan, now):
        # Only the actuator-side acceptance of the proposed plan commits a lock.
        if self.unlock_s is not None and self.replacement_s is None and (
                abs(plan.intercept_time-self.lock_target_s)>1e-8 or not np.array_equal(plan.point,self.lock_point)):
            self.replacement_s = now
        if self.proposal is None or self.proposal[0] is not plan:
            return
        _, stamp, incoming, metrics = self.proposal
        self.anchor = (plan.point.copy(), plan.intercept_time, incoming.copy())
        self.lock_point, self.lock_target_s = self.anchor[:2]
        self.decision_s, self.lock_s, self.lock_metrics = stamp, now, metrics
        self.proposal = None

    def _gate(self, now, plan, incoming):
        rule = self.rule
        if rule.clock is not None:
            return now+1e-8 >= rule.clock, None
        if rule.lead is not None:
            return plan.intercept_time-now <= rule.lead+1e-8, None
        if not rule.frames:
            return False, None
        spread = agreement(self.history, plan.point[2], rule.frames, self.period)
        if spread is None or spread[0]>rule.spread_m or spread[1]>rule.spread_s:
            return False, None
        radius, width = uncertainty(self.observer, plan.intercept_time, incoming) if rule.confidence else (None,None)
        passes = not rule.confidence or (radius<=.012 and width<=.004)
        return passes, (*spread, radius, width)

    def _release_needed(self, estimate, now):
        point, stamp, _ = self.anchor
        if not self.reversible or stamp-now < .16:
            return False
        predicted = crossing(estimate, point[2])
        disagree = predicted is None or (np.linalg.norm(predicted[1][:2]-point[:2]) > .020
                                        or abs(predicted[0]-stamp) > .006)
        consecutive = self.unlock_tick is not None and now-self.unlock_tick <= self.period+1e-7
        self.unlock_streak = (self.unlock_streak+1 if consecutive else 1) if disagree else 0
        self.unlock_tick = now
        return self.unlock_streak >= 2

    def solve(self, data, p, v, now, previous, durations=None):
        start = time.perf_counter()
        estimate = self.observer.estimate
        self.history.append(estimate)
        self.proposal = None
        if self.anchor is not None and self._release_needed(estimate, now):
            self.anchor = None
            self.unlock_s = now
        if self.anchor is None:
            plan, info = super().solve(data,p,v,now,previous,durations=durations)
            if plan is not None and self.lock_s is None:
                incoming = v+(plan.intercept_time-now)*self.model.opt.gravity
                passed, metrics = self._gate(now,plan,incoming)
                if passed:
                    self.proposal = (plan,now,incoming,metrics)
        else:
            point, stamp, incoming = self.anchor
            duration = stamp-now-self.period
            if duration >= .10:
                plan, info = super().solve(data,p,v,now,previous,durations=[duration],fixed_target=(point,incoming))
                if plan is not None and (abs(plan.intercept_time-stamp)>1e-8 or not np.array_equal(plan.point,point)):
                    raise AssertionError("Locked target changed")
            else:
                plan, info = None, dict(feasible_candidates=0)
        info["wall_s"] = time.perf_counter()-start
        return plan, info
