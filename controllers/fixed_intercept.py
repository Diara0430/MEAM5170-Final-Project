"""A shared, observation-derived target for paired terminal-speed experiments."""
from dataclasses import dataclass, field
import time

import mujoco
import numpy as np

from controllers.catching_mpc import CatchingMPC


@dataclass
class FixedIntercept:
    observed_s: float | None = None
    release_s: float | None = None
    time: float | None = None
    point: np.ndarray | None = None
    incoming: np.ndarray | None = None
    projection: np.ndarray | None = None
    levels: list = field(default_factory=list)
    preparation_ms: float = 0.
    reason: str = "no_feasible_anchor"

    def target_speed(self, label):
        if not self.levels or label == "locked30":
            return None
        if label == "low":
            return self.levels[0]
        if label == "high":
            return self.levels[-1]
        if label == "mid":
            middle = .5*(self.levels[0] + self.levels[-1])
            return min(self.levels, key=lambda v: abs(v-middle))
        raise ValueError(f"Unknown fixed-point speed {label}")


class PreparationFinished(Exception):
    """End the shared simulated observation prefix after the fixture is prepared."""


class CalibrationMPC(CatchingMPC):
    def __init__(self, *args, fixture, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixture = fixture
        self.next_attempt = .26

    def solve(self, data, p, v, now, previous, durations=None):
        empty = dict(wall_s=0., feasible_candidates=0)
        if now + 1e-9 < self.next_attempt:
            return None, empty
        if now > .45:
            raise PreparationFinished
        reserve = .10
        release = now + reserve
        # Evaluate from the held arm pose for a common release time. The full
        # rolling controller must solve again from the measured state at release.
        ahead_p = p + reserve*v + .5*reserve**2*self.model.opt.gravity
        ahead_v = v + reserve*self.model.opt.gravity
        remaining = [t-reserve for t in durations if t-reserve >= .10]
        start = time.perf_counter()
        nominal, _ = super().solve(data, ahead_p, ahead_v, release, previous, durations=remaining)
        if nominal is None:
            self.next_attempt = now + reserve
            return None, empty
        f = self.fixture
        f.observed_s, f.release_s = now, release
        f.time, f.point = nominal.intercept_time, nominal.point.copy()
        f.incoming = v + (f.time-now)*self.model.opt.gravity
        direction = f.incoming / np.linalg.norm(f.incoming)
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = data.qpos
        q, qv, _ = nominal.reference(f.time)
        probe.qpos[self.checker.qadr] = q
        mujoco.mj_forward(self.model, probe)
        palm = probe.body(self.checker.palm)
        center = palm.xpos + palm.xmat.reshape(3,3) @ self.checker.local_capture
        jac = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, probe, jac, None, center, self.checker.palm)
        f.projection = direction @ jac[:, self.checker.dofs]
        baseline_speed = float(f.projection @ qv)
        grid = np.unique(np.r_[np.arange(0., .701, .05), max(0., baseline_speed)])
        for speed in grid:
            self.last_solution = None
            plan, _ = super().solve(data, ahead_p, ahead_v, release, previous,
                        durations=[f.time-release-self.period], fixed_target=(f.point, f.incoming),
                        speed_request=(f.projection, speed))
            if plan is not None:
                f.levels.append(float(speed))
        f.preparation_ms = 1000*(time.perf_counter()-start)
        if f.preparation_ms > reserve*1000:
            f.reason = "calibration_deadline"
        else:
            f.reason = "ready" if f.levels else "no_certified_speed"
        raise PreparationFinished


class FixedPointMPC(CatchingMPC):
    def __init__(self, *args, fixture, label, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixture = fixture
        self.label = label

    def solve(self, data, p, v, now, previous, durations=None):
        f = self.fixture
        empty = dict(wall_s=0., feasible_candidates=0)
        if f.time is None or f.reason == "calibration_deadline" or now + 1e-9 < f.release_s:
            return None, empty
        duration = f.time-now-self.period
        if duration < .10:
            return None, empty
        speed = f.target_speed(self.label)
        if self.label != "locked30" and speed is None:
            return None, empty
        request = None if speed is None else (f.projection, speed)
        plan, info = super().solve(data, p, v, now, previous, durations=[duration],
                                  fixed_target=(f.point, f.incoming), speed_request=request)
        if plan is not None and (abs(plan.intercept_time-f.time)>1e-9 or
                                 not np.array_equal(plan.point, f.point)):
            raise AssertionError("Paired target or intercept time changed")
        return plan, info
