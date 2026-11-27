"""Experimental calibration: certify only the nominal intercept speed."""
import time
import mujoco
import numpy as np
from controllers.fixed_intercept import CalibrationMPC, PreparationFinished
from controllers.catching_mpc import CatchingMPC


class FastCalibrationMPC(CalibrationMPC):
    """Experimental single nominal speed; not a low/high speed grid certificate."""
    def __init__(self, *args, fixture, **kwargs):
        super().__init__(*args, fixture=fixture, **kwargs)
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
        nominal, _ = CatchingMPC.solve(self, data, ahead_p, ahead_v, release, previous, durations=remaining)
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
        grid = [max(0., baseline_speed)]
        for speed in grid:
            self.last_solution = None
            plan, _ = CatchingMPC.solve(self, data, ahead_p, ahead_v, release, previous,
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

