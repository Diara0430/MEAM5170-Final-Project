"""Try a small tangential palm offset while waiting for the first centered intercept."""
import time

import numpy as np

from controllers.lock_timing import TimingMPC


class PalmAimRecovery(TimingMPC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_delivered = False
        self.try_offset = False
        self.pending_aim = None
        self.offset_trials = self.offset_deliveries = 0
        self.first_offset = None

    def on_applied(self, plan, now):
        super().on_applied(plan, now)
        if self.contact_selector is not None:
            self.contact_selector.applied(plan)
        if self.pending_aim is None or self.pending_aim[0] is not plan:
            return
        if self.pending_aim[1]:
            self.offset_deliveries += 1
            if self.first_offset is None:
                self.first_offset = now
        else:
            self.center_delivered = True

    def solve(self, data, p, v, now, previous, durations=None):
        started = time.perf_counter()
        aim = np.zeros(3)
        if not self.center_delivered and self.try_offset and durations is not None and len(durations):
            palm = data.body(self.checker.palm)
            rotation = palm.xmat.reshape(3,3)
            center = palm.xpos + rotation @ self.checker.local_capture
            # Use an observed descending crossing to decide whether extra reach is needed.
            crossing = self.observer.estimate.descending_crossing(center[2])
            if crossing is not None:
                point, _ = self.observer.estimate.at(crossing)
                toward = center-point
                tangent = toward - rotation[:,2] * (rotation[:,2] @ toward)
                distance = np.linalg.norm(tangent)
                if distance > .045:
                    aim = .016 * tangent/distance
                    self.offset_trials += 1
        self.try_offset = not self.try_offset
        # This translates only the commanded palm center. The ball track used by
        # the torque preview and by the plant is unchanged.
        shifted = bool(np.any(aim))
        adjust = getattr(self.contact_selector, "contact_offset", None)
        spot = None
        if adjust is not None and self.center_delivered:
            spot = adjust(data, previous, now)
        target = p+aim if shifted else p
        if spot is not None:
            target = target+spot
        plan, info = super().solve(data, target, v, now, previous, durations=durations)
        refine = getattr(self.contact_selector, "refine", None)
        if plan is not None and refine is not None:
            plan = refine(data, plan, self, now, started+self.period)
        if plan is not None and spot is not None:
            plan.impact_offset = True
        self.pending_aim = (plan, shifted) if plan is not None else None
        info["wall_s"] = time.perf_counter()-started
        return plan, info
