"""Require recent applied arm targets and ball estimates to agree before committing."""
from collections import deque

import numpy as np

from controllers.lock_timing import TimingMPC


class SettledInterceptMPC(TimingMPC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delivered = deque(maxlen=2)
        self.gate_checks = self.gate_rejections = 0

    def on_applied(self, plan, now):
        super().on_applied(plan,now)
        if self.lock_s is None:
            self.delivered.append((now,plan.point.copy(),plan.intercept_time))

    def _gate(self, now, plan, incoming):
        passed,metrics = super()._gate(now,plan,incoming)
        if not passed:
            return False,None
        self.gate_checks += 1
        if len(self.delivered)==2:
            first, second = self.delivered
            # QP solutions reach the arm one camera cycle after their proposal.
            consecutive = abs(second[0]-now)<1e-7 and abs(second[0]-first[0]-self.period)<1e-7
            points = np.array([first[1],second[1],plan.point])
            times = [first[2],second[2],plan.intercept_time]
            spread = np.max(np.linalg.norm(points[:,None]-points[None,:],axis=2))
            if consecutive and spread<=.010 and np.ptp(times)<=.004:
                return True,metrics
        self.gate_rejections += 1
        return False,None


class EarlyStableMPC(TimingMPC):
    """Lock a stable prediction only while a full 350 ms would remain on delivery."""
    def _gate(self, now, plan, incoming):
        remaining_on_delivery = plan.intercept_time - (now + self.period)
        if remaining_on_delivery < .350 - 1e-9:
            return False, None
        return super()._gate(now, plan, incoming)
