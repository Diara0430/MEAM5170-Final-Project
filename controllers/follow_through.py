"""Joint-space stopping reference for the velocity-matched catch experiment."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BPoly


@dataclass(frozen=True)
class FollowThrough:
    acceleration_limit: float = 3.0
    jerk_limit: float = 120.0
    minimum_time: float = 0.20

    def __post_init__(self):
        settings = (self.acceleration_limit, self.jerk_limit, self.minimum_time)
        if not np.all(np.isfinite(settings)) or min(settings) <= 0:
            raise ValueError("Stopping limits and duration must be finite and positive")

    def duration(self, velocity):
        speed = np.max(np.abs(velocity))
        return max(self.minimum_time, 1.5 * speed / self.acceleration_limit,
                   np.sqrt(6 * speed / self.jerk_limit))

    def fits(self, position, velocity, lower, upper):
        end = position + 0.5 * self.duration(velocity) * velocity
        return bool(np.all(position >= lower) and np.all(position <= upper)
                    and np.all(end >= lower) and np.all(end <= upper))

    def reference_from_state(self, now, position, velocity):
        """Stop monotonically from measured q,v with zero endpoint accelerations."""
        duration = self.duration(velocity)
        end = position + 0.5 * duration * velocity
        zero = np.zeros_like(velocity)
        curve = BPoly.from_derivatives(
            [0.0, duration], [[position, velocity, zero], [end, zero, zero]],
            extrapolate=False,
        )
        return StopReference(now, duration, curve, curve.derivative(), curve.derivative(2))


@dataclass
class StopReference:
    start: float
    duration: float
    position: BPoly
    velocity: BPoly
    acceleration: BPoly

    def reference(self, t):
        elapsed = float(np.clip(t - self.start, 0.0, self.duration))
        return self.position(elapsed), self.velocity(elapsed), self.acceleration(elapsed)
