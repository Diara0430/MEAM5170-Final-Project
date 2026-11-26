"""A delayed 3-D position sensor and a known-gravity flight estimator (SI units)."""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallEstimate:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    gravity: np.ndarray

    def at(self, time):
        dt = time - self.time
        return (self.position + dt * self.velocity + 0.5 * dt**2 * self.gravity,
                self.velocity + dt * self.gravity)

    def descending_crossing(self, height):
        # The later root of z(t) = height, with negative vertical velocity.
        g = self.gravity[2]
        disc = self.velocity[2]**2 - 2 * g * (self.position[2] - height)
        if disc <= 0:
            return None
        dt = (-self.velocity[2] - np.sqrt(disc)) / g
        return self.time + dt if dt >= 0 else None


class BallObserver:
    """Receive one world-frame measurement per camera tick; fit only delivered frames."""
    def __init__(self, gravity, seed, noise_m=.002, latency_s=.020, min_frames=8, window=16, flight_fit=None):
        self.gravity = np.array(gravity, dtype=float)
        if (self.gravity.shape != (3,) or not np.isfinite(self.gravity).all()
                or self.gravity[2] >= 0 or not np.isfinite([noise_m, latency_s]).all()
                or min(noise_m, latency_s) < 0 or not 2 <= min_frames <= window):
            raise ValueError("Invalid camera or ballistic model settings")
        self.rng = np.random.default_rng(seed)
        self.noise_m = noise_m
        self.latency_s = latency_s
        self.flight_fit = flight_fit
        self.min_frames = min_frames
        self.pending = deque()
        self.frames = deque(maxlen=window)
        self.last_time = -np.inf
        self.estimate = None
        self.first_estimate = None

    def sample(self, now, position):
        if now <= self.last_time:
            raise ValueError("Camera timestamps must increase")
        self.last_time = now
        measured = np.asarray(position) + self.rng.normal(0, self.noise_m, 3)
        self.pending.append((now, measured))
        while self.pending and self.pending[0][0] + self.latency_s <= now + 1e-9:
            self.frames.append(self.pending.popleft())
        if len(self.frames) < self.min_frames:
            return None
        # Fit an intercept and slope after subtracting known gravity. Centering the
        # timestamps keeps the slope well-conditioned and accounts for sensor latency.
        dt = np.array([t - now for t, _ in self.frames])
        y = np.array([p for _, p in self.frames]) - 0.5 * dt[:, None]**2 * self.gravity
        if self.flight_fit is None:
            centered = dt - dt.mean()
            velocity = (centered @ y) / (centered @ centered)
            position = y.mean(axis=0) - dt.mean() * velocity
        else:
            position, velocity = self.flight_fit.fit(now, dt, y, self.noise_m)
        self.estimate = BallEstimate(now, position, velocity, self.gravity)
        if self.first_estimate is None:
            self.first_estimate = self.estimate
        return self.estimate


def crossing_durations(estimate, home_height, offsets, now, lag):
    """Candidate intercepts are on the descending arc, referenced to the initial palm."""
    durations = []
    for offset in offsets:
        crossing = estimate.descending_crossing(home_height + offset)
        if crossing is not None:
            duration = crossing - now - lag
            if .10 <= duration <= 1.20:
                durations.append(duration)
    return durations
