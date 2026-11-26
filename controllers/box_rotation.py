"""Delayed orientation observations for a rigid box. Public quaternions use MuJoCo wxyz."""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


def from_wxyz(quaternion):
    return Rotation.from_quat(np.roll(quaternion, -1))


def as_wxyz(rotation):
    return np.roll(rotation.as_quat(), 1)


class RotationTrack:
    def __init__(self, rng, latency, noise_deg):
        self.rng = rng
        self.latency = latency
        self.noise = np.deg2rad(noise_deg)
        self.pending = deque()
        self.frames = deque(maxlen=7)
        self.reference = None
        self.reference_time = None
        self.rate = None

    def sample(self, now, quaternion):
        measured = from_wxyz(quaternion) * Rotation.from_rotvec(self.rng.normal(0, self.noise, 3))
        self.pending.append((now, measured))
        while self.pending and self.pending[0][0] + self.latency <= now + 1e-9:
            self.frames.append(self.pending.popleft())
        if len(self.frames) < 3:
            return
        reference_time, reference = self.frames[-1]
        if reference_time == self.reference_time:
            return
        times = np.array([t-reference_time for t, _ in self.frames])
        offsets = np.array([(reference.inv()*r).as_rotvec() for _, r in self.frames])
        centered = times-times.mean()
        self.rate = (centered @ offsets) / (centered @ centered)
        intercept = offsets.mean(axis=0) - times.mean()*self.rate
        self.reference = reference * Rotation.from_rotvec(intercept)
        self.reference_time = reference_time

    def at(self, when):
        if self.reference is None:
            return None
        return self.reference * Rotation.from_rotvec((when-self.reference_time)*self.rate)

    def write(self, data, when):
        rotation = self.at(when)
        if rotation is not None:
            flight = data.joint("flight")
            flight.qpos[3:] = as_wxyz(rotation)
            flight.qvel[3:] = self.rate


class BoxExtent:
    """Match the box leading surface to the baseline's nominal spherical contact surface."""
    def __init__(self, model, observer):
        self.half_size = model.geom("projectile").size.copy()
        self.track = observer.preview_rotation
        self.deliveries = 0

    def adjust_target(self, point, normal, stamp):
        rotation = self.track.at(stamp)
        if rotation is None:
            return point
        half_extent = np.abs(rotation.inv().apply(normal)) @ self.half_size
        return point + (half_extent - .025)*normal

    def choose(self, data, now, first, second, deadline):
        return first

    def applied(self, plan):
        self.deliveries += 1

    def summary(self):
        return {"extent_plans": self.deliveries}
