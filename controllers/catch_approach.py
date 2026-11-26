"""Opt-in geometric timing, bounded palm aim, and speed-dependent matching."""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CatchApproach:
    fraction: float = 0.20
    nonlinear: bool = False
    tilt: bool = False
    entry: bool = False

    def __post_init__(self):
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("Velocity matching fraction must be between zero and one")

    def match_fraction(self, speed):
        if not self.nonlinear:
            return self.fraction
        # Smoothly raise the target from 10% at 2.4 m/s to 30% at 3.6 m/s.
        x = np.clip((speed - 2.4) / 1.2, 0.0, 1.0)
        return float(0.10 + 0.20 * x * x * (3.0 - 2.0 * x))

    def palm_target(self, home, incoming):
        if not self.tilt:
            return home
        # The catch model's local +z points into the palm.
        normal = home[:, 2]
        axis = np.cross(normal, incoming)
        sine = np.linalg.norm(axis)
        if sine < 1e-9:
            return home
        incidence = np.arctan2(sine, incoming @ normal)
        # Choose one candidate before the QP to keep the planning budget fixed.
        choices = np.minimum(np.deg2rad([0.0, 4.0, 8.0]), incidence)
        score = (incidence - choices) ** 2 + 0.25 * choices ** 2
        return Rotation.from_rotvec(axis * choices[np.argmin(score)] / sine).as_matrix() @ home


class EntryTrigger:
    """Predict entry into the moving palm's capture plane over a short horizon."""
    lead = 0.026
    contact_grace = 0.012

    def __init__(self, model, checker):
        self.palm = checker.palm
        self.offset = checker.local_capture
        self.radius = checker.radius
        self.jac = np.empty((3, model.nv))
        self.ang_jac = np.empty_like(self.jac)

    def decision(self, model, data, now, first_contact):
        palm = data.body(self.palm)
        rotation = palm.xmat.reshape(3, 3)
        center = palm.xpos + rotation @ self.offset
        ball = data.joint("flight")
        relative = ball.qpos[:3] - center
        mujoco.mj_jac(model, data, self.jac, self.ang_jac, center, self.palm)
        omega = self.ang_jac @ data.qvel
        velocity = rotation.T @ (ball.qvel[:3] - self.jac @ data.qvel - np.cross(omega, relative))
        position = rotation.T @ relative
        # Instantaneous palm-frame velocity plus known gravity over at most 26 ms.
        # Use the first forward crossing, interpolated over one physical step.
        if velocity[2] > 0 and position[2] <= 0:
            gravity = rotation.T @ model.opt.gravity
            dt = model.opt.timestep
            times = np.linspace(0, self.lead, int(np.ceil(self.lead / dt)) + 1)
            z = position[2] + velocity[2] * times + 0.5 * gravity[2] * times**2
            hits = np.flatnonzero(z >= 0)
            if hits.size:
                i = int(hits[0])
                when = 0.0 if i == 0 else times[i-1] - z[i-1] * (times[i] - times[i-1]) / (z[i] - z[i-1])
                lateral = position[:2] + velocity[:2] * when + 0.5 * gravity[:2] * when**2
                if np.linalg.norm(lateral) <= 1.25 * self.radius:
                    return "predicted_entry"
        if first_contact is not None:
            if position[2] >= 0 or velocity[2] <= 0 or now - first_contact >= self.contact_grace:
                return "contact_fallback"
        return None
