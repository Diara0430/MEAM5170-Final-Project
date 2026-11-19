"""Position targets for the scene's 16 LEAP right-hand actuators (radians)."""
import numpy as np


class LeapHand:
    def __init__(self, model):
        # Numerical joint-name order, independent of actuator ordering in MJCF.
        self.ids = [model.actuator(f"hand/{i}").id for i in range(16)]
        self.limits = model.actuator_ctrlrange[self.ids]
        self.open_pose = np.zeros(16)
        self.pregrasp = np.array([0, .15, .18, .12] * 3 + [.35, .25, .15, .12])
        self.closed = np.array([0, 1.20, 1.30, .95] * 3 + [1.25, .8, 1.1, .65])

    def command(self, data, start, end, fraction):
        """Blend two hand poses. Hold the end pose for fraction >= 1."""
        t = np.clip(fraction, 0., 1.)
        target = start + (end - start) * (t * t * (3 - 2 * t))
        data.ctrl[self.ids] = np.clip(target, self.limits[:, 0], self.limits[:, 1])
