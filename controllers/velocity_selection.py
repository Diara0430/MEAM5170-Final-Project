"""Score an already feasible interception using its attainable terminal velocity."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VelocitySelection:
    prepare_early: bool = False

    def score(self, incoming, hand_velocity, normal, position_error, stop_margin):
        relative = incoming - hand_velocity
        normal_speed = float(relative @ normal)
        tangent = relative - normal_speed * normal
        # Rank normal closing speed first, with softer costs for sliding, tracking
        # error and using the last 35 mrad of joint travel during a smooth stop.
        return (max(0., normal_speed)**2 + .20 * float(tangent @ tangent)
                + 2500. * float(position_error @ position_error)
                + .25 * max(0., 1. - stop_margin / .035)**2)
