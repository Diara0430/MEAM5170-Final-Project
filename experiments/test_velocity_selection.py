import unittest

import numpy as np

from controllers.velocity_selection import VelocitySelection


class VelocitySelectionTest(unittest.TestCase):
    def test_relative_normal_speed_and_stopping_reserve(self):
        policy = VelocitySelection()
        incoming = np.array([1., 0., -5.])
        normal = np.array([0., 0., -1.])
        zero = np.zeros(3)
        aligned = np.array([0., 0., -.4])
        lateral = np.array([.4, 0., 0.])
        def score(velocity, margin=.1):
            return policy.score(incoming, velocity, normal, zero, margin)
        self.assertLess(score(aligned), score(lateral))
        self.assertLess(score(aligned), score(zero))
        self.assertLess(score(aligned), score(aligned, .010))
        self.assertLess(score(aligned), score(-aligned))
