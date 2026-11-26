import unittest

import numpy as np

from controllers.ball_observer import BallObserver, crossing_durations


class BallObservationTest(unittest.TestCase):
    def test_delayed_frames_and_future_crossings(self):
        gravity = np.array([0., 0., -9.81])
        p0, v0 = np.array([1., .2, .6]), np.array([-.9, -.2, 4.9])
        sensor = BallObserver(gravity, 7, noise_m=0.)
        for frame in range(9):
            t = .02*frame
            position = p0 + v0*t + .5*gravity*t*t
            # At 160 ms this measurement is still in transit.
            estimate = sensor.sample(t, position if frame < 8 else position + 100.)
            if frame < 8:
                self.assertIsNone(estimate)
        self.assertAlmostEqual(estimate.time, .16)
        for horizon in (0., .2, .4):
            t = .16 + horizon
            p, v = estimate.at(t)
            np.testing.assert_allclose(p, p0 + v0*t + .5*gravity*t*t, atol=1e-11)
            np.testing.assert_allclose(v, v0 + gravity*t, atol=1e-11)
        offsets = (.065, 0., -.065)
        durations = crossing_durations(estimate, .6, offsets, .16, .02)
        self.assertEqual(len(durations), 3)
        for duration, offset in zip(durations, offsets):
            p, v = estimate.at(.16 + .02 + duration)
            self.assertAlmostEqual(p[2], .6+offset)
            self.assertLess(v[2], 0.)


if __name__ == "__main__":
    unittest.main()
