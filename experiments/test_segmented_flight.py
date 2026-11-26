import unittest

import numpy as np

from controllers.ball_observer import BallObserver
from controllers.segmented_flight import SegmentedFlightFit


class SegmentedFitTest(unittest.TestCase):
    def track(self, shifted):
        gravity = np.array([0., 0., -9.81])
        fit = SegmentedFlightFit()
        camera = BallObserver(gravity, 151, noise_m=0., latency_s=.04, window=48, flight_fit=fit)
        for frame in range(85):
            now = .02*frame
            bias = np.array([.032,0.,0.]) if shifted and frame<26 else np.zeros(3)
            pos = np.array([now,-.5*now,.8+4*now]) + .5*gravity*now**2
            estimate = camera.sample(now,pos+bias)
            if frame in (30,59,82):
                np.testing.assert_allclose(estimate.at(now)[0],pos,atol=1e-8)
                np.testing.assert_allclose(estimate.velocity,np.array([1.,-.5,4.])+gravity*now,atol=1e-8)
        self.assertEqual(fit.change_count,int(shifted))

    def test_clean_and_shifted_flight(self):
        for shifted in (False,True):
            with self.subTest(shifted=shifted):
                self.track(shifted)


if __name__ == "__main__":
    unittest.main()
