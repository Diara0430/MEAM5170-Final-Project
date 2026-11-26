import unittest
from types import SimpleNamespace

import numpy as np

from controllers.ball_observer import BallEstimate
from controllers.lock_timing import TimingMPC, agreement, crossing, uncertainty


class TimingTest(unittest.TestCase):
    def test_same_height_and_gap(self):
        g = np.array([0.,0.,-9.81])
        launch = BallEstimate(0.,np.array([1.,0.,.7]),np.array([-.8,.1,4.9]),g)
        history = [BallEstimate(t,*launch.at(t),g) for t in (.20,.22,.24,.26)]
        distance, dt = agreement(history,.7,4,.02)
        self.assertLess(max(distance,dt),1e-10)
        self.assertIsNone(agreement(history[:-1],.7,4,.02))
        self.assertIsNone(agreement([history[0],*history[2:]],.7,3,.02))
        late = BallEstimate(1.3,*launch.at(1.3),g)
        self.assertAlmostEqual(crossing(late,.7)[0],crossing(launch,.7)[0])

    def test_forecast_width_increases_with_horizon(self):
        g = np.array([0.,0.,-9.81])
        e = BallEstimate(.3,np.zeros(3),np.array([.5,.1,-3.]),g)
        frames = [(t,e.at(t)[0]) for t in np.arange(.00,.30,.02)]
        observer = SimpleNamespace(frames=frames,estimate=e)
        near = uncertainty(observer,.4,e.velocity)
        far = uncertainty(observer,.9,e.velocity)
        self.assertTrue(all(0<a<b for a,b in zip(near,far)))

    def test_only_delivered_reference_locks(self):
        c = TimingMPC.__new__(TimingMPC)
        c.unlock_s = c.lock_s = c.anchor = None
        plan = SimpleNamespace(point=np.ones(3),intercept_time=1.)
        c.proposal = (plan,.40,np.ones(3),None)
        c.on_applied(SimpleNamespace(),.42)
        self.assertIsNone(c.lock_s)
        c.on_applied(plan,.42)
        self.assertEqual(c.lock_s,.42)
        self.assertEqual(c.decision_s,.40)

    def test_no_single_frame_release_or_late_release(self):
        c = TimingMPC.__new__(TimingMPC)
        c.reversible = True
        c.anchor = (np.array([0.,0.,.3]),1.,np.zeros(3))
        c.unlock_streak, c.unlock_tick = 0, None
        bad = BallEstimate(.5,np.array([1.,0.,1.]),np.array([0.,0.,-1.]),np.array([0.,0.,-9.81]))
        self.assertFalse(c._release_needed(bad,.5))
        self.assertTrue(c._release_needed(bad,.52))
        self.assertFalse(c._release_needed(bad,.9))


if __name__ == "__main__":
    unittest.main()
