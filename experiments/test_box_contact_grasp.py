import unittest
from unittest.mock import patch

import numpy as np

from controllers.box_contact_grasp import ContactGrasp
from controllers.hand import LeapHand
from controllers.sphere_cage import SphereCageClosure
from experiments.intercept_tracking import initial
from scene_setup.scene import build_scene


class ContactGraspTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_scene("box")

    def setup_hand(self, policy):
        return ContactGrasp(self.model, LeapHand(self.model), policy), initial(self.model)

    @patch.object(SphereCageClosure, "command")
    def test_one_side_yields_and_opposition_releases(self, unused):
        hand, data = self.setup_hand("tactile")
        hand.support.loaded = {0, -1}
        for i in range(8):
            data.ctrl[:] = 0.
            hand.command(self.model, data, .002*i)
        self.assertLess(hand.bias[0], 0.)
        self.assertTrue(np.all(hand.bias[1:] > 0.))
        self.assertLessEqual(hand.max_torque_change, .450001)
        hand.support.holding = True
        for i in range(12):
            hand.command(self.model, data, .020+.002*i)
        np.testing.assert_array_equal(hand.bias, np.zeros(4))

    @patch.object(SphereCageClosure, "command")
    def test_recapture_requires_preceding_opposition(self, unused):
        hand, data = self.setup_hand("recapture")
        hand.command(self.model, data, .010)
        self.assertEqual(hand.recapture_frames, 0)
        hand.support.holding = True
        hand.command(self.model, data, .020)
        hand.support.holding = False
        hand.command(self.model, data, .022)
        self.assertEqual(hand.recapture_frames, 1)
        hand.command(self.model, data, .070)
        self.assertEqual(hand.recapture_frames, 1)

    @patch.object(SphereCageClosure, "prepare")
    def test_lead_requires_nearby_observed_arrival(self, unused):
        hand, data = self.setup_hand("lead")
        hand.prepare(self.model, data, .10, 1.)
        self.assertEqual(hand.before_frames, 0)
        hand.prepare(self.model, data, .97, 1.)
        self.assertGreater(hand.before_frames, 0)
