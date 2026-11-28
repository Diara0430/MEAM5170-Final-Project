"""Check support-assist command bounds and shadow behavior."""
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from controllers.hand import LeapHand
from controllers.sphere_cage import SphereCageClosure
from controllers.sustained_support_grasp import SustainedSupportGrasp
from scene_setup.scene import build_scene


class SupportAssistTest(unittest.TestCase):
    def test_shadow_does_not_change_commands_and_active_respects_limits(self):
        model=build_scene('box');hand=LeapHand(model)
        active=SustainedSupportGrasp(model,hand)
        shadow=SustainedSupportGrasp(model,hand,shadow=True)
        da=mujoco.MjData(model);ds=mujoco.MjData(model)
        midpoint=np.mean(hand.limits,axis=1)
        da.ctrl[hand.ids]=midpoint;ds.ctrl[hand.ids]=midpoint
        active.support.loaded={1};shadow.support.loaded={1}
        active.support.holding=False;shadow.support.holding=False
        with patch.object(SphereCageClosure,'command',return_value=None):
            active.command(model,da,.04)
            shadow.command(model,ds,.04)
        np.testing.assert_array_equal(ds.ctrl[hand.ids],midpoint)
        self.assertGreater(active.delivered_steps,0)
        self.assertEqual(shadow.delivered_steps,0)
        self.assertGreater(active.max_torque_change,0)
        self.assertLessEqual(active.max_torque_change,.25+1e-9)
        self.assertTrue(np.all(da.ctrl[hand.ids]>=hand.limits[:,0]))
        self.assertTrue(np.all(da.ctrl[hand.ids]<=hand.limits[:,1]))
        # 承载中指回让，未承载无名指向内；偏置每步受限于 1 rad/s。
        self.assertLess(active.bias[1],0.)
        self.assertGreater(active.bias[2],0.)
        self.assertLessEqual(np.max(np.abs(active.bias)),model.opt.timestep+1e-12)

if __name__=='__main__': unittest.main()
