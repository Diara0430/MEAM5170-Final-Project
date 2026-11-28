"""Checks for geometry-selected support and shadow safety."""
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from scene_setup.scene import build_scene
from controllers.hand import LeapHand
from controllers.geometry_support_grasp import GeometrySupportGrasp
from controllers.sphere_cage import SphereCageClosure


class GeometrySupportTest(unittest.TestCase):
    def test_selection_shadow_and_loaded_finger_preservation(self):
        model=build_scene('box');hand=LeapHand(model)
        active=GeometrySupportGrasp(model,hand);shadow=GeometrySupportGrasp(model,hand,shadow=True)
        da=mujoco.MjData(model);ds=mujoco.MjData(model)
        midpoint=np.mean(hand.limits,axis=1)
        for data in (da,ds):
            data.ctrl[hand.ids]=midpoint
            for b in active.tip_bodies:data.xpos[b]=[.3,0.,0.]
            data.xpos[active.tip_bodies[0]]=[0.,0.,.05]
        for controller in (active,shadow):
            controller.contact_point=np.zeros(3);controller.contact_normal=np.array([0.,0.,1.])
            controller.touch_groups={1}
        with patch.object(SphereCageClosure,'command',return_value=None):
            active.command(model,da,.03);shadow.command(model,ds,.03)
        self.assertEqual(active.selected,0)
        self.assertGreater(active.delivered_steps,0)
        self.assertEqual(shadow.delivered_steps,0)
        np.testing.assert_array_equal(ds.ctrl[hand.ids],midpoint)
        self.assertLessEqual(active.max_torque_change,.2+1e-9)
        np.testing.assert_array_equal(da.ctrl[np.asarray(hand.ids)[[5,6,7]]],midpoint[[5,6,7]])
        # 偏置尚未衰减完时，刚承载的手指也不再接收额外推进。
        da.ctrl[hand.ids]=midpoint;active.touch_groups={0}
        with patch.object(SphereCageClosure,'command',return_value=None):active.command(model,da,.032)
        np.testing.assert_array_equal(da.ctrl[np.asarray(hand.ids)[[1,2,3]]],midpoint[[1,2,3]])

if __name__=='__main__':unittest.main()
