"""Tests for oriented box distance and bounded finger linearization."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.analyze_moving_surface import box_signed_distance,local_finger_residual


class MovingSurfaceTest(unittest.TestCase):
    def test_box_surface_inside_outside_and_rotation(self):
        pose=np.array([1.,2.,3.,1.,0.,0.,0.]);half=np.array([.032,.021,.017])
        self.assertAlmostEqual(box_signed_distance([1,2,3],pose,half),-.017)
        self.assertAlmostEqual(box_signed_distance([1.032,2,3],pose,half),0.)
        self.assertAlmostEqual(box_signed_distance([1.042,2,3],pose,half),.010)
        quat=Rotation.from_euler('z',90,degrees=True).as_quat()
        rotated=np.r_[pose[:3],np.roll(quat,1)]
        self.assertAlmostEqual(box_signed_distance([1,2.042,3],rotated,half),.010)

    def test_bounded_local_motion_and_invalid_joint_range(self):
        J=np.c_[np.eye(3),np.zeros(3)];q=np.zeros(4);lo=-np.ones(4);hi=np.ones(4)
        residual,status=local_finger_residual(J,np.zeros(3),np.array([.1,0,0]),q,lo,hi)
        self.assertEqual(status,'ok');self.assertLess(residual,1e-6)
        residual,status=local_finger_residual(J,np.zeros(3),np.array([.3,0,0]),q,lo,hi)
        self.assertEqual(status,'ok');self.assertAlmostEqual(residual,.1,places=5)
        residual,status=local_finger_residual(J,np.zeros(3),np.zeros(3),np.full(4,2.),lo,hi)
        self.assertIsNone(residual);self.assertEqual(status,'joint_range_inconsistent')

if __name__=='__main__':unittest.main()
