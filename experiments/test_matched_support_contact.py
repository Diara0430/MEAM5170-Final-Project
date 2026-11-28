"""Check physical contact-point relative velocity decomposition."""
import unittest
import numpy as np
from scene_setup.scene import build_scene
from experiments.intercept_tracking import initial
from experiments.matched_support_contact_audit import contact_relative_velocity


class ContactPointVelocityTest(unittest.TestCase):
    def test_tangent_and_normal_motion(self):
        model=build_scene('box');data=initial(model)
        data.qvel[:]=0.
        data.joint('flight').qvel[:3]=[.25,0.,-.4]
        point=data.body('projectile').xpos.copy()
        obj=model.body('projectile').id;palm=model.body('hand/palm_lower').id
        normal,tangent=contact_relative_velocity(model,data,point,obj,palm,np.array([0.,0.,1.]))
        self.assertAlmostEqual(normal,-.4,places=8)
        self.assertAlmostEqual(tangent,.25,places=8)
        data.qvel[:]=0.
        normal,tangent=contact_relative_velocity(model,data,point,obj,palm,np.array([0.,0.,1.]))
        self.assertAlmostEqual(normal,0.,places=8)
        self.assertAlmostEqual(tangent,0.,places=8)

if __name__=='__main__':unittest.main()
