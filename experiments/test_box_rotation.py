import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from controllers.box_rotation import BoxExtent, RotationTrack, as_wxyz
from experiments.shape_catch import box_launch


class RotationTests(unittest.TestCase):
    def test_planning_clearance_for_supported_shapes(self):
        from controllers.intercept_planner import InterceptPlanner
        from experiments.intercept_tracking import initial
        from scene_setup.scene import build_scene

        for shape in ("box", "ball"):
            model = build_scene(shape)
            planner = InterceptPlanner(model, initial(model))
            self.assertAlmostEqual(planner.radius, np.linalg.norm(model.geom("projectile").size))

    def test_physical_spin_convention(self):
        import mujoco
        from controllers.box_rotation import from_wxyz
        from scene_setup.scene import build_scene

        model = build_scene("box")
        data = mujoco.MjData(model)
        quat, omega = box_launch(1051, 2)
        flight = data.joint("flight")
        flight.qpos[:] = [4., 4., 4., *quat]
        flight.qvel[3:] = omega
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        measured = (from_wxyz(quat).inv()*from_wxyz(flight.qpos[3:])).as_rotvec()
        np.testing.assert_allclose(measured, omega*model.opt.timestep, atol=3e-5)

    def test_delay_sign_and_body_rate(self):
        track = RotationTrack(np.random.default_rng(10), .04, 0.)
        start = Rotation.from_euler("xyz", [.4, -.7, .1])
        omega = np.array([.3, 1.4, -.5])
        for i in range(9):
            t = .02*i
            quat = as_wxyz(start*Rotation.from_rotvec(t*omega))
            track.sample(t, -quat if i % 2 else quat)
            if i < 4:
                self.assertIsNone(track.at(t))
        self.assertAlmostEqual(track.reference_time, .12)
        goal = start*Rotation.from_rotvec(.26*omega)
        self.assertLess((goal.inv()*track.at(.26)).magnitude(), 1e-10)
        np.testing.assert_allclose(track.rate, omega, atol=1e-10)

    def test_box_extent_and_launch_pairing(self):
        for u, v in zip(box_launch(1051, 2), box_launch(1051, 2)):
            np.testing.assert_equal(u, v)
        extent = BoxExtent.__new__(BoxExtent)
        extent.half_size = np.array([.032, .021, .017])
        extent.track = type("Track", (), {"at": lambda self, t: Rotation.identity()})()
        for axis in np.eye(3):
            delta = extent.adjust_target(np.zeros(3), axis, 1.)
            np.testing.assert_allclose(delta, (extent.half_size @ axis - .025) * axis)
