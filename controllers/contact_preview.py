"""Use a delayed ball track only inside the arm filter's pre-contact dynamics model."""
import time

import mujoco
import numpy as np

from controllers.discrete_arm_filter import DiscreteArmFilter


class ContactPreviewFilter(DiscreteArmFilter):
    def __init__(self, model, observer, enabled):
        super().__init__(model)
        self.observer = observer
        self.enabled = enabled
        self.robot_with_estimate = mujoco.MjData(model)
        self.palm = model.body("hand/palm_lower").id
        self.offset = np.array([0., -.055, -.08])
        self.touched = False
        self.near_times_ms = []
        self.near_steps = 0
        self.last_near_t = None

    def command(self, data, plan, t):
        start = time.perf_counter()
        estimate = self.observer.estimate
        if self.touched or estimate is None:
            return super().command(data, plan, t)
        p, v = estimate.at(t)
        palm = data.body(self.palm)
        to_ball = p - (palm.xpos + palm.xmat.reshape(3, 3) @ self.offset)
        # Keep the preview active as the ball passes the virtual center and enters the fingers.
        near = to_ball @ to_ball <= .10**2 and v @ palm.xmat.reshape(3, 3)[:, 2] > 0.
        if not near:
            return super().command(data, plan, t)
        self.near_steps += 1
        self.last_near_t = t
        if self.enabled:
            # The incoming data is the robot-only pre-contact view. The simulated
            # projectile is reconstructed from the sensor fit in this private copy.
            view = self.robot_with_estimate
            mujoco.mj_copyData(view, self.model, data)
            view.joint("flight").qpos[:3] = p
            view.joint("flight").qvel[:3] = v
            mujoco.mj_forward(self.model, view)
            super().command(view, plan, t)
            data.ctrl[self.ids] = view.ctrl[self.ids]
        else:
            super().command(data, plan, t)
        self.near_times_ms.append(1000*(time.perf_counter()-start))
