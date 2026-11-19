"""分段接近盒子对侧面；每段都从实际关节姿态和物体位姿重算。"""
import mujoco

from controllers.thumb_face import ThumbFaceStep


class ThumbFaceFeedback(ThumbFaceStep):
    def __init__(self):
        super().__init__("ik")
        self.segment = 0
        self.hold_time = None
        self.frozen = False

    def __call__(self, model, data, hand, t, tilt):
        ids = hand.ids[12:]
        if self.start is None:
            if t + .160 > tilt:
                return
            self.start = t
            self.plan(model, data, hand)
            if self.face is not None:
                self.scratch = mujoco.MjData(model)
                self.segment = 1
                self.segment_start = t
                self.duration = .160
                self.initial_command = data.ctrl[ids].copy()
                self.command = self.initial_command.copy()
                self.target = self.ik_target
        if self.face is None:
            return
        if not self.frozen:
            # 接触力来自上一物理步。达到阈值后保持刚才实际发出的目标，避免额外挤压。
            if self.contact_loads(model,data)[1] >= .02:
                self.hold_time = t
                self.frozen = True
            else:
                fraction = min((t-self.segment_start)/self.duration, 1.)
                self.command = self.initial_command + fraction**2*(3-2*fraction)*(self.target-self.initial_command)
                if t >= tilt-1e-12:
                    self.frozen = True
                elif t + 1e-12 >= self.segment_start + self.duration:
                    self.scratch.qpos[:] = data.qpos
                    mujoco.mj_kinematics(model,self.scratch)
                    mujoco.mj_comPos(model,self.scratch)
                    self.target, _ = self.local_step(model,self.scratch)
                    self.initial_command = self.command.copy()
                    self.segment_start = t
                    self.duration = min(.050,tilt-t)
                    self.segment += 1
        data.ctrl[ids] = self.command
