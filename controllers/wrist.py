"""由掌心相对漂移调节倾腕轨迹的进度速度，所有量使用 SI 单位。"""
import numpy as np


class WristProgress:
    max_duration = 4.0

    def __init__(self, mode):
        if mode not in ("fixed", "slow", "adaptive"):
            raise ValueError("Unknown wrist mode")
        self.mode = mode
        self.completed_at = None
        self.reference = None
        self.previous_radius = 0.0
        self.radial_speed = 0.0
        self.progress = 0.0
        self.rate = 0.5 if mode == "slow" else 1.0
        self.risk_time = 0.0
        self.slow_time = 0.0
        self.trace = []

    def advance(self, model, data, t, tilt_start):
        if t < tilt_start:
            return 0.0
        dt = model.opt.timestep
        palm = data.body("hand/palm_lower")
        relative = palm.xmat.reshape(3, 3).T @ (data.body("projectile").xpos - palm.xpos)
        if self.reference is None:
            self.reference = relative.copy()
        radius = float(np.linalg.norm(relative - self.reference))
        # 对径向漂移速度作 20 ms 低通，减少接触切换时的单步抖动。
        raw_speed = (radius - self.previous_radius) / dt
        self.radial_speed += dt / (.020 + dt) * (raw_speed - self.radial_speed)
        self.previous_radius = radius
        if self.completed_at is None:
            if self.mode == "adaptive":
                # 漂移超过 5 mm 且仍以 10 mm/s 向外运动，持续 10 ms 才减速。
                risk = radius > .005 and self.radial_speed > .010
                self.risk_time = self.risk_time + dt if risk else 0.0
                target_rate = .25 if self.risk_time >= .010 else 1.0
                # 减速比恢复更快，避免速度命令突然变化。
                limit = (10.0 if target_rate < self.rate else 2.0) * dt
                self.rate += float(np.clip(target_rate - self.rate, -limit, limit))
            if self.rate < .999:
                self.slow_time += dt
            # 与原始轨迹相同，从 t=tilt_start 的零进度开始。
            if t > tilt_start + 1e-12:
                self.progress = min(1.0, self.progress + dt * self.rate)
            if self.progress >= 1.0 - 1e-12:
                self.progress = 1.0
                self.completed_at = t
        self.trace.append(dict(time_s=t-tilt_start, drift_mm=1000*radius,
                               radial_speed_mm_s=1000*self.radial_speed,
                               progress=self.progress, rate=self.rate))
        return self.progress
