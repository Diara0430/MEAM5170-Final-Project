"""竖直外力脉冲下的闭指时机对照。每个随机种子的两个控制器共用同一外力。"""
import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controllers.arrival import remaining_flight_time
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run


class TimingTrial:
    def __init__(self, pulse, online):
        self.onset, self.length, self.az = pulse
        self.online = online
        self.crossing = None
        self.prediction_at_close = None
        self.outcome = {}

    def before_step(self, model, data, flight, target, t, nominal, waiting):
        # 外力只由试验环境施加。预测器的输入只有当前高度、竖直速度、目标高度和重力。
        body = model.body("projectile")
        active = self.onset <= t + 1e-12 < self.onset + self.length
        data.xfrc_applied[body.id, 2] = body.mass[0] * self.az if active else 0.
        if self.online and waiting:
            return t + remaining_flight_time(flight.qpos[2], flight.qvel[2], target[2], model.opt.gravity[2])
        return nominal

    def observe_crossing(self, t, dt, z0, z1, target_z):
        # 两步之间线性插值。只有首次接触前穿越目标高度才用于评价预测误差。
        if self.crossing is None and z0 >= target_z >= z1 and z0 > z1:
            self.crossing = t + dt * (z0 - target_z) / (z0 - z1)

    def result(self, seed):
        error = None
        if self.crossing is not None and self.prediction_at_close is not None:
            error = 1000 * (self.prediction_at_close - self.crossing)
        contact = self.outcome["first_contact_s"]
        close = self.outcome["close_s"]
        return dict(seed=seed, controller="online" if self.online else "fixed",
                    gust_onset_s=self.onset, gust_duration_s=self.length, gust_az_m_s2=self.az,
                    prediction_at_close_s=self.prediction_at_close, plane_crossing_s=self.crossing,
                    prediction_error_ms=error, contact_minus_close_ms=None if contact is None or close is None
                    else 1000 * (contact - close), **self.outcome)


def gust_for_seed(seed):
    rng = np.random.default_rng(seed)
    onset, length = .002 * rng.integers([20, 15], [46, 31])
    return onset, length, rng.choice([-1, 1]) * rng.uniform(2., 6.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    # 无扰动验算 + 24 对随机试验。工况在比较前固定，按 2 ms 仿真时间步施力。
    for seed in range(-1, 24):
        pulse = (0., 0., 0.)
        if seed >= 0:
            pulse = gust_for_seed(seed)
        for online in (False, True):
            trial = TimingTrial(pulse, online)
            run("early", .04, .19, True, shape="box", pitch_deg=40., palm_x_mm=10.,
                arrival_speed="fast", timing_trial=trial)
            rows.append(trial.result(seed))
    write_rows(args.output, rows)
