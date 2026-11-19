"""40° 快速长方体的接触冲量实验。运行 python -m experiments.contact_impulse --output-dir <新目录>。"""
import argparse
import csv
from pathlib import Path
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.dynamic_catch import run


def components(name, vector):
    return {f"{name}_{axis}": float(value) for axis, value in zip("xyz", vector)}


def rotation(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


class ImpactWindow:
    """所有矢量以世界坐标表示；力矩和角动量均关于物体质心。"""

    def __init__(self, closure_ms):
        self.closure_ms = closure_ms
        self.impulse = np.zeros(3)
        self.angular_impulse = np.zeros(3)
        self.samples = []
        self.contacts = []

    def step(self, model, data, t, first_touch, incoming_twist):
        dt = model.opt.timestep
        # 使两组从各自首次接触开始，均包含 250 ms 的完整积分步。
        if len(self.samples) >= round(.250 / dt):
            return
        body = model.body("projectile")
        geom = model.geom("projectile").id
        flight = data.joint("flight")
        mass = model.body_mass[body.id]
        if not self.samples:
            # 该场景的自由关节原点在质心。模型惯量由主轴系旋至刚体系。
            if np.linalg.norm(body.ipos) > 1e-12:
                raise ValueError("This free-joint momentum calculation requires a centered object.")
            principal = rotation(body.iquat)
            self.inertia = principal @ np.diag(body.inertia) @ principal.T
            self.p0 = mass * incoming_twist[:3]
            self.h0 = data.body("projectile").xmat.reshape(3, 3) @ self.inertia @ incoming_twist[3:]

        force = np.zeros(3)
        torque = np.zeros(3)
        center = data.body("projectile").xipos
        for i, contact in enumerate(data.contact):
            if geom not in (contact.geom1, contact.geom2) or contact.efc_address < 0:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            # mj_contactForce 的方向作用于 geom2；转换成作用于物体的世界系力。
            sign = 1 if geom == contact.geom2 else -1
            basis = sign * contact.frame.reshape(3, 3).T
            normal = basis[:, 0] * wrench[0]
            tangent = basis[:, 1:] @ wrench[1:3]
            arm = contact.pos - center
            moment = np.cross(arm, normal + tangent) + basis @ wrench[3:]
            force += normal + tangent
            torque += moment
            other = contact.geom1 if sign == 1 else contact.geom2
            self.contacts.append(dict(closure_ms=self.closure_ms, step_start_s=t, contact_id=i,
                other_body=model.body(model.geom_bodyid[other]).name, normal_load_N=wrench[0],
                **components("normal_N", normal), **components("tangent_N", tangent),
                **components("lever_m", arm), **components("moment_Nm", moment)))

        # 重力通过质心，仅计入线冲量。自由飞行的步也要累计重力。
        self.impulse += dt * (force + mass * model.opt.gravity)
        self.angular_impulse += dt * torque
        # qpos、qvel 已在步末；接触几何对应步初。必须以步末四元数旋转角动量。
        p = mass * flight.qvel[:3]
        h = rotation(flight.qpos[3:]) @ self.inertia @ flight.qvel[3:]
        dp, dh = p - self.p0, h - self.h0
        self.samples.append(dict(closure_ms=self.closure_ms, first_hand_s=first_touch,
            step_start_s=t, elapsed_end_s=(len(self.samples) + 1) * dt,
            speed_m_s=np.linalg.norm(flight.qvel[:3]), spin_rad_s=np.linalg.norm(flight.qvel[3:]),
            **components("contact_N", force), **components("torque_Nm", torque),
            **components("net_impulse_Ns", self.impulse), **components("delta_p_Ns", dp),
            **components("angular_impulse_Nms", self.angular_impulse), **components("delta_H_Nms", dh),
            p_residual_Ns=np.linalg.norm(dp - self.impulse),
            H_residual_Nms=np.linalg.norm(dh - self.angular_impulse)))


def write_rows(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="新建目录保存两组对照数据")
    args = parser.parse_args()
    windows, totals = [], []
    for duration in (190, 220):
        window = ImpactWindow(duration)
        passed = run("early", .04, duration / 1000, True, shape="box", pitch_deg=40,
                     palm_x_mm=10, arrival_speed="fast", after_contact_step=window.step)
        if not window.samples or not np.isclose(window.samples[-1]["elapsed_end_s"], .250):
            raise RuntimeError("The complete 250 ms impact window was not recorded.")
        totals.append(dict(passed=int(passed), **window.samples[-1]))
        windows.append(window)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_rows(args.output_dir / "balance.csv", totals)
    write_rows(args.output_dir / "time_series.csv", [r for w in windows for r in w.samples])
    write_rows(args.output_dir / "contact_points.csv", [r for w in windows for r in w.contacts])
