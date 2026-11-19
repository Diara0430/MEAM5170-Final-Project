"""单关节对照。运行 python -m experiments.dip3_timing --output <新的 CSV 路径>。"""
import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.contact_impulse import ImpactWindow, write_rows
from experiments.dynamic_catch import run
from experiments.link_angular_impulse import analyze


def trial(dip_ms, other_ms=220):
    window = ImpactWindow(other_ms)
    joint_state = {}

    def observe(model, data, t, first_touch, twist):
        window.step(model, data, t, first_touch, twist)
        # 控制器读取步初时钟，位置是积分后的步末值；均与 50 ms 角速度对齐。
        if np.isclose(t + model.opt.timestep - first_touch, .050, rtol=0, atol=1e-10):
            joint_state["command_rad"] = float(data.actuator("hand/10").ctrl[0])
            joint_state["position_rad"] = float(data.joint("hand/10").qpos[0])

    passed = run("early", .04, other_ms / 1000, True, shape="box", pitch_deg=40., palm_x_mm=10.,
                 arrival_speed="fast", after_contact_step=observe, dip3_duration=dip_ms / 1000)
    # 沿各次试验在 20 ms 的角动量方向投影，时间窗和归一化方式与前一轮一致。
    links = {r["link"]: r for r in analyze(str(other_ms), window.samples, window.contacts)}
    samples = {round(r["elapsed_end_s"], 9): r for r in window.samples}
    net = links["Net"]
    return dict(pitch_deg=40, palm_x_mm=10, arrival_speed="fast", lead_ms=40,
                other_closure_ms=other_ms, dip3_closure_ms=dip_ms, caught=int(passed),
                first_touch_s=samples[.05]["first_hand_s"],
                dip3_command_rad_at_50ms=joint_state["command_rad"],
                dip3_position_rad_at_50ms=joint_state["position_rad"],
                spin_rad_s_at_20ms=samples[.02]["spin_rad_s"],
                spin_rad_s_at_50ms=samples[.05]["spin_rad_s"],
                dip3_impulse_mNms=links.get("dip_3", {}).get("projected_mNms", 0.),
                net_impulse_mNms=net["projected_mNms"],
                state_delta_H_mNms=net["state_delta_H_projected_mNms"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [trial(ms) for ms in range(160, 281, 15)]
    # 反向对照：用其余关节复现成功工况，再单独把 DIP 3 恢复到 220 ms。
    rows.extend(trial(ms, other_ms=190) for ms in (190, 220))
    write_rows(args.output, rows)
