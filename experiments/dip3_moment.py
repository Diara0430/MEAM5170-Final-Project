"""DIP 3 指令固定为 220 ms，分解两次碰撞的投影角冲量。

python -m experiments.dip3_moment --output-dir results/dip3_moment
"""
import argparse
from itertools import permutations
from pathlib import Path

import numpy as np

from experiments.contact_impulse import ImpactWindow, write_rows
from experiments.dynamic_catch import run
from experiments.link_angular_impulse import vector


def decompose(other_ms):
    window = ImpactWindow(other_ms)
    caught = run("early", .04, other_ms / 1000, True, shape="box", pitch_deg=40,
                 palm_x_mm=10, arrival_speed="fast", dip3_duration=.22,
                 after_contact_step=window.step)
    reference = next(r for r in window.samples if np.isclose(r["elapsed_end_s"], .020))
    h = vector(reference, "delta_H_Nms")  # 发射自旋为零，因此此处就是 20 ms 时的角动量。
    axis = h / np.linalg.norm(h)
    dt = window.samples[0]["elapsed_end_s"]
    points = []
    for row in window.contacts:
        elapsed = round(row["step_start_s"] - reference["first_hand_s"], 9)
        if row["other_body"] != "hand/dip_3" or not .020 <= elapsed < .050:
            continue
        arm = vector(row, "lever_m")
        normal = vector(row, "normal_N")
        tangent = vector(row, "tangent_N")
        for component, force in (("normal", normal), ("tangential", tangent), ("resultant", normal + tangent)):
            mag = np.linalg.norm(force)
            if mag == 0.:
                continue
            torque = np.cross(arm, force)
            points.append(dict(other_ms=other_ms, dip3_ms=220, elapsed_s=elapsed,
                contact_id=row["contact_id"], component=component, force_N=mag,
                radius_mm=1000*np.linalg.norm(arm), perpendicular_arm_mm=1000*np.linalg.norm(torque)/mag,
                torque_mNm=1000*np.linalg.norm(torque), projected_torque_mNm=1000*(axis @ torque),
                **{k: row[k] for k in row if k.startswith(("normal_N_", "tangent_N_", "lever_m_"))}))

    summaries = []
    for component in ("normal", "tangential", "resultant"):
        group = [r for r in points if r["component"] == component]
        # J = P * B * C。P 为接触点力的大小积分，B 为力加权垂直力臂，
        # C 为力矩大小加权的有符号方向余弦；各量都用同一窗口的全部接触点计算。
        load = sum(r["force_N"] for r in group) * dt
        unsigned = sum(r["torque_mNm"] for r in group) * dt
        projected = sum(r["projected_torque_mNm"] for r in group) * dt
        active_s = len({r["elapsed_s"] for r in group}) * dt
        summaries.append(dict(other_ms=other_ms, dip3_ms=220, caught=int(caught), component=component,
            active_ms=active_s*1000, mean_active_load_N=load/active_s,
            load_Ns=load, perpendicular_arm_mm=unsigned/load, direction_cos=projected/unsigned,
            weighted_radius_mm=sum(r["radius_mm"]*r["force_N"] for r in group)*dt/load,
            projected_mNms=projected, axis_x=axis[0], axis_y=axis[1], axis_z=axis[2]))
    np.testing.assert_allclose(sum(r["projected_mNms"] for r in summaries[:2]), summaries[2]["projected_mNms"])
    return summaries, points


def factor_changes(old, new):
    # 三因素乘积存在交互；平均六种替换顺序，把交互对称分摊，三项严格加总。
    keys = ("load_Ns", "perpendicular_arm_mm", "direction_cos")
    changes = np.zeros(3)
    for order in permutations(range(3)):
        values = np.array([old[k] for k in keys])
        for i in order:
            before = np.prod(values)
            values[i] = new[keys[i]]
            changes[i] += (np.prod(values) - before) / 6
    delta = new["projected_mNms"] - old["projected_mNms"]
    np.testing.assert_allclose(sum(changes), delta, atol=1e-10)
    return dict(component=new["component"], delta_mNms=delta, load_change_mNms=changes[0],
                arm_change_mNms=changes[1], direction_change_mNms=changes[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    old, old_points = decompose(220)
    new, new_points = decompose(190)
    changes = [factor_changes(a, b) for a, b in zip(old, new)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_rows(args.output_dir / "summary.csv", old + new)
    write_rows(args.output_dir / "changes.csv", changes)
    write_rows(args.output_dir / "contacts.csv", old_points + new_points)
