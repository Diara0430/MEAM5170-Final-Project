"""下降中的物体质心到达固定水平面的预测时间。使用世界坐标和 SI 单位。"""
from math import sqrt


def remaining_flight_time(z, vz, plane_z, gz):
    """求 z + vz*t + gz*t²/2 = plane_z 的下降交点，本实验的 gz 小于零。"""
    height = z - plane_z
    if height <= 0:
        return 0.
    if gz >= 0:
        raise ValueError("Expected downward gravity.")
    # 有理化正根，避免下降速度较大时的减法消去。
    return 2 * height / (sqrt(vz * vz - 2 * gz * height) - vz)
