"""按接触侧调节 LEAP 四根手指的闭合进度。所有输入来自已经完成的物理步。"""
from itertools import combinations

import mujoco
import numpy as np


class ContactClosure:
    def __init__(self, model, hand, max_delay_s=None):
        self.dt = model.opt.timestep
        if max_delay_s is not None and (not np.isfinite(max_delay_s) or max_delay_s < 0):
            raise ValueError("Maximum finger delay must be finite and nonnegative.")
        self.max_delay_s = np.inf if max_delay_s is None else max_delay_s
        self.projectile = model.geom("projectile").id
        self.palm = model.body("hand/palm_lower").id
        # LEAP 的数值关节编号按四根手指分组；由运动树找到每个碰撞体所属手指。
        joint_groups = {model.actuator_trnid[a, 0]: i // 4 for i, a in enumerate(hand.ids)}
        self.group = {}
        for geom, body in enumerate(model.geom_bodyid):
            while body != self.palm and body != 0:
                if model.body_jntnum[body] and model.body_jntadr[body] in joint_groups:
                    self.group[geom] = joint_groups[model.body_jntadr[body]]
                    break
                body = model.body_parentid[body]
            else:
                if body == self.palm:
                    self.group[geom] = -1  # 掌面可提供对侧支撑，但没有闭指指令。
        self.loaded = set()
        self.opposed_time = 0.
        self.holding = False
        self.delay = np.zeros(4)

    def observe(self, model, data):
        # 只计入有压缩力的手-物接触。法向统一为指向物体的方向。
        self.loaded.clear()
        contacts = []
        center = data.body("projectile").xipos
        for i, c in enumerate(data.contact):
            if self.projectile not in (c.geom1, c.geom2) or c.efc_address < 0:
                continue
            on_second = c.geom2 == self.projectile
            group = self.group.get(c.geom1 if on_second else c.geom2)
            if group is None:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            if wrench[0] < .02:  # N；忽略几乎不承载的接触点。
                continue
            self.loaded.add(group)
            normal = c.frame[:3] if on_second else -c.frame[:3]
            contacts.append((group, c.pos - center, normal))

        # 不同指（或指与掌）在质心两侧，且法向夹角大于 120°，才算对向支撑。
        opposed = any(a != b and ra @ rb < 0. and na @ nb < -.5
                      for (a, ra, na), (b, rb, nb) in combinations(contacts, 2))
        self.opposed_time = self.opposed_time + self.dt if opposed else 0.
        self.holding = self.opposed_time + 1e-12 >= .006

    def fraction(self, elapsed, durations):
        # 单侧承载的指以一半速度闭合，其余指保持原速。双侧稳定后各指恢复原速并保持终点。
        # 累积延迟以保证目标连续；首次闭指时从原来的零进度开始。
        if not self.holding and elapsed > 0:
            for group in self.loaded:
                if group >= 0:
                    # 每根手指独立计时。达到上限后恢复原速，保持目标连续。
                    self.delay[group] = min(self.delay[group] + .5 * self.dt, self.max_delay_s)
        return np.maximum(elapsed - np.repeat(self.delay, 4), 0.) / durations
