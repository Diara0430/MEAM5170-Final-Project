"""盒子对侧面的一次局部逆运动学调整，关节 12–15；单位为米、秒和弧度。"""
import mujoco
import numpy as np

from controllers.contact_closure import ContactClosure


def bounded_step(jac, error, lower, upper):
    """在关节边界内最小化 ||J dq - e||² + (0.008 m/rad)² ||dq||²。"""
    hessian = jac.T @ jac + .008**2 * np.eye(4)
    rhs = jac.T @ error
    dq = np.zeros(4)
    for _ in range(200):
        prev = dq.copy()
        for i in range(4):
            dq[i] = np.clip(dq[i] + (rhs[i] - hessian[i] @ dq) / hessian[i,i], lower[i], upper[i])
        if np.max(np.abs(prev-dq)) < 1e-9:
            break
    return dq


class ThumbFaceStep:
    def __init__(self, mode):
        if mode not in ("fixed","anchor","ik"):
            raise ValueError("Unknown thumb control")
        self.mode = mode
        self.start = None
        self.face = None
        self.reason = "not_triggered"

    def coordinates(self, data):
        rot = data.geom_xmat[self.obj].reshape(3,3)
        point = data.xpos[self.tip] + data.xmat[self.tip].reshape(3,3) @ self.local_pad
        return rot.T @ (point - data.geom_xpos[self.obj])

    def gap(self, data):
        point = self.coordinates(data)
        closest = np.clip(point, -self.size, self.size)
        closest[self.axis] = self.sign * self.size[self.axis]
        return float(np.linalg.norm(point - closest))

    def contact_loads(self, model, data):
        """返回拇指总法向载荷和末节在目标面上的载荷（N）。"""
        total = pad_load = 0.
        rot = data.geom_xmat[self.obj].reshape(3,3)
        for i,c in enumerate(data.contact):
            if self.obj not in (c.geom1,c.geom2) or c.efc_address < 0:
                continue
            other = c.geom2 if c.geom1 == self.obj else c.geom1
            if other not in self.thumb_geoms:
                continue
            f = np.empty(6)
            mujoco.mj_contactForce(model,data,i,f)
            total += max(f[0],0.)
            outward = c.frame[:3] if c.geom1 == self.obj else -c.frame[:3]
            local = rot.T @ (c.pos-data.geom_xpos[self.obj])
            if (other == self.pad_geom and outward @ (self.sign*rot[:,self.axis]) >= .7071 and
                    abs(local[self.axis]-self.sign*self.size[self.axis]) <= .002):
                pad_load += max(f[0],0.)
        return total,pad_load

    def local_step(self, model, scratch, point=None):
        # 用同一个材料点与盒子局部面标号，但每次按最新世界位姿更新雅可比和目标点。
        rotation = scratch.geom_xmat[self.obj].reshape(3,3)
        local = self.coordinates(scratch)
        target = np.clip(local,-.8*self.size,.8*self.size)
        target[self.axis] = self.sign*self.size[self.axis]
        error = rotation @ (target-local)
        error *= min(1., .012/max(np.linalg.norm(error),1e-12))
        if point is None:
            point = scratch.xpos[self.tip] + scratch.xmat[self.tip].reshape(3,3) @ self.local_pad
        jac = np.empty((3,model.nv))
        mujoco.mj_jac(model,scratch,jac,None,point,self.tip)
        measured = scratch.qpos[self.adr].copy()
        anchor = np.clip(measured,self.lo,self.hi)
        dq = bounded_step(jac[:,self.dof],error,np.maximum(self.lo,anchor-.35)-measured,
                          np.minimum(self.hi,anchor+.35)-measured)
        return measured+dq, target

    def plan(self, model, data, hand):
        self.obj = model.geom("projectile").id
        self.tip = model.body("hand/thumb_fingertip").id
        groups = ContactClosure(model, hand).group
        self.pad_geom = next(g for g in groups if model.geom_bodyid[g] == self.tip and model.geom_contype[g])
        self.thumb_geoms = {g for g,group in groups.items() if group == 3}
        self.size = model.geom_size[self.obj].copy()
        # 用上一个已完成物理步的法向力选面，再在当前关节状态计算目标。
        loads = np.zeros((3,2))
        rot = data.geom_xmat[self.obj].reshape(3,3)
        for i,c in enumerate(data.contact):
            if self.obj not in (c.geom1,c.geom2) or c.efc_address < 0:
                continue
            other = c.geom2 if c.geom1 == self.obj else c.geom1
            if groups.get(other) not in (0,1,2):
                continue
            f = np.empty(6)
            mujoco.mj_contactForce(model,data,i,f)
            outward = rot.T @ (c.frame[:3] if c.geom1 == self.obj else -c.frame[:3])
            axis = np.argmax(np.abs(outward))
            loads[axis,int(outward[axis] > 0)] += max(f[0],0.)
        if loads.max() < .02:
            self.reason = "no_loaded_finger_face"
            return
        self.axis, supported_sign = np.unravel_index(loads.argmax(),loads.shape)
        self.sign = 1 if supported_sign == 0 else -1
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        mujoco.mj_kinematics(model,scratch)
        mujoco.mj_comPos(model,scratch)
        rotation = scratch.geom_xmat[self.obj].reshape(3,3)
        normal = self.sign * rotation[:,self.axis]
        center = scratch.geom_xpos[self.obj] + normal * self.size[self.axis]
        # th_tip 是标记点。由面中心方向向标记点发射射线，在指腹网格上固定一个材料点。
        site = scratch.site("hand/th_tip").xpos
        ray = site - center
        ray /= max(np.linalg.norm(ray),1e-12)
        origin = site - .10*ray
        distance = mujoco.mj_rayMesh(model,scratch,self.pad_geom,origin,ray)
        if distance < 0:
            self.reason = "no_pad_intersection"
            return
        point = origin + distance * ray
        self.local_pad = scratch.xmat[self.tip].reshape(3,3).T @ (point-scratch.xpos[self.tip])
        self.face = ("+" if self.sign > 0 else "-") + "xyz"[self.axis]
        self.reason = "planned"
        self.initial_gap = self.gap(scratch)
        jac = np.empty((3,model.nv))
        joints = [model.joint(f"hand/{i}") for i in range(12,16)]
        adr = [j.qposadr[0] for j in joints]
        self.adr = adr
        self.dof = [j.dofadr[0] for j in joints]
        self.measured = scratch.qpos[adr].copy()
        limits = model.jnt_range[[j.id for j in joints]]
        lo = np.maximum(limits[:,0],hand.limits[12:,0])
        hi = np.minimum(limits[:,1],hand.limits[12:,1])
        anchor = np.clip(self.measured,lo,hi)
        self.lo, self.hi = lo, hi
        self.ik_target, target = self.local_step(model,scratch,point)
        self.anchor_target = anchor
        scratch.qpos[adr] = self.ik_target
        mujoco.mj_kinematics(model,scratch)
        self.predicted_gap = self.gap(scratch)
        # 单独诊断静态运动学可达性：固定盒子和手腕，允许多步迭代。
        # 这里的结果不发给执行器，也不包含接触阻挡和力矩饱和。
        self.best_static_gap = min(self.initial_gap, self.predicted_gap)
        goal = scratch.geom_xpos[self.obj] + rotation @ target
        for _ in range(60):
            point = scratch.xpos[self.tip] + scratch.xmat[self.tip].reshape(3,3) @ self.local_pad
            residual = goal-point
            if np.linalg.norm(residual) < .001:
                break
            mujoco.mj_comPos(model,scratch)
            mujoco.mj_jac(model,scratch,jac,None,point,self.tip)
            q = scratch.qpos[adr]
            delta = bounded_step(jac[:,[j.dofadr[0] for j in joints]],residual,
                                 np.maximum(lo-q,-.20),np.minimum(hi-q,.20))
            if max(abs(delta)) < 1e-6:
                break
            scratch.qpos[adr] = q + delta
            mujoco.mj_kinematics(model,scratch)
            self.best_static_gap = min(self.best_static_gap,self.gap(scratch))

    def __call__(self, model, data, hand, t, tilt):
        if self.start is None:
            if t+.16 > tilt:
                return
            self.start = t
            self.plan(model,data,hand)
        if self.face is None or self.mode == "fixed":
            return
        # 从原来的位置命令平滑转入参考目标，全程通过实际执行器施力。
        frac = min((t-self.start)/.16,1.)
        weight = frac**2*(3-2*frac)
        target = self.ik_target if self.mode=="ik" else self.anchor_target
        ids = hand.ids[12:]
        data.ctrl[ids] += weight*(target-data.ctrl[ids])
