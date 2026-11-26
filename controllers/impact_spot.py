"""Choose an interior ball contact location using a directional impulse model."""
import mujoco
import numpy as np


class ImpactSpot:
    def __init__(self, model, observer, *, shadow=False):
        self.model, self.observer, self.shadow = model, observer, shadow
        self.probe = mujoco.MjData(model)
        self.mass = np.empty((model.nv,model.nv))
        self.jac = np.empty((3,model.nv))
        self.palm = model.body("hand/palm_lower").id
        self.ball_mass = model.body_mass[model.body("projectile").id]
        joints = [model.joint(f"joint{i}") for i in range(1,8)]
        self.qadr = [j.qposadr[0] for j in joints]
        self.dofs = [j.dofadr[0] for j in joints]
        self.local_center = np.array([0.,-.055,-.08])
        self.decided = False
        self.arm_target_offset = None
        self.deliveries = 0
        self.center_jump = self.best_jump = None

    def contact_offset(self, data, previous, now):
        if self.decided:
            return None if self.shadow else self.arm_target_offset
        stamp = previous.intercept_time
        if not .18 <= stamp-now <= .55:
            return None
        self.decided = True
        m, d = self.model, self.probe
        mujoco.mj_copyData(d,m,data)
        q, vel, _ = previous.reference(stamp)
        d.qpos[self.qadr] = q
        mujoco.mj_fwdPosition(m,d)
        mujoco.mj_fullM(m,d,self.mass)
        arm_mass = self.mass[np.ix_(self.dofs,self.dofs)]
        palm = d.body(self.palm)
        rotation = palm.xmat.reshape(3,3)
        center = palm.xpos + rotation @ self.local_center
        normal = rotation[:,2]
        _, ball_v = self.observer.estimate.at(stamp)
        theta = np.arange(8)*np.pi/4
        offsets = np.zeros((9,3))
        offsets[1:] = .006*(np.outer(np.cos(theta),rotation[:,0])
                             + np.outer(np.sin(theta),rotation[:,1]))
        best = 0
        for i,delta in enumerate(offsets):
            mujoco.mj_jac(m,d,self.jac,None,center+delta,self.palm)
            jn = normal @ self.jac[:,self.dofs]
            response = np.linalg.solve(arm_mass, jn)
            speed = max(0.,float(normal@ball_v - jn@vel))
            impulse = speed / (1/self.ball_mass + jn@response)
            jump = float(np.max(np.abs(response))*impulse)
            if i==0:
                self.center_jump = self.best_jump = jump
                center_impulse = impulse
            elif impulse<=1.01*center_impulse and jump<self.best_jump:
                best, self.best_jump = i, jump
        if best and self.best_jump < .98*self.center_jump:
            # The chosen ball contact lies at +delta in the palm tangent plane.
            # Move the target palm center in the opposite direction.
            self.arm_target_offset = -offsets[best]
        return None if self.shadow else self.arm_target_offset

    def choose(self, data, now, first, second, deadline):
        return first

    def applied(self, plan):
        self.deliveries += int(getattr(plan,"impact_offset",False))

    def summary(self):
        return dict(spot_assessed=int(self.decided),spot_proposed=int(self.arm_target_offset is not None),
                    spot_deliveries=self.deliveries,
                    center_joint_jump=self.center_jump, chosen_joint_jump=self.best_jump)
