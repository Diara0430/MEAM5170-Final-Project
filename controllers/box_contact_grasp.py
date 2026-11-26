"""Time-limited LEAP flexion changes from delayed arrival estimates and completed tactile steps."""
import numpy as np

from controllers.sphere_cage import SphereCageClosure


class ContactGrasp(SphereCageClosure):
    def __init__(self, model, hand, policy, *, shadow=False):
        super().__init__(model, hand)
        if policy not in ("tactile", "lead", "lead_tactile", "recapture"):
            raise ValueError("Unknown contact policy")
        self.shadow = shadow
        self.delivered_steps = 0
        self.lead = policy in ("lead","lead_tactile")
        self.tactile = policy in ("tactile","lead_tactile")
        self.recapture = policy == "recapture"
        self.dt = model.opt.timestep
        # Three digit flexors each. Keep base abduction (0, 4, 8, 12) under the original controller.
        self.flex = np.array([[1,2,3],[5,6,7],[9,10,11],[13,14,15]])
        self.ids_flex = np.asarray(hand.ids)[self.flex]
        self.stiffness = model.actuator_gainprm[self.ids_flex,0]
        self.weights = np.array([[.4,.8,.6],[.8,1.,.7],[.8,1.,.7],[.4,.7,.5]])
        self.bias = np.zeros(4)
        self.before_frames = self.after_frames = self.yield_frames = 0
        self.max_torque_change = 0.
        self.first_support = False
        self.was_opposed = False
        self.assist_until = -1.
        self.recapture_frames = 0

    def _move(self, target, rate):
        self.bias += np.clip(target-self.bias, -rate*self.dt, rate*self.dt)

    def _apply(self, data):
        ids = self.ids_flex
        original = data.ctrl[ids].copy()
        correction = np.clip(self.bias[:,None]*self.weights, -.45/self.stiffness, .45/self.stiffness)
        command = np.clip(original+correction, self.hand.limits[self.flex,0], self.hand.limits[self.flex,1])
        if not self.shadow:
            data.ctrl[ids] = command
        amount = float(np.max(np.abs((command-original)*self.stiffness)))
        self.max_torque_change = max(self.max_torque_change, amount)
        changed = amount > 1e-7
        self.delivered_steps += int(changed and not self.shadow)
        return changed

    def prepare(self, model, data, now, nominal):
        super().prepare(model, data, now, nominal)
        if not self.lead:
            return
        # The crossing time comes from the delivered camera fits. Smooth out updates
        # when a new crossing estimate arrives and never close all the way in flight.
        u = float(np.clip((now-nominal+.095)/.080,0.,1.))
        desired = .10 * u*u*(3.-2.*u) if now <= nominal+.040 else 0.
        self._move(desired, 1.8)
        if np.any(self.bias):
            self.before_frames += int(self._apply(data))

    def command(self, model, data, elapsed):
        super().command(model, data, elapsed)
        if self.support.holding:
            self.first_support = True
        if self.recapture and self.was_opposed and not self.support.holding and elapsed <= .120:
            self.assist_until = elapsed+.040
        self.was_opposed = self.support.holding
        loaded = self.support.loaded
        if self.recapture and not self.support.holding and elapsed < self.assist_until:
            target = np.array([.025 if i in loaded else .10 for i in range(4)])
            self.recapture_frames += 1
        elif self.tactile and elapsed <= .090 and not self.first_support and loaded:
            # A palm-only hit allows the four free digits to advance. Previously
            # loaded digits yield slightly while their opposite side closes.
            target = np.array([-.055 if i in loaded else .10 for i in range(4)])
            self.yield_frames += int(any(i in loaded for i in range(4)))
        else:
            target = 0.
        self._move(target, 2.)
        if np.any(self.bias):
            self.after_frames += int(self._apply(data))

    def diagnostics(self):
        return dict(hand_changes_applied=self.delivered_steps, lead_steps=self.before_frames, tactile_steps=self.after_frames,
                    recapture_steps=self.recapture_frames, yielding_steps=self.yield_frames, max_flex_torque_delta_Nm=self.max_torque_change)
