"""Shape-dependent thumb aperture before contact; optional loaded-finger slip feedback."""
import numpy as np

from controllers.catch_stability import SlipGrip
from controllers.sphere_cage import SphereCageClosure


class BoxGrasp(SphereCageClosure):
    def __init__(self, model, hand, observer, *, slip_feedback=True):
        super().__init__(model, hand)
        self.track = observer.rotation_track
        self.half_size = model.geom("projectile").size.copy()
        self.width = None
        self.slip = SlipGrip() if slip_feedback else None

    def prepare(self, model, data, now, nominal):
        # Freeze once, at the very start of the smooth thumb preshape trajectory.
        start = self.pre_start if self.pre_start is not None else nominal-self.cage_config.preshape_lead
        if self.width is None and now >= start-1e-9:
            rotation = self.track.at(nominal)
            if rotation is not None:
                span = data.site("hand/th_tip").xpos - data.site("hand/tip_2").xpos
                norm = np.linalg.norm(span)
                if norm > 1e-6:
                    self.width = 2*np.abs(rotation.inv().apply(span/norm)) @ self.half_size
                    scale = np.clip(1.-2.5*(self.width-.050), .90, 1.08)
                    self.prepose[1:] *= scale
                    self.index_goal[1:] *= scale
        super().prepare(model, data, now, nominal)

    def command(self, model, data, elapsed):
        super().command(model, data, elapsed)
        # Only after the 300 ms settling period. The existing feedback adds at most
        # 0.12 rad to loaded fingers, under its original contact and load tests.
        if self.slip is not None and elapsed >= .30:
            self.slip(model, data, self.hand, elapsed, .30)
