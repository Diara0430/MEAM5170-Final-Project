"""Compare ball-derived rolling controllers on identical rotating cuboids and throws."""
import argparse
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from controllers.box_rotation import BoxExtent, RotationTrack, as_wxyz, from_wxyz
from controllers.catch_approach import CatchApproach
from controllers.fixed_point_trim import FixedPointTrim
from controllers.impact_spot import ImpactSpot
from controllers.intercept_impact import InterceptImpact
from controllers.short_impact import ShortImpact
from controllers.velocity_selection import VelocitySelection
from experiments.contact_joint_replay import ContactTrace
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.rolling_reach import compare
from experiments.rolling_success import StudyTrack
from scene_setup.scene import build_scene


METHODS = ("center", "pose", "extent", "single", "early_lock", "nonlinear", "stop",
           "trim", "short", "spot", "inertia", "preview", "rank", "prepare")


class BoxObserver(StudyTrack):
    def __init__(self, *args, seed, case, profile, use_pose, **kwargs):
        super().__init__(*args, seed=seed, case=case, profile=profile, **kwargs)
        rng = np.random.default_rng(np.random.SeedSequence([seed, case, 110]))
        self.rotation_track = RotationTrack(rng, self.latency_s, 1. if profile == "normal" else 2.)
        self.preview_rotation = self.rotation_track if use_pose else None


def box_launch(seed, case):
    rng = np.random.default_rng(np.random.SeedSequence([seed, case, 420]))
    attitude = Rotation.random(random_state=rng)
    axis = rng.normal(size=3)
    spin = (0., .8, 2., 3.)[(case//2 + seed) % 4]
    return as_wxyz(attitude), spin*axis/np.linalg.norm(axis)


class ShapeAudit(ContactTrace):
    def __init__(self, model):
        super().__init__(model)
        self.history = deque(maxlen=60)
        self.error = self.future_error = self.position_error = None
        self.hold_reference = None
        self.max_rotation = None
        self.observed_contact = False
        self.travel = 0.
        self.home = None

    def record(self, model, data, info, observer, tracker):
        super().record(model, data, info, observer, tracker)
        t, onset = float(data.time), info["first_contact_s"]
        center = capture_center(data)
        if self.home is None:
            self.home = center
        if onset is None:
            self.travel = max(self.travel, np.linalg.norm(center-self.home))
            # Log near 50 Hz. Save a strictly pre-contact estimate for the 100 ms forecast.
            if round(t/model.opt.timestep) % round(.020/model.opt.timestep) == 1:
                r = observer.rotation_track.at(t+.100)
                if r is not None and observer.estimate is not None:
                    self.history.append((t+.100, r, observer.estimate.at(t+.100)[0]))
            return
        actual = from_wxyz(data.joint("flight").qpos[3:])
        if not self.observed_contact:
            self.observed_contact = True
            estimated = observer.rotation_track.at(t)
            if estimated is not None:
                self.error = np.rad2deg((actual.inv()*estimated).magnitude())
            if self.history:
                stamp, rot, pos = min(self.history, key=lambda x: abs(x[0]-t))
                if abs(stamp-t) <= .010001:
                    self.future_error = np.rad2deg((actual.inv()*rot).magnitude())
                    self.position_error = 1000*np.linalg.norm(data.joint("flight").qpos[:3]-pos)
        if t >= onset+.3-1e-10:
            palm = from_wxyz(data.body("hand/palm_lower").xquat)
            relative = palm.inv()*actual
            if self.hold_reference is None:
                self.hold_reference, self.max_rotation = relative, 0.
            angle = np.rad2deg((self.hold_reference.inv()*relative).magnitude())
            self.max_rotation = max(self.max_rotation, angle)

    def summary(self, row):
        return dict(**super().summary(row), palm_travel_mm=1000*self.travel,
                    onset_pose_error_deg=self.error, forecast_100ms_error_deg=self.future_error,
                    forecast_100ms_position_mm=self.position_error, hold_rotation_deg=self.max_rotation)


def simulate(model, seed, case, p, v, profile, method):
    so = dict(launch_rotation=box_launch(seed, case))
    if method == "single":
        so["heights"] = (0.,)
    if method in ("nonlinear", "stop"):
        so["approach"] = CatchApproach(fraction=0. if method=="stop" else .30, nonlinear=method=="nonlinear")
    if method in ("rank", "prepare"):
        so["velocity_selection"] = VelocitySelection(prepare_early=method=="prepare")
    selector = {"extent": BoxExtent, "trim": FixedPointTrim, "short": ShortImpact,
                "spot": ImpactSpot, "inertia": partial(InterceptImpact, method="inertia"),
                "preview": partial(InterceptImpact, method="preview")}.get(method)
    sensor = partial(BoxObserver, use_pose=method!="center")
    trace = ShapeAudit(model)
    row = compare(model, seed, case, p, v, profile, "track_reserve", True,
                  record_frame=trace.record, selector_factory=selector, sensor_type=sensor,
                  rule_name="stable4" if method=="early_lock" else "rolling", trial_options=so)
    quat, omega = so["launch_rotation"]
    return dict(**row, **trace.summary(row), shape="box", method=method,
                spin_rad_s=np.linalg.norm(omega), **{f"launch_q{i}": x for i,x in enumerate(quat)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--profile", choices=("normal","noisy","corrected"), required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("Specify each method once")
    model = build_scene("box")
    center = capture_center(initial(model))
    rows = []
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        shift = (args.seed+case) % len(args.methods)
        for method in args.methods[shift:]+args.methods[:shift]:
            rows.append(simulate(model, args.seed, case, p, v, args.profile, method))
    columns = dict.fromkeys(k for r in rows for k in r)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir/"trials.csv", [{**columns, **r} for r in rows])


if __name__ == "__main__":
    main()
