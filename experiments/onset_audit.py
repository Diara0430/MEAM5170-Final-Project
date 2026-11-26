"""Replay archived throws and compare causal collision forecasts with measured contact."""
import argparse
from functools import partial
from pathlib import Path

import mujoco
import numpy as np

from controllers.deadline_arm_filter import DeadlineArmFilter
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from scene_setup.scene import build_scene


class OnsetAudit:
    def __init__(self, model):
        self.probe = mujoco.MjData(model)
        self.ball = model.geom("projectile").id
        self.hand = {i for i, b in enumerate(model.geom_bodyid)
                     if (model.body(b).name or "").startswith("hand/")}
        self.palm = model.body("hand/palm_lower").id
        self.jac = np.empty((3, model.nv))
        self.prev_v = None
        self.rows = []

    def forecast(self, model, data, p, v):
        # Hold the last already applied actuator setpoints over a 30 ms lookahead.
        probe = self.probe
        mujoco.mj_copyData(probe, model, data)
        probe.joint("flight").qpos[:] = [*p, 1., 0., 0., 0.]
        probe.joint("flight").qvel[:] = [*v, 0., 0., 0.]
        mujoco.mj_forward(model, probe)
        for _ in range(round(.030/model.opt.timestep)):
            t = float(probe.time)
            mujoco.mj_step(model, probe)
            if any((c.geom1 == self.ball and c.geom2 in self.hand)
                   or (c.geom2 == self.ball and c.geom1 in self.hand) for c in probe.contact):
                return t
        return None

    def record(self, model, data, info, observer, tracker):
        t = float(data.time)
        dt = model.opt.timestep
        velocity = data.qvel[tracker.dofs].copy()
        prior, self.prev_v = self.prev_v, velocity
        estimate = observer.estimate
        if prior is None or estimate is None:
            return
        p, v = estimate.at(t)
        center = capture_center(data)
        contact = info["first_contact_s"]
        if (contact is None and np.linalg.norm(p-center) > .20) or (contact is not None and t > contact+.0221):
            return
        forecast = self.forecast(model, data, p, v) if contact is None else None
        mujoco.mj_jac(model, data, self.jac, None, center, self.palm)
        palm_v = self.jac @ data.qvel
        plan = info.get("applied_reference")
        acceleration = (velocity-prior)/dt
        self.rows.append(dict(frame_s=t, accel_step_s=t-dt, predicted_touch_s=forecast,
                              actual_touch_s=contact, planned_intercept_s=None if plan is None else plan.intercept_time,
                              palm_along_m_s=float(palm_v @ v / np.linalg.norm(v)),
                              **{f"a{i+1}": float(x) for i, x in enumerate(acceleration)}))

    def summary(self, contact):
        out = dict(forecast_s=None, actual_minus_forecast_ms=None, actual_minus_plan_ms=None,
                   palm_10ms_m_s=None, first_joints="", first_breach_ms=None, peak_joint=None, peak_ms=None)
        if contact is None or not self.rows:
            return out
        before = [r for r in self.rows if abs(r["frame_s"]-(contact-.010)) < 1e-7
                  and r["actual_touch_s"] is None]
        if before:
            row = before[0]
            predicted, plan = row["predicted_touch_s"], row["planned_intercept_s"]
            out.update(forecast_s=predicted, actual_minus_forecast_ms=None if predicted is None else 1000*(contact-predicted),
                       actual_minus_plan_ms=None if plan is None else 1000*(contact-plan),
                       palm_10ms_m_s=row["palm_along_m_s"])
        window = [r for r in self.rows if contact-1e-8 <= r["accel_step_s"] < contact+.020-1e-8]
        if window:
            a = np.abs([[r[f"a{i}"] for i in range(1,8)] for r in window])
            t, j = np.unravel_index(np.argmax(a), a.shape)
            out.update(peak_joint=int(j+1), peak_ms=1000*(window[t]["accel_step_s"]-contact))
            bad = np.flatnonzero(np.max(a, axis=1)>4.)
            if bad.size:
                i = bad[0]
                out.update(first_joints="/".join(str(j+1) for j in np.flatnonzero(a[i]>4.)),
                           first_breach_ms=1000*(window[i]["accel_step_s"]-contact))
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows, frames = [], []
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        audit = OnsetAudit(model)
        result = trial(model, args.seed, case, p, v, True,
                       tracker_factory=partial(DeadlineArmFilter, fast=True, refine=True), record_frame=audit.record)
        rows.append({**result, **audit.summary(result["first_contact_s"])})
        frames.extend(dict(seed=args.seed, scenario=case, **r) for r in audit.rows)
    save_csv(args.output_dir/"audit.csv", rows)
    if frames:
        save_csv(args.output_dir/"frames.csv", frames)


if __name__ == "__main__":
    main()
