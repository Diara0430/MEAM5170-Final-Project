"""Replay catches with a read-only contact and joint audit at each physical step."""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.rolling_reach import compare
from scene_setup.scene import build_scene


class ContactTrace:
    def __init__(self, model):
        self.projectile = model.geom("projectile").id
        self.names = {i: model.body(b).name for i, b in enumerate(model.geom_bodyid)
                      if (model.body(b).name or "").startswith("hand/")}
        self.previous = None
        self.rows = []

    def record(self, model, data, info, observer, tracker):
        dt = model.opt.timestep
        current = data.qvel[tracker.dofs].copy()
        previous, self.previous = self.previous, current
        onset = info["first_contact_s"]
        if previous is None or onset is None:
            return
        step = round((float(data.time) - dt - onset) / dt)
        if not 0 <= step < round(.020 / dt):
            return
        loads = {}
        wrench = np.empty(6)
        for index, c in enumerate(data.contact):
            if self.projectile not in (c.geom1, c.geom2):
                continue
            other = c.geom1 if c.geom2 == self.projectile else c.geom2
            name = self.names.get(other)
            if name is None:
                continue
            loads.setdefault(name, 0.)
            if c.efc_address >= 0:
                mujoco.mj_contactForce(model, data, index, wrench)
                loads[name] += max(0., float(wrench[0]))
        dominant = max(loads, key=loads.get) if loads else ""
        accel = (current - previous) / dt
        self.rows.append(dict(ms=round(1000*step*dt, 5), parts="|".join(sorted(loads)),
                              dominant=dominant, normal_load_N=sum(loads.values()),
                              **{f"a{i}": a for i, a in enumerate(accel, 1)},
                              **{f"q{i}": q for i, q in enumerate(data.qpos[tracker.qadr], 1)}))

    def summary(self, row):
        result = dict(first_parts="", first_dominant="", peak_axis=None, peak_ms=None,
                      breach_axes="", breach_ms=None, breach_dominant="", trace_peak=None)
        if not self.rows:
            if int(row["contacted"]):
                raise ValueError("Contact without a recorded step")
            return result
        a = np.abs([[r[f"a{i}"] for i in range(1, 8)] for r in self.rows])
        frame, axis = np.unravel_index(a.argmax(), a.shape)
        peak = a[frame, axis]
        if not np.isclose(peak, row["peak_contact20_accel"], rtol=0, atol=1e-7):
            raise ValueError("Physical-step acceleration disagrees with original grader")
        result.update(first_parts=self.rows[0]["parts"], first_dominant=self.rows[0]["dominant"],
                      peak_axis=axis+1, peak_ms=self.rows[frame]["ms"], trace_peak=peak)
        crossed = np.flatnonzero(a.max(axis=1) > 4.)
        if crossed.size:
            first = crossed[0]
            result.update(breach_axes="|".join(str(i+1) for i in np.flatnonzero(a[first] > 4.)),
                          breach_ms=self.rows[first]["ms"], breach_dominant=self.rows[first]["dominant"])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--profile", required=True, choices=("normal", "noisy"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    model = build_scene("ball")
    center = capture_center(initial(model))
    rows, frames = [], []
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        trace = ContactTrace(model)
        row = compare(model, args.seed, case, p, v, args.profile, "track_reserve", True,
                      record_frame=trace.record)
        rows.append({**row, **trace.summary(row)})
        frames.extend(dict(seed=args.seed, scenario=case, profile=args.profile, **r) for r in trace.rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "trials.csv", rows)
    if frames:
        save_csv(args.output_dir / "frames.csv", frames)


if __name__ == "__main__":
    main()
