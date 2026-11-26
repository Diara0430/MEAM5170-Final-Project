"""Measure the first 100 ms of relative cuboid motion, then track the two-second hold verdict."""
import argparse
from itertools import combinations
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.shape_catch import simulate
from scene_setup.scene import build_scene


class ContactWindow:
    def __init__(self, model):
        self.map = ContactClosure(model, LeapHand(model))
        self.rows = []
        self.origin = self.orientation = None
        self.floor = self.gap = self.drift = None
        self.has_support = self.lost_support = None
        self.thumb_streak = self.any_streak = 0
        self.thumb_time = self.any_time = None
        self.last_finger = None

    def record(self, model, data, info, observer, tracker):
        onset = info["first_contact_s"]
        if onset is None:
            return
        dt = model.opt.timestep
        # Frame contains the post-step state. The first collision ends at +2 ms.
        ms = round(1000*(float(data.time)-onset), 5)
        if info["floor"] and self.floor is None:
            self.floor = ms
        if info["finger_contact"]:
            self.last_finger = ms
        if ms >= 300.-1e-6:
            if not info["finger_contact"] and self.gap is None:
                self.gap = ms
            if info["drift_mm"] > 25. and self.drift is None:
                self.drift = ms
        if ms > 100.000001:
            return
        palm = data.body("hand/palm_lower")
        obj = data.body("projectile")
        rot = palm.xmat.reshape(3,3)
        pos = rot.T @ (obj.xpos-palm.xpos)
        rotation = Rotation.from_matrix(rot.T @ obj.xmat.reshape(3,3))
        if self.origin is None:
            self.origin, self.orientation = pos.copy(), rotation
        delta = 1000*(pos-self.origin)
        angle = np.rad2deg((self.orientation.inv()*rotation).magnitude())
        wrench = np.empty(6)
        loads = np.zeros(5)  # index, middle, ring, thumb, palm
        contacts = []
        for i, c in enumerate(data.contact):
            if c.efc_address < 0 or self.map.projectile not in (c.geom1,c.geom2):
                continue
            on_second = c.geom2 == self.map.projectile
            group = self.map.group.get(c.geom1 if on_second else c.geom2)
            if group is None:
                continue
            mujoco.mj_contactForce(model, data, i, wrench)
            loads[group] += max(0., wrench[0])
            if wrench[0] >= .02:
                contacts.append((group, c.pos-obj.xipos, c.frame[:3] if on_second else -c.frame[:3]))
        pairs = [(a,b) for (a,ra,na),(b,rb,nb) in combinations(contacts,2)
                 if a != b and ra @ rb < 0 and na @ nb < -.5]
        thumb = any(3 in pair and -1 not in pair for pair in pairs)
        self.any_streak = self.any_streak+1 if pairs else 0
        self.thumb_streak = self.thumb_streak+1 if thumb else 0
        if self.any_time is None and self.any_streak*dt >= .006-1e-10:
            self.any_time = ms
        if self.thumb_time is None and self.thumb_streak*dt >= .006-1e-10:
            self.thumb_time = ms
        loaded = bool(contacts)
        if loaded:
            self.has_support = ms
        elif self.has_support is not None and self.lost_support is None:
            self.lost_support = ms
        self.rows.append(dict(ms=ms, x_mm=1000*pos[0], y_mm=1000*pos[1], z_mm=1000*pos[2],
                              dx_mm=delta[0], dy_mm=delta[1], dz_mm=delta[2],
                              travel_mm=np.linalg.norm(delta), angle_deg=angle,
                              index_N=loads[0], middle_N=loads[1], ring_N=loads[2],
                              thumb_N=loads[3], palm_N=loads[4], loaded=int(loaded),
                              opposed=int(bool(pairs)), thumb_opposed=int(thumb)))

    def summary(self):
        r = dict(first_opposition_ms=self.any_time, thumb_opposition_ms=self.thumb_time,
                 first_unloaded_ms=self.lost_support, floor_ms=self.floor, first_failed_hold_contact_ms=self.gap,
                 over25_after_settle_ms=self.drift, last_finger_ms=self.last_finger, window_frames=len(self.rows))
        for t in (10,20,40,60,80,100):
            frame = next((f for f in self.rows if abs(f["ms"]-t)<1e-6), {})
            for field in ("travel_mm","angle_deg","loaded","thumb_N","opposed"):
                r[f"{field}_{t}"] = frame.get(field)
        return r


def run(seeds, profile, conditions, directory, frames=False):
    model = build_scene("box")
    center = capture_center(initial(model))
    trials, paths = [], []
    for seed in seeds:
        for case, p, v, *_ in throws(center, model.opt.gravity, seed):
            shift = (seed+case) % len(conditions)
            for method, grasp in conditions[shift:]+conditions[:shift]:
                audit = ContactWindow(model)
                row = simulate(model, seed, case, p, v, profile, method, grasp=grasp, record_frame=audit.record)
                trials.append({**row, **audit.summary()})
                if frames:
                    paths.extend(dict(seed=seed, scenario=case, profile=profile, method=method,
                                      grasp=grasp, **f) for f in audit.rows)
    directory.mkdir(parents=True, exist_ok=True)
    cols = dict.fromkeys(k for r in trials for k in r)
    save_csv(directory/"trials.csv", [{**cols, **r} for r in trials])
    if paths:
        save_csv(directory/"frames.csv", paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--profile", choices=("normal","noisy","corrected"), required=True)
    parser.add_argument("--conditions", nargs="+", default=["pose/original","axis_lever/original"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", action="store_true")
    args = parser.parse_args()
    options = [tuple(c.split("/")) for c in args.conditions]
    if (any(len(c)!=2 or c[0] not in ("pose","axis_lever") for c in options) or
            len(set(options)) != len(options) or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Unique seeds and distinct method/grasp pairs required")
    run(args.seeds, args.profile, options, args.output_dir, args.frames)
