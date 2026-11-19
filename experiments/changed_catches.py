"""复现固定闭合与反馈结果不同的阵风。记录碰撞与对侧支撑建立时的力学量。"""
import argparse
import csv
from pathlib import Path
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial


class ImpactTrace:
    def __init__(self, seed, controller):
        self.labels = dict(seed=seed, controller=controller)
        self.samples = []
        self.last_delay = np.zeros(4)
        self.first_slow = self.first_support = None

    def step(self, model, data, elapsed, closure, velocity):
        # 控制指令作用于本步起点，接触检测结果最早能在下一步用于控制。
        ms = round(1000 * elapsed, 6)
        end_ms = round(1000 * (elapsed + closure.dt), 6)
        if closure.holding and self.first_support is None:
            self.first_support = end_ms
        slowed = np.flatnonzero(closure.delay > self.last_delay)
        if slowed.size and self.first_slow is None:
            self.first_slow = ms
        self.last_delay[:] = closure.delay
        if ms > 200:
            return

        # group -1 为掌面，0..3 为四指。法向力的标量和描述夹持载荷，不是合力。
        loads = np.zeros(5)
        slips = []
        projectile = model.geom_bodyid[closure.projectile]
        for i, contact in enumerate(data.contact):
            if closure.projectile not in (contact.geom1, contact.geom2) or contact.efc_address < 0:
                continue
            other = contact.geom1 if contact.geom2 == closure.projectile else contact.geom2
            group = closure.group.get(other)
            if group is None:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            if wrench[0] <= 0:
                continue
            loads[group + 1] += wrench[0]
            if group >= 0:
                # 步初几何和步初广义速度对齐；分别求同一世界接触点两侧的速度。
                jo, jf = np.empty((3, model.nv)), np.empty((3, model.nv))
                mujoco.mj_jac(model, data, jo, None, contact.pos, projectile)
                mujoco.mj_jac(model, data, jf, None, contact.pos, model.geom_bodyid[other])
                slips.append(np.linalg.norm(contact.frame.reshape(3, 3)[1:] @ ((jo - jf) @ velocity)))
        dof = model.joint("flight").dofadr[0]
        self.samples.append(dict(**self.labels, ms=ms, step_end_ms=end_ms,
            finger_normal_N=sum(loads[1:]), palm_normal_N=loads[0],
            **{f"finger_{j}_normal_N": force for j, force in enumerate(loads[1:], 1)},
            finger_contact_points=len(slips), finger_slip_m_s=np.mean(slips) if slips else None,
            spin_rad_s=np.linalg.norm(velocity[dof + 3:dof + 6]),
            slowed_fingers="|".join(str(j + 1) for j in slowed),
            opposed_this_step=int(closure.opposed_time > 0), support_confirmed_at_end=int(closure.holding)))


def replay(row):
    seed, name = int(row["seed"]), row["controller"]
    trace = ImpactTrace(seed, name)
    pulse = [float(row[k]) for k in ("gust_onset_s", "gust_duration_s", "gust_az_m_s2")]
    timing = TimingTrial(pulse, online=False)
    # 精确复用原试验的已记录阵风；基线也观测支撑状态，但固定闭合不使用它。
    run("early", .04, .19, True, shape=row["shape"], pitch_deg=float(row["pitch_deg"]),
        palm_x_mm=float(row["palm_x_mm"]), arrival_speed=row["speed"], timing_trial=timing,
        hand_feedback=name == "feedback", contact_observer=trace.step)
    for key in ("passed", "failure_stage", "first_contact_s", "close_s", "max_drift_mm"):
        if str(timing.outcome[key]) != row[key]:
            raise ValueError(f"Replay differs for seed {seed}, {name}, {key}")
    for ms in (20, 50):
        spin = next(s["spin_rad_s"] for s in trace.samples if s["ms"] == ms)
        if abs(spin - float(row[f"spin_{ms}ms_rad_s"])) > .000006:
            raise ValueError(f"Replay spin differs at {ms} ms for seed {seed}, {name}")
    return trace, dict(**trace.labels, **timing.outcome, first_slow_ms=trace.first_slow,
                       first_confirmed_support_ms=trace.first_support)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=Path(__file__).resolve().parents[1] /
                        "results/contact_feedback/trials.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    with args.trials.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            seed = int(row["seed"])
            if seed >= 0:
                pair = cases.setdefault(seed, {})
                if row["controller"] in pair:
                    raise ValueError(f"Duplicate controller for seed {seed}")
                pair[row["controller"]] = row
    samples, events = [], []
    for pair in cases.values():
        if pair["fixed"]["passed"] == pair["feedback"]["passed"]:
            continue
        for name in ("fixed", "feedback"):
            trace, event = replay(pair[name])
            samples.extend(trace.samples)
            events.append(event)
    if not events:
        parser.error("No paired gusts with different catch outcomes.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_rows(args.output_dir / "impact.csv", samples)
    write_rows(args.output_dir / "events.csv", events)
