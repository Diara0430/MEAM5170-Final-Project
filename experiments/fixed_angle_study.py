"""Pair cuboid catches at the same observed intercept and requested palm speed."""
import argparse
from functools import partial
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from controllers.fixed_intercept import FixedIntercept, CalibrationMPC, FixedPointMPC, PreparationFinished
from experiments.box_fixed_speeds import arm
from experiments.contact_preview import trial
from experiments.follow_through_ablation import save_csv
from experiments.intercept_tracking import initial
from experiments.observed_long_arc import capture_center, throws
from experiments.shape_catch import BoxObserver, box_launch
from scene_setup.scene import build_scene


class RotatedFixedMPC(FixedPointMPC):
    def __init__(self, *args, angle, **kwargs):
        super().__init__(*args, **kwargs)
        # 手掌局部 z 轴为掌面法向；只改变绕法向的旋转。
        self.orientation = self.orientation @ Rotation.from_euler('z', angle, degrees=True).as_matrix()
        self.feasible_plans = 0

    def solve(self, *args, **kwargs):
        plan, info = super().solve(*args, **kwargs)
        self.feasible_plans += int(plan is not None)
        return plan, info


def run(model, seed, case, p, v, profile, angles):
    sensor = partial(BoxObserver, seed=seed, case=case, profile=profile, long_track=True, use_pose=True)
    common = dict(tracker_factory=arm, observer_factory=sensor, launch_rotation=box_launch(seed, case))
    fixture = FixedIntercept()
    try:
        trial(model, seed, case, p, v, True, planner_factory=partial(CalibrationMPC, fixture=fixture,
              planning_speed_limit=.97), **common)
    except PreparationFinished:
        pass
    rows = []
    shift = (seed + case) % len(angles)
    for angle in angles[shift:] + angles[:shift]:
        planner = None
        def factory(*args, **kwargs):
            nonlocal planner
            planner = RotatedFixedMPC(*args, fixture=fixture, label='mid', angle=angle,
                                      planning_speed_limit=.97, **kwargs)
            return planner
        audit = {"initial": None, "contact": None, "relative": None, "after": None, "previous_arm_velocity": None, "axis6_peak": None}
        def record(m, d, info, observer, tracker):
            palm = d.body("hand/palm_lower")
            R = palm.xmat.reshape(3, 3).copy()
            if audit["initial"] is None:
                audit["initial"] = R
            onset = info["first_contact_s"]
            # 用相邻物理步的第六轴速度计算首触后 20 ms 加速度峰值。
            current_velocity = d.qvel[tracker.dofs].copy()
            previous_velocity = audit["previous_arm_velocity"]
            audit["previous_arm_velocity"] = current_velocity
            if onset is not None and previous_velocity is not None:
                step = round((float(d.time) - m.opt.timestep - onset) / m.opt.timestep)
                if 0 <= step < round(.020 / m.opt.timestep):
                    acceleration = abs(float((current_velocity[5] - previous_velocity[5]) / m.opt.timestep))
                    audit["axis6_peak"] = max(audit["axis6_peak"] or 0., acceleration)
            if onset is None:
                return
            quat = d.joint("flight").qpos[3:].copy()
            obj_R = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
            relative = R.T @ obj_R
            center = palm.xpos + R @ np.array([0., -.055, -.08])
            if audit["contact"] is None:
                target = audit["initial"] @ Rotation.from_euler("z", angle, degrees=True).as_matrix()
                error = np.rad2deg(Rotation.from_matrix(target.T @ R).magnitude())
                jac = np.empty((3, m.nv))
                mujoco.mj_jac(m, d, jac, None, center, m.body("hand/palm_lower").id)
                direction = fixture.incoming / np.linalg.norm(fixture.incoming) if fixture.incoming is not None else np.zeros(3)
                actual_speed = float(direction @ jac @ d.qvel)
                audit["contact"] = (error, actual_speed, R.T @ (d.joint("flight").qpos[:3] - center))
                audit["relative"] = relative
            if audit["after"] is None and float(d.time) >= onset + .100 - 1e-9:
                pos = R.T @ (d.joint("flight").qpos[:3] - center)
                angle_change = np.rad2deg(Rotation.from_matrix(audit["relative"].T @ relative).magnitude())
                audit["after"] = (float(np.linalg.norm(pos-audit["contact"][2]))*1000, angle_change)
        result = trial(model, seed, case, p, v, True, planner_factory=factory, record_frame=record, **common)
        result.update(angle_deg=angle, profile=profile, fixture_status=fixture.reason,
                      fixed_time=fixture.time, fixed_x=None if fixture.point is None else fixture.point[0],
                      fixed_y=None if fixture.point is None else fixture.point[1],
                      fixed_z=None if fixture.point is None else fixture.point[2],
                      requested_speed_m_s=fixture.target_speed('mid'),
                      feasible_plans=planner.feasible_plans, calibration_ms=fixture.preparation_ms,
                      axis6_peak_contact20_rad_s2=audit["axis6_peak"],
                      contact_orientation_error_deg=None if audit["contact"] is None else audit["contact"][0],
                      contact_palm_speed_m_s=None if audit["contact"] is None else audit["contact"][1],
                      relative_travel_100ms_mm=None if audit["after"] is None else audit["after"][0],
                      relative_rotation_100ms_deg=None if audit["after"] is None else audit["after"][1])
        rows.append(result)
        print(f"{seed} case={case} angle={angle:g} fixture={fixture.reason} plans={planner.feasible_plans} applied={result['accepted']} held={result['held']} qualified={result['qualified']}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cases', type=int, default=8)
    parser.add_argument('--profile', choices=('normal','noisy','corrected'), default='normal')
    parser.add_argument('--angles', nargs='+', type=float, default=[0., -10., 10., -20., 20.])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    model = build_scene('box')
    center = capture_center(initial(model))
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case, p, v, *_ in throws(center, model.opt.gravity, args.seed):
        if case >= args.cases:
            continue
        rows.extend(run(model, args.seed, case, p, v, args.profile, args.angles))
        columns = dict.fromkeys(k for row in rows for k in row)
        save_csv(args.output_dir/'trials.csv', [{**columns, **r} for r in rows])

if __name__ == '__main__':
    main()
