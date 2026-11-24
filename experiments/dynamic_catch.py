"""固定手掌接球。在 VS Code 直接运行，或使用 python -m experiments.dynamic_catch。"""
import argparse
import csv
from contextlib import ExitStack, nullcontext
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

# 让 VS Code 的“运行 Python 文件”和从仓库根目录启动都能找到项目模块。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.hand import LeapHand
from controllers.contact_closure import ContactClosure
from experiments.static_grasp import contacts_with
from scene_setup.scene import build_scene


# 倍率作用在物体到达目标点时的速度；--speed 只改变动画播放速度。
ARRIVAL_SPEED_SCALES = {"slow": 0.75, "medium": 1.0, "fast": 1.25}
HOLD_DURATION_S = .5


def position_palm(model, data, home, arm_dofs, goal, window, speed, desired_rotation=None):
    """在独立状态中求逆运动学，再让真实仿真中的七个关节跟踪目标。"""
    joints = [model.joint(f"joint{i}") for i in range(1, 8)]
    qadr = [j.qposadr[0] for j in joints]
    ctrl = [model.actuator(f"act{i}").id for i in range(1, 8)]
    ik = mujoco.MjData(model)
    ik.qpos[:] = data.qpos
    palm = ik.body("hand/palm_lower")
    reference = data.body("hand/palm_lower").xmat.reshape(3, 3).copy()
    if desired_rotation is not None:
        reference = desired_rotation
    jac_p = np.empty((3, model.nv))
    jac_r = np.empty_like(jac_p)
    limits = model.jnt_range[[j.id for j in joints]]
    if desired_rotation is not None:
        limits = np.column_stack((np.maximum(limits[:,0],model.actuator_ctrlrange[ctrl,0]),
                                  np.minimum(limits[:,1],model.actuator_ctrlrange[ctrl,1])))
    for _ in range(100):
        mujoco.mj_forward(model, ik)
        current = palm.xmat.reshape(3, 3)
        # 三个轴的叉积给出小角度姿态误差；平移时也要保持掌面朝向。
        angular_error = .5 * np.cross(current.T, reference.T).sum(axis=0)
        error = np.r_[goal - palm.xpos, angular_error]
        if np.linalg.norm(error) < 1e-6:
            break
        mujoco.mj_jacBody(model, ik, jac_p, jac_r, palm.id)
        jac = np.vstack((jac_p[:, arm_dofs], jac_r[:, arm_dofs]))
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
        if desired_rotation is not None:
            delta = np.clip(delta,-.15,.15)
        ik.qpos[qadr] = np.clip(ik.qpos[qadr] + delta, limits[:, 0], limits[:, 1])
    else:
        raise ValueError("Requested palm pose exceeds the reach of this IK configuration.")

    # 来物暂放远处地面。发射前用位置执行器平滑移掌，再留时间稳定。
    data.joint("flight").qpos[:3] = [2., 2., .04]
    mujoco.mj_forward(model, data)
    target = ik.qpos[qadr].copy()
    dt = model.opt.timestep
    for k in range(round(1.5 / dt)):
        u = min((k + 1) * dt / .6, 1.)
        data.ctrl[ctrl] = home + (target - home) * u*u*(3 - 2*u)
        data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
        mujoco.mj_step(model, data)
        if window and k % 8 == 0:
            if not window.is_running():
                return None
            window.sync()
            time.sleep(8 * dt / speed)
    mujoco.mj_forward(model, data)
    return target


def record_contacts(writer, model, data, geom, hand_geoms, velocity, spin, t, mode, shape, on_floor,
                    pitch_deg, palm_x_mm, strategy, arrival_speed, lead):
    # 每个接触点单独一行。同一步出现多行，说明物体同时碰到了多个接触点。
    obj_body = model.geom_bodyid[geom]
    wrote_contact = False
    for c in data.contact:
        if c.dist > 0 or geom not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == geom else c.geom1
        if other not in hand_geoms:
            continue
        other_body = model.geom_bodyid[other]
        # 在同一个世界坐标接触点，分别计算物体表面和手指表面的线速度。
        jac_obj = np.empty((3, model.nv))
        jac_hand = np.empty_like(jac_obj)
        mujoco.mj_jac(model, data, jac_obj, None, c.pos, obj_body)
        mujoco.mj_jac(model, data, jac_hand, None, c.pos, other_body)
        relative_velocity = (jac_obj - jac_hand) @ velocity
        # 接触坐标系的后两轴张成切平面。投影后的速度大小衡量局部滑动。
        slip = np.linalg.norm(c.frame.reshape(3, 3)[1:] @ relative_velocity)
        writer.writerow([shape, mode, f"{t:.3f}", model.body(other_body).name,
                         f"{spin:.5f}", f"{slip:.5f}", int(on_floor), pitch_deg, palm_x_mm, strategy, arrival_speed, lead])
        wrote_contact = True
    if not wrote_contact:
        # 离手后继续记录；此时没有指面接触，滑动速度留空。
        writer.writerow([shape, mode, f"{t:.3f}", "", f"{spin:.5f}", "", int(on_floor),
                         pitch_deg, palm_x_mm, strategy, arrival_speed, lead])


def run(mode, lead, duration, headless, shape="ball", speed=1., writer=None, aim_x_mm=0.,
        pitch_deg=0., palm_x_mm=None, summary=None, strategy="fixed", arrival_speed="medium",
        after_contact_step=None, dip3_duration=None, timing_trial=None, hand_feedback=False,
        contact_observer=None, max_delay_ms=None, thumb_base_offset_deg=0., post_close_control=None,
        wrist_control=None, frame_observer=None, arm_buffer=None, evaluation="tilt",
        hold_duration_s=HOLD_DURATION_S, pretilt_deg=0.):
    if not np.isfinite(pretilt_deg) or abs(pretilt_deg)>90:
        raise ValueError("Pre-catch palm rotation must be within +/-90 degrees")
    if evaluation not in ("tilt", "hold"):
        raise ValueError("Evaluation must be tilt or hold")
    if not np.isfinite(hold_duration_s) or hold_duration_s <= 0:
        raise ValueError("Hold duration must be finite and positive")
    if evaluation == "hold" and wrist_control is not None:
        raise ValueError("Hold-only evaluation requires no tilt controller")
    # 1. 替换来物的几何形状，各组复用同样的发射条件和手指控制器。
    model = build_scene(shape)
    data = mujoco.MjData(model)
    hand = LeapHand(model)
    # 只改变拇指根部关节 12 的最终位置目标；原有控制器默认偏移为零。
    thumb_delta = np.deg2rad(thumb_base_offset_deg)
    if not hand.limits[12, 0] <= hand.closed[12] + thumb_delta <= hand.limits[12, 1]:
        raise ValueError("Thumb base target exceeds its actuator limits.")
    thumb_start = None
    # 固定闭合也可旁路观测对侧支撑；只有 hand_feedback 能改变目标。
    hand_feedback = hand_feedback or max_delay_ms is not None
    max_delay_s = None if max_delay_ms is None else max_delay_ms / 1000
    closure = ContactClosure(model, hand, max_delay_s) if hand_feedback or contact_observer is not None else None
    if dip3_duration is not None:
        if not np.isfinite(dip3_duration) or dip3_duration <= 0:
            raise ValueError("DIP 3 duration must be finite and positive.")
        # 用碰撞体所属关节查找执行器；保持另外 15 个关节的目标曲线相同。
        joint_id = model.body("hand/dip_3").jntadr[0]
        actuator_id = model.actuator(model.joint(joint_id).name).id
        if model.actuator_trnid[actuator_id, 0] != joint_id:
            raise ValueError("DIP 3 actuator transmission does not match its joint.")
        duration = float(duration)
        finger_durations = np.full(len(hand.ids), duration)
        finger_durations[hand.ids.index(actuator_id)] = dip3_duration
    else:
        finger_durations = duration
    home = [0., .4, 0., 2., 0., -1.54, 0.]
    arm_dofs = []
    for i, angle in enumerate(home, 1):
        data.joint(f"joint{i}").qpos[:] = angle
        data.actuator(f"act{i}").ctrl[:] = angle
        arm_dofs.append(model.joint(f"joint{i}").dofadr[0])
    mujoco.mj_forward(model, data)

    # 2. 手掌朝上。沿手掌局部 x 轴平移来物轨迹，保持初速度和来物姿态相同。
    palm = data.body("hand/palm_lower")
    rotation = palm.xmat.reshape(3, 3)
    normal_at_start = -rotation[:, 2].copy()
    origin = palm.xpos.copy()
    launch_axis = rotation[:, 0].copy()
    goal = origin + launch_axis * (palm_x_mm or 0.) / 1000
    target = palm.xpos + rotation @ [aim_x_mm / 1000., -.055, -.08]
    flight = data.joint("flight")
    arrival = .20
    start = target + [.055, 0., .14]
    # 只改物体的初始朝向：绕世界水平 Y 轴旋转，四元数顺序为 w、x、y、z。
    # 每组初始自旋为零，同一速度档的两种掌位策略共用完全相同的发射条件。
    half_pitch = np.deg2rad(pitch_deg) / 2
    # 由 p(T) = p0 + v0*T + g*T²/2 得初速度，此后球只受重力与碰撞影响。
    launch_velocity = (target - start) / arrival - .5 * model.opt.gravity * arrival
    # 保留原来的中速轨迹。其余两档只缩放目标点速度矢量；到达方向和时间不变。
    # 按匀加速方程倒推发射速度与发射点，避免把入射角或预计到达时间同时改掉。
    arrival_velocity = launch_velocity + model.opt.gravity * arrival
    if arrival_speed != "medium":
        arrival_velocity = arrival_velocity * ARRIVAL_SPEED_SCALES[arrival_speed]
        launch_velocity = arrival_velocity - model.opt.gravity * arrival
        start = target - launch_velocity * arrival - .5 * model.opt.gravity * arrival**2
    object_geom = model.geom("projectile").id
    floor = model.geom("floor").id
    hand_geoms = {i for i, b in enumerate(model.geom_bodyid)
                  if (model.body(b).name or "").startswith("hand/")}
    fingers = {i for i in hand_geoms if model.geom_bodyid[i] != palm.id}

    # 3. 三组共用相同的倾腕与保持时间。预计到达时间指球心到达目标的时刻。
    dt = model.opt.timestep
    tilt_start = max(.90, arrival + duration + .25)
    hold_start = tilt_start + (1. if evaluation == "tilt" else 0.)
    first_touch = close_start = reference = None
    first_links = []
    precontact_speed = None
    last_touch = first_floor = None
    last_hand_before_floor = first_excess_drift = None
    grounded = False
    hold_ok = True
    drift = 0.
    if headless:
        display = nullcontext(None)
    else:
        from mujoco import viewer
        display = viewer.launch_passive(model, data)
    with display as window:
        if window:
            window.cam.lookat[:] = target
            window.cam.distance = .65
            window.cam.elevation = -20
        if palm_x_mm is not None or pretilt_deg:
            desired_rotation = None
            if pretilt_deg:
                # 保持世界系接取中心、来物轨迹不变，只改变接球前的掌面朝向。
                angle=np.deg2rad(pretilt_deg)
                turn=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
                desired_rotation=rotation.copy()@turn
                goal=target-desired_rotation@np.array([aim_x_mm/1000.,-.055,-.08])
            home = position_palm(model, data, np.asarray(home), arm_dofs, goal, window, speed,desired_rotation)
            if home is None:
                return False
            if desired_rotation is not None:
                position_error=np.linalg.norm(palm.xpos-goal)
                orientation_error=np.arccos(np.clip((np.trace(desired_rotation.T@rotation)-1)/2,-1,1))
                if position_error>.005 or orientation_error>np.deg2rad(3):
                    raise ValueError("Pre-catch pose tracking exceeds 5 mm or 3 degrees")
                normal_at_start=-rotation[:,2].copy()
        achieved_mm = 1000 * np.dot(palm.xpos - origin, launch_axis)
        error_mm = 1000 * np.linalg.norm(palm.xpos - goal)
        # 发射点和初速度在移掌前已按世界坐标确定。所有试验时间以释放为零点。
        flight.qpos[:3] = start
        flight.qpos[3:] = [np.cos(half_pitch), 0., np.sin(half_pitch), 0.]
        flight.qvel[:] = np.r_[launch_velocity, np.zeros(3)]
        flight_dof = model.joint("flight").dofadr[0]
        data.qacc_warmstart[flight_dof:flight_dof + 6] = 0
        mujoco.mj_forward(model, data)
        if contacts_with(data, object_geom):
            raise RuntimeError("Object starts in contact; check the launch pose.")
        # 自适应轨迹允许延长倾腕；完成后仍须经历完整的保持阶段。
        end_limit = hold_start + hold_duration_s
        if wrist_control is not None:
            hold_start = tilt_start + wrist_control.max_duration
            end_limit = hold_start + hold_duration_s
        for k in range(int(np.ceil(end_limit / dt))):
            t = k * dt
            if t >= hold_start + hold_duration_s - 1e-12:
                break
            planned_t = arrival
            if timing_trial is not None:
                z_before = float(flight.qpos[2])
                planned_t = timing_trial.before_step(model, data, flight, target, t, arrival,
                                                    close_start is None)
            # 4. open 全程张手；contact 在检测到碰撞后的下一步启动；
            # early 在预计到达前 lead 秒启动。只在第一次触发时记下起点。
            if close_start is None:
                if mode == "early" and t + 1e-12 >= planned_t - lead:
                    close_start = t
                    if timing_trial is not None:
                        timing_trial.prediction_at_close = planned_t
                elif mode == "contact" and first_touch is not None:
                    close_start = t
            if close_start is not None:
                elapsed = t - close_start
                fraction = (closure.fraction(elapsed, finger_durations) if hand_feedback
                            else elapsed / finger_durations)
                hand.command(data, hand.open_pose, hand.closed, fraction)
                if thumb_delta and first_touch is not None:
                    # 等初次碰撞后 250 ms 且原闭合完成，才用 160 ms 平滑扫动拇指。
                    # 若剩余时间不足以在倾腕前完成，则跳过本次调整。
                    if thumb_start is None and t >= first_touch + .250 and np.all(fraction >= 1.):
                        if t + .160 <= tilt_start + 1e-12:
                            thumb_start = t
                    if thumb_start is not None:
                        progress = min((t - thumb_start) / .160, 1.)
                        data.ctrl[hand.ids[12]] += thumb_delta * progress**2 * (3 - 2*progress)
                if (post_close_control is not None and first_touch is not None and
                        np.all(fraction >= 1.) and t >= first_touch + .050):
                    post_close_control(model, data, hand, t, tilt_start)

            # 5. 接球时维持手掌姿态；随后在一秒内平滑倾腕 105°，考验抓持。
            u = np.clip(t - tilt_start, 0., 1.)
            if evaluation == "hold":
                u = 0.
            if wrist_control is not None:
                u = wrist_control.advance(model, data, t, tilt_start)
                if wrist_control.completed_at is not None:
                    hold_start = wrist_control.completed_at
            data.actuator("act6").ctrl[:] = home[5] + np.deg2rad(105) * u*u*(3 - 2*u)
            if arm_buffer is not None:
                arm_buffer.command(model, data, t, tilt_start)
            data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
            # mj_step 返回时接触几何仍对应本步起点，速度已经积分到下一步。
            # 记录时保存步初速度，让接触点位置和速度对应同一个时刻。
            if writer is not None or contact_observer is not None:
                velocity = data.qvel.copy()
                spin = np.linalg.norm(flight.qvel[3:])
            if first_touch is None:
                # 本步发生碰撞时，用积分前质心速度，避免混入接触冲量。
                incoming_velocity = flight.qvel[:3].copy()
                if after_contact_step is not None:
                    incoming_twist = flight.qvel.copy()
            mujoco.mj_step(model, data)
            if closure is not None:
                closure.observe(model, data)

            # 6. 只统计球的接触对。碰地即失去接球资格，手指互碰不计入接球。
            touching = contacts_with(data, object_geom)
            if timing_trial is not None and first_touch is None and not touching:
                timing_trial.observe_crossing(t, dt, z_before, flight.qpos[2], target[2])
            on_floor = floor in touching
            # 用首次落地前最后一次手部接触定位脱手，排除落地后再次碰手的情况。
            if touching & hand_geoms and first_floor is None:
                last_hand_before_floor = t
            grounded |= on_floor
            if on_floor and first_floor is None:
                first_floor = t
            if touching & hand_geoms:
                if first_touch is None:
                    first_touch = t
                    precontact_speed = float(np.linalg.norm(incoming_velocity))
                    first_links = sorted({model.body(model.geom_bodyid[g]).name
                                          for g in touching & hand_geoms})
                last_touch = t
            if after_contact_step is not None and first_touch is not None:
                after_contact_step(model, data, t, first_touch, incoming_twist)
            if contact_observer is not None and first_touch is not None:
                contact_observer(model, data, t - first_touch, closure, velocity)
            if writer is not None and first_touch is not None:
                record_contacts(writer, model, data, object_geom, hand_geoms, velocity,
                                spin, t, mode, shape, on_floor, pitch_deg, palm_x_mm, strategy, arrival_speed, lead)
            if t >= tilt_start:
                relative = rotation.T @ (data.body("projectile").xpos - palm.xpos)
                if reference is None:
                    reference = relative.copy()
                drift = max(drift, np.linalg.norm(relative - reference))
                if drift > .025 and first_excess_drift is None:
                    first_excess_drift = t
            # 7. 最后 0.5 秒要求球连续与手指接触，同时检查确实完成了倾腕。
            if t >= hold_start:
                hold_ok &= bool(touching & fingers)
                if evaluation == "tilt":
                    hold_ok &= (-rotation[:, 2] @ normal_at_start) <= 0.
            # 录制回调读取仿真状态，图像采样不会改变控制命令或积分步长。
            if frame_observer is not None:
                frame_observer(model, data, t, tilt_start, hold_start)
            if window and k % 8 == 0:
                if not window.is_running():
                    print("Viewer closed before evaluation finished.")
                    return False
                window.sync()
                # 仅改变播放节奏，仿真的时间步长与控制器保持一致。
                time.sleep(8 * dt / speed)
    passed = first_touch is not None and hold_ok and not grounded and drift <= .025
    if wrist_control is not None and wrist_control.completed_at is None:
        passed = False
    if arm_buffer is not None and arm_buffer.infeasible:
        passed = False
    if first_touch is None or (grounded and last_hand_before_floor is None):
        failure_stage = "missed_hand"
    elif grounded:
        if last_hand_before_floor < tilt_start:
            failure_stage = "dropped_before_tilt"
        elif last_hand_before_floor < hold_start:
            failure_stage = "dropped_during_tilt"
        else:
            failure_stage = "dropped_during_hold"
    elif first_excess_drift is not None:
        failure_stage = "drift_during_tilt" if first_excess_drift < hold_start else "drift_during_hold"
    elif wrist_control is not None and wrist_control.completed_at is None:
        failure_stage = "tilt_timeout"
    elif arm_buffer is not None and arm_buffer.infeasible:
        failure_stage = "buffer_joint_limit"
    elif not hold_ok:
        failure_stage = "hold_check_failed"
    else:
        failure_stage = "passed"
    if timing_trial is not None:
        timing_trial.outcome = dict(passed=int(passed), failure_stage=failure_stage,
                                    close_s=close_start, first_contact_s=first_touch,
                                    max_drift_mm=round(1000 * drift, 3))
    # 正值表示闭指先于首次手部接触；未闭指或未碰手的试验在 CSV 留空。
    close_lead = None if first_touch is None or close_start is None else round(first_touch - close_start, 6)
    def stamp(value):
        return "-" if value is None else f"{value:.3f}s"
    print(f"{shape} {mode:7} {'PASS' if passed else 'FAIL'} | contact={stamp(first_touch)} "
          f"close={stamp(close_start)} drift={drift*1000:.1f}mm floor={grounded} "
          f"last_hand={stamp(last_touch)} first_floor={stamp(first_floor)} "
          f"pitch={pitch_deg:g}deg speed={arrival_speed} first_links={','.join(first_links) or '-'}"
          + (f" palm={achieved_mm:.2f}mm error={error_mm:.2f}mm" if palm_x_mm is not None else ""))
    if summary is not None:
        summary.writerow([shape, mode, pitch_deg, aim_x_mm, palm_x_mm, round(achieved_mm, 4),
                          round(error_mm, 4), *start, *launch_velocity, lead, duration, int(passed),
                          first_touch, close_start, last_touch, first_floor, round(drift * 1000, 3),
                          int(grounded), "|".join(first_links), strategy, arrival_speed, arrival,
                          np.linalg.norm(arrival_velocity), precontact_speed, close_lead, *target,
                          failure_stage, tilt_start, hold_start, first_excess_drift])
    return passed


if __name__ == "__main__":
    # 8. 默认展示提前闭指。compare 逐组重置场景并输出三组结果。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("open", "contact", "early", "compare"), default="early")
    parser.add_argument("--object", choices=("ball", "box"), default="ball", help="来物形状")
    parser.add_argument("--pitch-deg", type=float, nargs="+", default=[0.], metavar="DEG",
                        help="物体绕世界 Y 轴的初始姿态角（度），可输入多个角度依次测试")
    parser.add_argument("--aim-x-mm", type=float, default=0., help="沿掌心局部 x 轴平移发射轨迹（毫米）")
    parser.add_argument("--palm-x-mm", nargs="+", type=float, help="发射前沿掌心初始 x 轴移动手掌（毫米）")
    parser.add_argument("--pitch-policy", action="store_true",
                        help="增加按已知长方体姿态选掌位的试验，可与固定掌位做对照")
    parser.add_argument("--summary", type=Path, help="每次试验一行，写入新的汇总 CSV")
    parser.add_argument("--speed", type=float, default=1., help="窗口播放倍率，0.25 表示四分之一速")
    parser.add_argument("--arrival-speed", nargs="+", choices=ARRIVAL_SPEED_SCALES, default=["medium"],
                        help="目标点速度档位，可依次测试 slow、medium、fast")
    parser.add_argument("--trace", type=Path, help="将逐步接触记录保存到新的 CSV 文件")
    parser.add_argument("--lead", type=float, nargs="+", default=[.06],
                        help="相对预计到达时刻的提前闭指时间（秒），可依次测试多个数值")
    parser.add_argument("--duration", type=float, default=.22, help="从张手到闭合目标的时间（秒）")
    parser.add_argument("--headless", action="store_true", help="关闭窗口，直接输出结果")
    parser.add_argument("--hand-feedback", action="store_true", help="按手指承载和对侧接触调节闭合速度")
    parser.add_argument("--thumb-base-offset-deg", type=float, default=0.,
                        help="接取后平滑调整拇指根部目标角度，正负号按关节 12 的模型坐标")
    parser.add_argument("--max-delay-ms", type=float, help="启用反馈并限制每指的累计闭合延迟（毫秒）")
    args = parser.parse_args()
    if args.max_delay_ms is not None and (not np.isfinite(args.max_delay_ms) or args.max_delay_ms < 0):
        parser.error("--max-delay-ms must be finite and nonnegative")
    if not all(0 <= lead <= .20 for lead in args.lead) or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("lead must be in [0, 0.20] and duration must be finite and positive")
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if not np.isfinite(args.aim_x_mm):
        parser.error("--aim-x-mm must be finite")
    if not np.isfinite(args.pitch_deg).all():
        parser.error("--pitch-deg angles must be finite")
    if args.palm_x_mm and not np.isfinite(args.palm_x_mm).all():
        parser.error("--palm-x-mm must be finite")
    if args.pitch_policy and (args.object != "box" or min(args.pitch_deg) < 0 or max(args.pitch_deg) > 45):
        parser.error("--pitch-policy is calibrated for boxes at 0 to 45 degrees")
    if args.summary and args.trace and args.summary.resolve() == args.trace.resolve():
        parser.error("--trace and --summary need different filenames")
    with ExitStack() as stack:
        def open_csv(path):
            if path is None:
                return None
            try:
                stream = stack.enter_context(path.open("x", newline="", encoding="utf-8-sig"))
            except OSError as error:
                parser.error(str(error))
            return csv.writer(stream)

        summary = open_csv(args.summary)
        writer = open_csv(args.trace)
        if summary is not None:
            summary.writerow(["object", "mode", "pitch_deg", "aim_x_mm", "target_palm_x_mm",
                              "actual_palm_x_mm", "palm_error_3d_mm", "launch_x_m", "launch_y_m",
                              "launch_z_m", "vx_m_s", "vy_m_s", "vz_m_s", "lead_s", "duration_s",
                              "passed", "first_hand_s", "close_s", "last_hand_s", "first_floor_s",
                              "max_drift_mm", "hit_floor", "first_links", "strategy", "arrival_speed",
                              "planned_arrival_s", "planned_speed_m_s", "precontact_speed_m_s",
                              "contact_minus_close_s", "target_x_m", "target_y_m", "target_z_m",
                              "failure_stage", "tilt_start_s", "hold_start_s", "first_excess_drift_s"])
        if writer is not None:
            writer.writerow(["object", "mode", "time_s", "hand_body", "spin_rad_s",
                             "tangential_slip_m_s", "floor_contact", "pitch_deg", "target_palm_x_mm", "strategy",
                             "arrival_speed", "lead_s"])
        modes = ("open", "contact", "early") if args.mode == "compare" else (args.mode,)
        for pitch_deg in args.pitch_deg:
            choices = [("fixed", offset) for offset in args.palm_x_mm or []]
            if args.pitch_policy:
                # 30° 用 20 mm、45° 用 10 mm；两者中点作分界，验证前固定规则。
                choices.append(("pitch_policy", 20. if pitch_deg <= 37.5 else 10.))
            for strategy, offset in choices or [("fixed", None)]:
                for mode in modes:
                    for arrival_speed in args.arrival_speed:
                        # 闭指提前量仅在 early 模式生效，其他对照模式每档速度只运行一次。
                        for lead in args.lead if mode == "early" else args.lead[:1]:
                            run(mode, lead, args.duration, args.headless, args.object, args.speed,
                                writer, args.aim_x_mm, pitch_deg, offset, summary, strategy, arrival_speed,
                                hand_feedback=args.hand_feedback, max_delay_ms=args.max_delay_ms,
                                thumb_base_offset_deg=args.thumb_base_offset_deg)
