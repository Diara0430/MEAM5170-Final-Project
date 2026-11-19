"""From repository root: python -m experiments.static_grasp --object box

Use --open-hand for the negative control and --headless for automated runs.
Inspired by the LEAP static grasp experiments at github.com/jason3535/dexhand-sim.
Here the existing xArm7 tilts the hand under gravity. Pass requires <=25 mm
of palm-relative drift throughout tilting and holding, no object-floor contact
during that interval, finger contact throughout the last 0.5 s, and >=90 deg
of measured palm rotation. Exit code 0 means pass and 1 means fail.
"""
import argparse
from contextlib import nullcontext
import time

import mujoco
import numpy as np

from controllers.hand import LeapHand
from scene_setup.scene import build_scene


def contacts_with(data, object_geom):
    others = set()
    for contact in data.contact:
        if contact.dist <= 0:
            if contact.geom1 == object_geom:
                others.add(contact.geom2)
            elif contact.geom2 == object_geom:
                others.add(contact.geom1)
    return others


def run(shape, headless, open_hand):
    model = build_scene(shape)
    data = mujoco.MjData(model)
    hand = LeapHand(model)
    # Palm up. Joint 6 rotates the wrist through the test tilt.
    home = [0., .4, 0., 2., 0., -1.54, 0.]
    arm_dofs = []
    for i, angle in enumerate(home, 1):
        data.joint(f"joint{i}").qpos[:] = angle
        data.actuator(f"act{i}").ctrl[:] = angle
        arm_dofs.append(model.joint(f"joint{i}").dofadr[0])
    mujoco.mj_forward(model, data)
    palm = data.body("hand/palm_lower")
    obj = data.body("projectile")
    data.joint("flight").qpos[:3] = palm.xpos + palm.xmat.reshape(3, 3) @ [0., -.055, -.105]
    mujoco.mj_forward(model, data)
    object_geom = model.geom("projectile").id
    floor = model.geom("floor").id
    if contacts_with(data, object_geom):
        raise RuntimeError("Object must start clear of the hand and floor")
    finger_geoms = {g for g, b in enumerate(model.geom_bodyid)
                    if (model.body(b).name or "").startswith("hand/") and b != palm.id}
    wrist = data.actuator("act6")
    dt = model.opt.timestep
    passed = True
    max_drift = 0.
    reference = None
    initial_normal = -palm.xmat.reshape(3, 3)[:, 2].copy()
    if headless:
        display = nullcontext(None)
    else:
        from mujoco import viewer
        display = viewer.launch_passive(model, data)
    with display as window:
        if window:
            window.cam.lookat[:] = palm.xpos
            window.cam.distance = .65
            window.cam.elevation = -25
        for k in range(round(2.5 / dt)):
            t = k * dt
            if not open_hand:
                if t < .25:
                    hand.command(data, hand.open_pose, hand.pregrasp, t / .25)
                else:
                    hand.command(data, hand.pregrasp, hand.closed, (t - .25) / .55)
            tilt = np.clip((t - 1.) / 1., 0., 1.)
            wrist.ctrl[:] = home[5] + np.deg2rad(105) * tilt * tilt * (3 - 2 * tilt)
            data.qfrc_applied[arm_dofs] = data.qfrc_bias[arm_dofs]
            mujoco.mj_step(model, data)
            if t >= 1.:
                relative = palm.xmat.reshape(3, 3).T @ (obj.xpos - palm.xpos)
                if reference is None:
                    reference = relative.copy()
                drift = np.linalg.norm(relative - reference)
                max_drift = max(max_drift, drift)
                touching = contacts_with(data, object_geom)
                passed &= floor not in touching and drift <= .025
                if t >= 2.:
                    passed &= bool(touching & finger_geoms)
            if window:
                if not window.is_running():
                    return False
                if k % 8 == 0:
                    window.sync()
                    time.sleep(8 * dt)
    final_normal = -palm.xmat.reshape(3, 3)[:, 2]
    angle = np.rad2deg(np.arccos(np.clip(initial_normal @ final_normal, -1., 1.)))
    passed &= angle >= 90.
    print(f"{shape}: {'PASS' if passed else 'FAIL'}, tilt {angle:.1f} deg, "
          f"max palm-relative drift {1000*max_drift:.1f} mm")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=["box", "ball"], default="box")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--open-hand", action="store_true", help="Control trial without closing fingers")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.object, args.headless, args.open_hand) else 1)
