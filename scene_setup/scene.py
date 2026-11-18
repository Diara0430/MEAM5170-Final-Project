"""3D catching scene. Run with --help for launch options."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np


MODELS = Path(__file__).resolve().parent / "models"
ARM_HOME = [0, -0.247, 0, 0.909, 0, 1.15644, 0]


def load_model(folder, xml):
    directory = MODELS / folder
    assets = {p.name: p.read_bytes() for p in (directory / "assets").iterdir()}
    return mujoco.MjSpec.from_string((directory / xml).read_text(encoding="utf-8"), assets=assets)


def build_scene(shape):
    arm = load_model("ufactory_xarm7", "xarm7_nohand.xml")
    hand = load_model("leap_hand", "right_hand.xml")
    arm.delete(arm.key("home"))
    palm = hand.body("palm_lower")
    palm.quat[:] = [0, 1, 0, 0]
    palm.pos[:] = [0.065, -0.04, 0]
    arm.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10
    arm.attach(hand, prefix="hand/", site=arm.site("attachment_site"))
    world = arm.worldbody
    world.add_light(pos=[0, -1, 2], dir=[0, 0.4, -1])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[2, 2, 0.02], rgba=[0.28, 0.33, 0.38, 1])
    obj = world.add_body(name="projectile")
    obj.add_freejoint(name="flight")
    geom_type, size = {
        "ball": (mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0]),
        "box": (mujoco.mjtGeom.mjGEOM_BOX, [0.032, 0.021, 0.017]),
    }[shape]
    obj.add_geom(name="projectile", type=geom_type, size=size, mass=0.06,
                 friction=[0.8, 0.005, 0.0001], rgba=[1, 0.45, 0.05, 1])
    return arm.compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=["ball", "box"], default="ball")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gust", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    model = build_scene(args.object)
    data = mujoco.MjData(model)
    for i, angle in enumerate(ARM_HOME, 1):
        data.joint(f"joint{i}").qpos[:] = angle
        data.actuator(f"act{i}").ctrl[:] = angle
    mujoco.mj_forward(model, data)
    # Aim the gravity-only trajectory at the open hand at t = 0.35 s.
    target = (data.body("hand/palm_lower").xpos + data.site("hand/tip_2").xpos) / 2
    start = target + [0.50, -0.12, 0.14]
    flight = data.joint("flight")
    flight.qpos[:3] = start
    flight.qvel[:3] = (target - start) / 0.35 - 0.5 * model.opt.gravity * 0.35
    if args.object == "box":
        flight.qvel[3:] = [1, 2, 0.5]
    mujoco.mj_forward(model, data)
    onset = rng.uniform(0.06, 0.15)
    duration = rng.uniform(0.04, 0.10)
    accel = rng.uniform(-2, 2, 3)
    obj = model.body("projectile").id
    if args.gust:
        print(f"Gust: {onset:.3f} s for {duration:.3f} s, acceleration {accel} m/s^2")
    first_contact = None
    def step():
        nonlocal first_contact
        active = args.gust and onset <= data.time < onset + duration
        data.xfrc_applied[obj, :3] = model.body_mass[obj] * accel if active else 0
        mujoco.mj_step(model, data)
        if first_contact is None:
            for contact in data.contact:
                bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                if obj in bodies and any((model.body(int(b)).name or "").startswith("hand/") for b in bodies):
                    first_contact = data.time
                    break

    if args.headless:
        while data.time < 1.5:
            step()
    else:
        from mujoco import viewer
        with viewer.launch_passive(model, data) as window:
            window.cam.lookat[:] = [0.45, 0, 0.45]
            window.cam.distance = 1.75
            window.cam.azimuth = 110
            window.cam.elevation = -15
            while window.is_running():
                before = time.monotonic()
                step()
                window.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - before)))
    print(f"First hand contact (s): {first_contact}; final object position: {flight.qpos[:3]}")


if __name__ == "__main__":
    main()
