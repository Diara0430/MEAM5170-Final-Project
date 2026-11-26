"""Render a recorded MuJoCo rollout with the observations and decision available at each frame."""
import argparse
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scene_setup.scene import build_scene


def dot(scene, xyz, radius, color):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, radius), np.asarray(xyz), np.eye(3).ravel(), np.array(color))
    geom.emission = .35
    scene.ngeom += 1


def render(sample, output):
    shape = sample.get("shape", "ball")
    model = build_scene(shape)
    model.vis.headlight.ambient[:] = [.38, .38, .38]
    model.vis.headlight.diffuse[:] = [.60, .60, .60]
    data = mujoco.MjData(model)
    frames = sample["path"]
    ball = np.array([f["ball"] for f in frames])
    hand = np.array([f["hand"] for f in frames])
    positions = np.vstack((ball, hand, np.zeros(3)))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (positions.min(axis=0) + positions.max(axis=0))/2
    camera.elevation = -10
    heading = ball[0, :2] - hand[0, :2]
    camera.azimuth = np.rad2deg(np.arctan2(heading[1], heading[0])) + 90
    camera.distance = max(2.1, 1.62*np.ptp(positions[:, 2]), 1.2*np.linalg.norm(heading))
    detail = mujoco.MjvCamera()
    detail.lookat[:] = hand[0]
    detail.azimuth, detail.elevation, detail.distance = camera.azimuth + 23, -18, .55
    touch = sample["contact"]
    pictures, durations, history = [], [], {}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except OSError:
            font = ImageFont.load_default()
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        for index, frame in enumerate(frames):
            now = frame["time_s"]
            data.qpos[:] = frame["qpos"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            if frame["measurement"] is not None:
                history[frame["measurement_time"]] = frame["measurement"]
            for point in list(history.values())[::2]:
                dot(renderer.scene, point, .010, [.1, .66, 1., 1.])
            prediction = frame["forecast"]
            if prediction is not None:
                stamp, p, v = prediction
                p, v = np.array(p), np.array(v)
                g = np.array(sample["first_estimate"]["g"])
                # Stop the dotted curve when it leaves the catch region.
                for t in np.arange(now + .04, now + 1.20, .04):
                    h = t - stamp
                    xyz = p + v*h + .5*g*h*h
                    if xyz[2] < max(.025, hand[0, 2] - .16):
                        break
                    dot(renderer.scene, xyz, .010, [.96, .83, .24, 1.])
            if frame["intercept"] is not None:
                dot(renderer.scene, frame["intercept"], .021, [.92, .2, .64, .82])
            dot(renderer.scene, frame["hand"], .013, [.16, .90, .47, 1.])
            overview = Image.fromarray(renderer.render()).crop((100, 0, 540, 360))
            renderer.update_scene(data, camera=detail)
            if frame["intercept"] is not None:
                dot(renderer.scene, frame["intercept"], .010, [.92, .2, .64, .87])
            dot(renderer.scene, frame["hand"], .007, [.16, .90, .47, 1.])
            inset = Image.fromarray(renderer.render()).crop((190, 44, 450, 344))
            canvas = Image.new("RGB", (708, 424), "#172937")
            canvas.paste(overview, (0, 32))
            canvas.paste(inset, (446, 92))
            draw = ImageDraw.Draw(canvas)
            draw.text((454, 62), "Palm close-up", fill="white", font=font)
            if touch is not None and now >= touch:
                phase = "Hand contact"
                if now > touch + .30:
                    phase = "Holding" if sample["held"] else "After contact"
            elif frame["intercept"] is not None:
                phase = "Moving to selected interception point"
            elif prediction is not None:
                phase = "Predicting flight"
            else:
                phase = f"Observing {shape}"
            draw.text((12, 8), f"{now:.2f} s   {phase}", fill="white", font=font)
            for x, label, color in [(12, "Observations", "#32b8ff"), (153, "Forecast", "#f6d640"),
                                    (269, "Selected point", "#f864bc"), (449, "Palm center", "#45e98a")]:
                draw.ellipse((x,405,x+8,413), fill=color)
                draw.text((x+15,399), label, fill="white", font=font)
            pictures.append(canvas)
            gap = frames[index+1]["time_s"] - now if index+1 < len(frames) else .50
            durations.append(max(20, round(2000*gap)))  # half-speed replay
    # Share one palette so stationary pixels stay identical between GIF frames.
    swatches = Image.new("RGB", (354, 212*5))
    for i, frame in enumerate(np.linspace(0, len(pictures)-1, 5, dtype=int)):
        swatches.paste(pictures[frame].resize((354, 212)), (0, i*212))
    palette = swatches.quantize(colors=216)
    colors = palette.getpalette()[:216*3]
    colors.extend([50,184,255, 246,214,64, 248,100,188, 69,233,138,
                   230,60,180, 60,230,120, 255,120,20, 32,108,255])
    palette.putpalette(colors + [0]*(768-len(colors)))
    pictures = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in pictures]
    pictures[0].save(output, save_all=True, append_images=pictures[1:], duration=durations,
                     loop=0, optimize=True, disposal=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.recording.suffix == ".gz" else open
    with opener(args.recording, "rt", encoding="utf-8") as source:
        render(json.load(source), args.output)
