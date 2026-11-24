"""直接运行导出原速/半速/自适应的并排视频；--view 打开交互式 MuJoCo 窗口。

python experiments/replay_wrist.py --view --seed 435 --mode adaptive
python experiments/replay_wrist.py --seeds 435 454 --output-dir results/wrist_replay
视频额外依赖 pillow、imageio、imageio-ffmpeg。所有视频标签使用英文。
"""
import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controllers.wrist import WristProgress
from controllers.catch_stability import ArmBuffer, SlipGrip
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import run
from experiments.gust_timing import TimingTrial, gust_for_seed


def replay(seed, mode, view=False, speed=.5, record=False):
    wrist = WristProgress("slow" if mode in ("buffer", "grip", "combined") else mode)
    buffer = ArmBuffer() if mode in ("buffer", "combined") else None
    grip = SlipGrip() if mode in ("grip", "combined") else None
    timing = TimingTrial(gust_for_seed(seed), False)
    frames = []
    renderer = None
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = .55
    camera.azimuth = 105
    camera.elevation = -25

    def capture(model, data, t, tilt, hold):
        nonlocal renderer
        # 每 20 ms 采样一帧，25 fps 播放得到二分之一速视频。
        if round(t/model.opt.timestep) % round(.020/model.opt.timestep):
            return
        from PIL import Image, ImageDraw, ImageFont
        if renderer is None:
            renderer = mujoco.Renderer(model, height=360, width=480)
            camera.lookat[:] = data.body("hand/palm_lower").xpos
        renderer.update_scene(data, camera=camera)
        panel = Image.new("RGB", (480, 456), "#10202d")
        panel.paste(Image.fromarray(renderer.render()), (0, 96))
        draw = ImageDraw.Draw(panel)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        phase = "Flight / closure" if t < tilt else "Tilt" if t < hold else "Final hold"
        drift = wrist.trace[-1]["drift_mm"] if wrist.trace else 0.
        draw.text((12, 8), f"Gust {seed} | {mode.upper()} | {t:.2f} s", font=font, fill="white")
        draw.text((12, 30), f"{phase} | Drift {drift:.1f} mm (limit 25)", font=font, fill="#ffb76b" if drift>25 else "white")
        # rate 为轨迹进度倍率。实际角速度还取决于平滑轨迹所在阶段。
        draw.text((12, 52), f"Progress rate {wrist.rate:.2f}x | Tilt {100*wrist.progress:.0f}%", font=font, fill="white")
        note = f"Pretilt recoil {buffer.actual_mm:.1f} mm" if buffer else "0.5x playback | Same gust and launch"
        draw.text((12, 74), note, font=font, fill="#99b5c9")
        frames.append(np.asarray(panel).copy())

    try:
        run("early", .04, .19, not view, shape="box", speed=speed,
            pitch_deg=40., palm_x_mm=0., arrival_speed="fast", timing_trial=timing,
            wrist_control=wrist, frame_observer=capture if record else None,
            arm_buffer=buffer, post_close_control=grip)
    finally:
        if renderer is not None:
            renderer.close()
    return dict(seed=seed, mode=mode, **timing.outcome), frames


def video(seed, output, stability=False):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    rows, panels = [], []
    for mode in (("slow", "buffer", "combined") if stability else ("fixed", "slow", "adaptive")):
        row, frames = replay(seed, mode, record=True)
        # 结束较早的试验冻结末帧，并清楚显示结果，保持共同释放时刻对齐。
        result = "PASS" if row["passed"] else "FAIL"
        last = Image.fromarray(frames[-1])
        draw = ImageDraw.Draw(last)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
        draw.rectangle((0, 96, 480, 142), fill="#10202d")
        draw.text((12, 99), f"Finished: {result} | Max drift {row['max_drift_mm']:.1f} mm", font=font, fill="white")
        draw.text((12, 120), "Frozen after evaluation", font=font, fill="white")
        frames.append(np.asarray(last).copy())
        rows.append(row)
        panels.append(frames)
    path = output/f"gust_{seed}_comparison.mp4"
    with imageio.get_writer(path, fps=25, codec="libx264", quality=8, macro_block_size=8) as writer:
        for i in range(max(map(len, panels))+50):
            frame = np.concatenate([p[min(i, len(p)-1)] for p in panels], axis=1)
            writer.append_data(frame)
            if i == 75:
                Image.fromarray(frame).save(output/f"gust_{seed}_preview.png")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--seed", type=int, default=435)
    parser.add_argument("--mode", choices=("fixed", "slow", "adaptive", "buffer", "grip", "combined"), default="adaptive")
    parser.add_argument("--stability", action="store_true", help="Compare slow baseline, recoil, and recoil plus grip feedback")
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[435, 454])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]/"results/wrist_replay")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.view:
        replay(args.seed, args.mode, view=True, speed=args.speed)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        rows = []
        for seed in args.seeds:
            rows.extend(video(seed, args.output_dir, args.stability))
        write_rows(args.output_dir/"replay_results.csv", rows)
