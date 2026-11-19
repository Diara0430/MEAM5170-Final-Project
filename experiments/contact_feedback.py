"""用配对工况比较固定闭合和接触反馈。图表为英文，时间以发射为零点。"""
import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.contact_impulse import write_rows
from experiments.dynamic_catch import HOLD_DURATION_S, run
from experiments.gust_timing import TimingTrial, gust_for_seed


class ContactTrace:
    """消费 dynamic_catch 已有的逐点记录；同一步多个接触点按步汇总。"""
    def __init__(self):
        self.spin = {}
        self.slip = {}
        self.finger_steps = set()

    def writerow(self, row):
        _, _, t, link, spin, slip, floor, *_ = row
        ms = round(1000 * float(t))
        self.spin[ms] = float(spin)
        if link and not floor:
            self.slip.setdefault(ms, []).append(float(slip))
            if link != "hand/palm_lower":
                self.finger_steps.add(ms)

    def metrics(self, first):
        # 滑速先对同一步的各接触点平均，再对有接触的步平均。无接触不计为零滑速。
        first_ms = None if first is None else round(first * 1000)
        early_slip = [np.mean(v) for t, v in self.slip.items()
                      if first_ms is not None and first_ms <= t < first_ms + 50]
        streak = longest = 0
        times = sorted(self.spin)
        if len(times) >= 2:
            dt_ms = times[1] - times[0]
            hold_start_ms = times[-1] + dt_ms - round(1000 * HOLD_DURATION_S)
            for ms in times:
                if ms < hold_start_ms:
                    continue
                streak = streak + dt_ms if ms in self.finger_steps else 0
                longest = max(longest, streak)
        return dict(spin_20ms_rad_s=self.spin.get(first_ms + 20) if first is not None else None,
                    spin_50ms_rad_s=self.spin.get(first_ms + 50) if first is not None else None,
                    mean_contact_slip_first50_m_s=np.mean(early_slip) if early_slip else None,
                    contact_steps_first50=len(early_slip), longest_finger_contact_in_hold_ms=longest)


def trial(shape, pitch, speed, seed, feedback):
    # 固定提前量与掌位。gust 使用原实验的种子、脉冲与固定到达时刻。
    pulse = (0., 0., 0.) if seed < 0 else gust_for_seed(seed)
    timing = TimingTrial(pulse, online=False)
    trace = ContactTrace()
    palm_mm = 10. if shape == "box" else 0.
    run("early", .04, .19, True, shape=shape, pitch_deg=pitch, palm_x_mm=palm_mm,
        arrival_speed=speed, writer=trace, timing_trial=timing, hand_feedback=feedback)
    return dict(shape=shape, pitch_deg=pitch, palm_x_mm=palm_mm, speed=speed, seed=seed,
                controller="feedback" if feedback else "fixed", gust_onset_s=pulse[0],
                gust_duration_s=pulse[1], gust_az_m_s2=pulse[2], **timing.outcome,
                **trace.metrics(timing.outcome["first_contact_s"]))


def plot(rows, path):
    # 输出独立的矢量图，在项目已有的 MuJoCo 环境即可运行。一个点代表一对相同阵风。
    gust = [row for row in rows if row["seed"] >= 0]
    pairs = {name: {r["seed"]: r for r in gust if r["controller"] == name}
             for name in ("fixed", "feedback")}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="425" viewBox="0 0 800 425">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial, sans-serif" font-size="14" fill="#223344" text-anchor="middle">',
           '<text x="400" y="33" font-size="22">Tilted box under 24 paired vertical gusts</text>',
           '<text x="400" y="57">Each point is one gust. The dashed line marks equal values.</text>']
    for left, key, title, unit in ((76, "spin_50ms_rad_s", "Spin at 50 ms", "rad/s"),
                                   (471, "mean_contact_slip_first50_m_s", "Mean contact slip, first 50 ms", "m/s")):
        points = [(r[key], pairs["feedback"][s][key]) for s, r in pairs["fixed"].items()]
        points = [(x, y) for x, y in points if x is not None and y is not None]
        limit = max((max(x, y) for x, y in points), default=1.) * 1.05 or 1.
        size, bottom = 244, 345
        svg.extend([f'<text x="{left+122}" y="88" font-size="17">{title}</text>',
                    f'<path d="M{left} 101 V345 h244" stroke="#334455" fill="none"/>',
                    f'<path d="M{left} 345 l244 -244" stroke="#8899aa" stroke-dasharray="5 4"/>',
                    f'<text x="{left+122}" y="401">Fixed ({unit})</text>',
                    f'<text transform="translate({left-54} 223) rotate(-90)">Feedback ({unit})</text>'])
        for tick in (0., .5, 1.):
            label = f"{tick * limit:.2g}"
            svg.extend([f'<text x="{left+size*tick}" y="367">{label}</text>',
                        f'<text x="{left-8}" y="{bottom-size*tick+4}" text-anchor="end">{label}</text>'])
        for x, y in points:
            svg.append(f'<circle cx="{left+size*x/limit:.2f}" cy="{bottom-size*y/limit:.2f}" '
                       'r="4" fill="#176b87" fill-opacity="0.72"/>')
    path.write_text("\n".join([*svg, "</g></svg>"]), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = [(shape, pitch, speed, -1) for shape, pitch in (("ball", 0.), ("box", 40.))
             for speed in ("slow", "medium", "fast")]
    cases.extend(("box", 40., "fast", seed) for seed in range(24))
    rows = [trial(*case, feedback) for case in cases for feedback in (False, True)]
    write_rows(args.output_dir / "trials.csv", rows)
    plot(rows, args.output_dir / "paired_feedback_en.svg")
