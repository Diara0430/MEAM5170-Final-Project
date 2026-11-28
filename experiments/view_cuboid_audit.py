"""Play a saved cuboid audit replay using the native MuJoCo viewer."""
import argparse
import gzip
import json
from pathlib import Path
import time
import mujoco
import mujoco.viewer
from scene_setup.scene import build_scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('replay',type=Path,help='Path to an audit *_replay.json.gz file')
    p.add_argument('--speed',type=float,default=.25,help='Playback speed, e.g. 0.25 for quarter speed')
    p.add_argument('--loop',action='store_true')
    args=p.parse_args()
    if not 0 < args.speed <= 4: p.error('speed must be in (0,4]')
    with gzip.open(args.replay,'rt',encoding='utf-8') as f:sample=json.load(f)
    frames=sample['path']
    if not frames: raise ValueError('The replay contains no saved frames')
    model=build_scene(sample.get('shape','box'));data=mujoco.MjData(model)
    print(f"Recorded run: seed={sample['seed']} case={sample['scenario']} held={sample['held']} qualified={sample['qualified']}",flush=True)
    with mujoco.viewer.launch_passive(model,data) as viewer:
        while viewer.is_running():
            start=time.perf_counter();t0=frames[0]['time_s']
            for frame in frames:
                if not viewer.is_running():return
                deadline=start+(frame['time_s']-t0)/args.speed
                while viewer.is_running() and time.perf_counter()<deadline:
                    time.sleep(min(.01,max(0.,deadline-time.perf_counter())))
                data.qpos[:]=frame['qpos'];mujoco.mj_forward(model,data);viewer.sync()
            if not args.loop:
                print('Replay finished. Close the viewer to exit.',flush=True)
                while viewer.is_running():viewer.sync();time.sleep(.05)
                return

if __name__=='__main__':main()
