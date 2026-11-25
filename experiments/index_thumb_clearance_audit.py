"""静态网格筛查食指让位姿态；不修改实际仿真，不代替动态碰撞检查。"""
import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from experiments.buffer_ablation import write_rows
from experiments.intercept_tracking import initial
from scene_setup.scene import build_scene


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    model=build_scene("ball");data=initial(model);hand=LeapHand(model);groups=ContactClosure(model,hand).group
    thumb=[g for g,p in groups.items() if p==3 and model.geom_contype[g]]
    index=[g for g,p in groups.items() if p==0 and model.geom_contype[g]]
    adr=[model.joint(f"hand/{i}").qposadr[0] for i in range(16)]
    rows=[]
    for name,goal in (("original",hand.closed[:4]),("clearance",np.array([0.,.8,.9,.5]))):
        for thumb_name,thumb_goal in (("preshape",np.array([.8,.5,1.,.65])),("original_end",hand.closed[12:])):
            fixed_gap=1.;grid_gap=1.
            for s in np.linspace(0,1,17):
                for t in np.linspace(0,1,17):
                    q=np.array(hand.closed)*s;q[:4]=goal*s;q[12:]=thumb_goal*t
                    data.qpos[adr]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                    gap=min(mujoco.mj_geomDistance(model,data,a,b,.1,None) for a in thumb for b in index)
                    grid_gap=min(grid_gap,float(gap))
                    if t==1.:fixed_gap=min(fixed_gap,float(gap))
            rows.append(dict(index_pose=name,thumb_pose=thumb_name,fixed_thumb_path_gap_mm=1000.*fixed_gap,
                two_progress_grid_gap_mm=1000.*grid_gap,index_1=float(goal[1]),index_2=float(goal[2]),index_3=float(goal[3])))
    write_rows(args.output_dir/"static_clearance.csv",rows)
    for row in rows:print(row,flush=True)


if __name__=="__main__":main()
