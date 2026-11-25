"""固定真实捕获状态，审查拇指碰撞网格到球的距离与关节限位内的静态可达性。

只操作独立MjData探针，不移动实际仿真中的球或关节。不把静态IK当作动态抓取成功。
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from experiments.buffer_ablation import write_rows
from scene_setup.scene import build_scene


def geometry(model):
    hand=LeapHand(model);group=ContactClosure(model,hand).group
    thumb=[g for g,p in group.items() if p==3 and model.geom_contype[g]]
    joints=[model.joint(f"hand/{i}") for i in range(12,16)]
    adr=[j.qposadr[0] for j in joints]
    ranges=model.jnt_range[[j.id for j in joints]]
    lo=np.maximum(ranges[:,0],hand.limits[12:,0]);hi=np.minimum(ranges[:,1],hand.limits[12:,1])
    return thumb,adr,lo,hi,group


def probe_reach(model,data,geoms,starts=10):
    thumb,adr,lo,hi,group=geometry(model);obj=model.geom("projectile").id
    probe=mujoco.MjData(model);probe.qpos[:]=data.qpos
    seed=data.qpos[adr].copy()
    def distances(q):
        probe.qpos[adr]=q;mujoco.mj_kinematics(model,probe);mujoco.mj_comPos(model,probe)
        return np.array([mujoco.mj_geomDistance(model,probe,g,obj,.5,None) for g in geoms])
    initial=float(min(distances(seed)))
    rng=np.random.default_rng(5170)
    seeds=[seed,np.clip(np.zeros(4),lo+1e-6,hi-1e-6),np.clip(np.array([1.25,.8,1.1,.65]),lo,hi)]
    seeds.extend(rng.uniform(lo,hi) for _ in range(max(0,starts-len(seeds))))
    best=(initial,seed.copy())
    for start in seeds:
        def residual(q):
            gap=float(min(distances(q)))
            return np.r_[max(gap+.0005,0.),.0002*(q-seed)]
        result=least_squares(residual,np.clip(start,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=120,
                             diff_step=1e-4,ftol=1e-8,xtol=1e-8,gtol=1e-9)
        gap=float(min(distances(result.x)))
        if gap<best[0]:best=(gap,result.x.copy())
    distances(best[1]);mujoco.mj_forward(model,probe)
    collisions=[c for c in probe.contact if obj not in (c.geom1,c.geom2) and c.dist<-.001
                and c.geom1 in group and c.geom2 in group]
    return dict(initial_gap_mm=1000.*initial,best_gap_mm=1000.*best[0],
                best_q=best[1],max_joint_change=float(max(abs(best[1]-seed))),self_penetrations=len(collisions))


def contact_rank(model,data,group):
    obj=model.geom("projectile").id;center=data.body("projectile").xipos;radius=model.geom_size[obj,0]
    columns=[];parts=set();loads=np.zeros(5)
    for i,c in enumerate(data.contact):
        if obj not in (c.geom1,c.geom2) or c.efc_address<0:continue
        other=c.geom1 if c.geom2==obj else c.geom2
        part=group.get(other)
        if part is None:continue
        force=np.empty(6);mujoco.mj_contactForce(model,data,i,force)
        if force[0]<.02:continue
        parts.add(part);loads[part+1]+=force[0]
        r=(c.pos-center)/radius
        cross=np.array([[0.,-r[2],r[1]],[r[2],0.,-r[0]],[-r[1],r[0],0.]])
        columns.append(np.vstack((np.eye(3),cross)))
    singular=np.linalg.svd(np.hstack(columns),compute_uv=False) if columns else np.zeros(6)
    rank=int(sum(singular>1e-3*max(singular,default=1.)))
    return dict(active_contacts=len(columns),active_parts="|".join(str(p) for p in sorted(parts)),
                scaled_wrench_rank=rank,smallest_singular=float(singular[-1]) if len(singular)==6 else 0.,
                thumb_load_N=loads[4])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    model=build_scene("ball");thumb,adr,lo,hi,group=geometry(model)
    tip=[g for g in thumb if model.geom_bodyid[g]==model.body("hand/thumb_fingertip").id]
    rows=[]
    for seed,case in ((806,11),(807,1)):
        source=ROOT/f"results/holding_replays/seed{seed}_case{case}_middle_compliant.npz"
        with np.load(source,allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"]));first=meta["outcome"]["first_contact_s"]
            for age in (.30,.60):
                index=min(len(z["time"])-1,int(np.searchsorted(z["time"],first+age)))
                data=mujoco.MjData(model);data.qpos[:]=z["qpos"][index];data.qvel[:]=z["qvel"][index]
                data.ctrl[:]=z["ctrl"][index];mujoco.mj_forward(model,data)
                palm=data.body("hand/palm_lower");local=palm.xmat.reshape(3,3).T@(data.body("projectile").xpos-palm.xpos)
                rank=contact_rank(model,data,group)
                for kind,geoms in (("thumb_tip",tip),("any_thumb_link",thumb)):
                    result=probe_reach(model,data,geoms)
                    q=result.pop("best_q")
                    row=dict(seed=seed,scenario=case,age_ms=age*1000.,target=kind,**result,**rank,
                        **{f"object_{a}_mm":1000.*local[i] for i,a in enumerate("xyz")},
                        **{f"q_{i+12}_rad":v for i,v in enumerate(q)})
                    rows.append(row);print(row,flush=True)
    write_rows(args.output_dir/"reach_audit.csv",rows)


if __name__=="__main__":main()
