"""球体包围对照：原中段控制、食指避碰、晚拇指接近、提前拇指整形。

807为开发检查集；修正食指自碰后809为新验证集。避碰/晚接近组前300 ms一致。
"""
import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.hand import LeapHand
from controllers.joint_aware_closure import JointAwareClosure
from controllers.sphere_cage import CageConfig,SphereCageClosure
from experiments.buffer_ablation import write_rows
from experiments.finger_closure_ablation import read
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from experiments.mpc_contact_diagnosis import ContactRecord
from experiments.thumb_sphere_reach import contact_rank
from scene_setup.scene import build_scene

CAGE_VARIANTS=(("middle_compliant",None),("index_clearance","clearance"),("thumb_contact","force"),("thumb_preshape","preshape"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[807,809]);parser.add_argument("--cases",type=int,nargs="+",default=list(range(12)))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    previous=ROOT/"results/holding_control_validation";old_rows=read(previous/"trials.csv");old_calls=read(previous/"solver_times.csv")
    model=build_scene("ball");data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    hand=LeapHand(model);dofs=[model.joint(f"hand/{i}").dofadr[0] for i in range(16)]
    rows=[];steps=[];control_rows=[];call_rows=[]
    for seed in args.seeds:
        for case,kind,p,v in scenarios(center,model.opt.gravity,seed=seed):
            if kind!="near" or case not in args.cases:continue
            selected=[r for r in old_calls if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="middle_compliant"]
            schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in selected} if selected else None
            checkpoint=None;clearance_checkpoint=None
            for name,mode in CAGE_VARIANTS:
                holder=[];captured=[];torques=[];all_self_depth=[];thumb_speeds=[];thumb_margins=[];index_speeds=[]
                thumb_joints=[model.joint(f"hand/{i}") for i in range(12,16)]
                thumb_adr=[j.qposadr[0] for j in thumb_joints];thumb_dof=[j.dofadr[0] for j in thumb_joints]
                ranges=model.jnt_range[[j.id for j in thumb_joints]]
                def factory(m,h):
                    control=JointAwareClosure(m,h,"compliant") if mode is None else SphereCageClosure(m,h,mode)
                    holder.append(control);return control
                record=ContactRecord(model,case,name,seed,True)
                thumb_geoms={g for g,p in record.closure.group.items() if p==3}
                def observe(m,d,t,first,close,velocity,reference):
                    record.observe(m,d,t,first,close,velocity,reference)
                    row=record.rows[-1]
                    rank=contact_rank(m,d,record.closure.group)
                    row.update(wrench_rank=rank["scaled_wrench_rank"],loaded_parts=len(rank["active_parts"].split("|")) if rank["active_parts"] else 0,
                        smallest_wrench_singular=rank["smallest_singular"])
                    depth=0.
                    for c in d.contact:
                        if (c.geom1 in thumb_geoms and c.geom2 in record.closure.group) or (c.geom2 in thumb_geoms and c.geom1 in record.closure.group):
                            depth=max(depth,max(0.,-float(c.dist)))
                    row["thumb_self_penetration_mm"]=1000.*depth
                def frame(m,d,info):
                    if info["first_contact_s"] is not None and abs(d.time-info["first_contact_s"]-.3)<m.opt.timestep/4:
                        captured.append(np.r_[d.qpos,d.qvel,d.ctrl].copy())
                    torques.append(float(max(abs(d.qfrc_actuator[dofs]))))
                    thumb_speeds.append(float(max(abs(d.qvel[thumb_dof]))))
                    index_speeds.append(float(max(abs(d.qvel[dofs[:4]]))))
                    thumb_margins.append(float(min(np.r_[d.qpos[thumb_adr]-ranges[:,0],ranges[:,1]-d.qpos[thumb_adr]])))
                    all_self_depth.append(max([0.]+[-float(c.dist) for c in d.contact if
                        (c.geom1 in thumb_geoms and c.geom2 in record.closure.group) or
                        (c.geom2 in thumb_geoms and c.geom1 in record.closure.group)]))
                result,calls,_=run_case(model,p,v,case,kind,seed,"mpc_match",solver_schedule=schedule,
                    hand_controller_factory=factory,step_observer=observe,frame_observer=frame,allow_schedule_extension=mode=="preshape")
                assert max(torques)<=.950001
                if mode!="preshape":assert len(captured)==1
                if checkpoint is None:
                    checkpoint=captured[0]
                    if schedule is None:schedule={round(r["time_s"]/model.opt.timestep):bool(int(r["overrun"])) for r in calls}
                    old=next((r for r in old_rows if int(r["seed"])==seed and int(r["scenario"])==case and r["strategy"]=="middle_compliant"),None)
                    if old:
                        for key in ("held","drift_mm","first_contact_s"):
                            assert np.isclose(result[key],float(old[key]),atol=1e-5,rtol=1e-8),(seed,case,key)
                elif mode=="clearance":clearance_checkpoint=captured[0]
                elif mode=="force":assert np.allclose(captured[0],clearance_checkpoint,atol=1e-12,rtol=0),(seed,case,name,"capture changed")
                late=[r for r in record.rows if r["contact_ms"]>=300.-1e-8]
                history=[] if mode is None else holder[0].rows
                result.update(strategy=name,thumb_loaded_hold_ms=sum(r["thumb_N"]>=.1 for r in late)*2.,
                    three_parts_rank6_hold_ms=sum(r["thumb_N"]>=.1 and r["loaded_parts"]>=3 and r["wrench_rank"]==6 for r in late)*2.,
                    median_thumb_hold_N=median(r["thumb_N"] for r in late) if late else 0.,
                    peak_thumb_hold_N=max([0.]+[r["thumb_N"] for r in late]),
                    max_thumb_self_penetration_mm=max([0.]+[r["thumb_self_penetration_mm"] for r in late]),
                    all_phase_thumb_self_penetration_mm=1000.*max(all_self_depth),
                    all_phase_thumb_peak_speed=max(thumb_speeds),thumb_joint_margin=min(thumb_margins),
                    all_phase_index_peak_speed=max(index_speeds),
                    cage_plans=sum(r["plan_called"] for r in history),cage_failed=sum(r["plan_failed"] for r in history),
                    cage_overruns=sum(r["plan_overrun"] for r in history),actual_hand_peak_torque=max(torques),
                    capture_match=int(mode is None or mode=="force"),capture_reference="index_clearance" if mode=="force" else "middle_compliant",
                    **asdict(CageConfig()))
                rows.append(result);steps.extend(record.rows)
                tags=dict(seed=seed,scenario=case,strategy=name)
                control_rows.extend(dict(**tags,**r) for r in history);call_rows.extend(dict(**r,strategy=name) for r in calls)
            print(seed,case,[(r["strategy"],r["held"],round(r["drift_mm"],1),r["thumb_loaded_hold_ms"],r["cage_failed"]) for r in rows[-4:]],flush=True)
            write_rows(args.output_dir/"trials.csv",rows)
    summary=[]
    for seed in args.seeds:
        for name,_ in CAGE_VARIANTS:
            group=[r for r in rows if r["seed"]==seed and r["strategy"]==name]
            summary.append(dict(seed=seed,strategy=name,n=len(group),held=sum(r["held"] for r in group),qualified=sum(r["qualified"] for r in group),
                thumb_bearing=sum(r["thumb_loaded_hold_ms"]>=50. for r in group),
                three_part_rank6=sum(r["three_parts_rank6_hold_ms"]>=50. for r in group),
                median_drift_mm=median(r["drift_mm"] for r in group),
                max_thumb_self_penetration_mm=max(r["max_thumb_self_penetration_mm"] for r in group),
                cage_plans=sum(r["cage_plans"] for r in group),cage_failed=sum(r["cage_failed"] for r in group),cage_overruns=sum(r["cage_overruns"] for r in group)))
    for file,content in (("summary.csv",summary),("time_series.csv",steps),("cage_control.csv",control_rows),("solver_times.csv",call_rows)):
        if content:write_rows(args.output_dir/file,content)
    for row in summary:print(row,flush=True)


if __name__=="__main__":main()
