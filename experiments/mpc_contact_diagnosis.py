"""复盘801轨迹或测量新轨迹的固定掌面、MPC零速度、MPC速度匹配。

重放历史超时决策，测量钩子不改变运动控制。力/接触/速度都按步初时刻对齐。
逐接触点只保存前300ms，逐步承载记录覆盖完整保持阶段。
"""
import argparse
import csv
from pathlib import Path
from statistics import median
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from experiments.contact_impulse import write_rows
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene

PARTS=("palm","finger1","finger2","finger3","thumb")


def first_time(rows,predicate):
    return next((r["contact_ms"] for r in rows if predicate(r)),None)


def longest_absence(rows,key,dt):
    longest=streak=0.
    for r in rows:
        streak=streak+dt if not r[key] else 0.
        longest=max(longest,streak)
    return round(longest,6)


class ContactRecord:
    def __init__(self,model,scenario,mode,seed=801,recorded=True):
        self.tags=dict(seed=seed,scenario=scenario,mode=mode,decision_source="recorded" if recorded else "measured")
        self.hand=LeapHand(model)
        self.closure=ContactClosure(model,self.hand)
        self.rows=[];self.points=[]
        self.palm_origin=self.rotation_origin=self.relative_origin=None
        self.incoming=None
        self.impulse=np.zeros(3)
        self.angular_impulse=np.zeros(3)

    def observe(self,model,data,t,first,close,velocity,reference):
        elapsed=round((t-first)*1000,6)
        palm=data.body(self.closure.palm)
        object_body=data.body("projectile")
        rotation=palm.xmat.reshape(3,3)
        relative=rotation.T@(object_body.xpos-palm.xpos)
        if self.palm_origin is None:
            self.palm_origin=palm.xpos.copy()
            self.rotation_origin=rotation.copy()
            self.relative_origin=relative.copy()
            dof=model.joint("flight").dofadr[0]
            incoming=velocity[dof:dof+3]
            self.incoming=incoming/max(np.linalg.norm(incoming),1e-12)
        loads=np.zeros(5);geometry=np.zeros(5,dtype=int)
        force=np.zeros(3);moment=np.zeros(3)
        max_slip=0.;near_slide_load=0.
        current=[]
        self.closure.observe(model,data)
        for i,c in enumerate(data.contact):
            if self.closure.projectile not in (c.geom1,c.geom2):continue
            on_second=c.geom2==self.closure.projectile
            other=c.geom1 if on_second else c.geom2
            group=self.closure.group.get(other)
            if group is None:continue
            j=group+1
            geometry[j] |= int(c.dist<=0)
            if c.efc_address<0:continue
            wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
            sign=1. if on_second else -1.
            world=sign*c.frame.reshape(3,3).T@wrench[:3]
            # 净矢量冲量保留方向；各点法向载荷总和还包括相互抵消的夹持力。
            force+=world
            moment+=np.cross(c.pos-object_body.xipos,world)+sign*c.frame.reshape(3,3).T@wrench[3:]
            if wrench[0]<.02:continue
            loads[j]+=wrench[0]
            jo,jh=np.empty((3,model.nv)),np.empty((3,model.nv))
            mujoco.mj_jac(model,data,jo,None,c.pos,object_body.id)
            mujoco.mj_jac(model,data,jh,None,c.pos,model.geom_bodyid[other])
            slip=float(np.linalg.norm(c.frame.reshape(3,3)[1:]@((jo-jh)@velocity)))
            if c.dim!=3 or min(c.friction[:2])<=0:
                raise ValueError("Expected 3D sliding friction contacts")
            eta=float(np.linalg.norm(wrench[1:3]/c.friction[:2])/wrench[0])
            max_slip=max(max_slip,slip)
            if eta>=.9 and slip>=.020:near_slide_load+=wrench[0]
            local=rotation.T@(c.pos-palm.xpos)
            current.append(dict(**self.tags,contact_ms=elapsed,part=PARTS[j],body=model.body(model.geom_bodyid[other]).name,
                normal_N=wrench[0],tangent_N=float(np.linalg.norm(wrench[1:3])),utilization=eta,
                slip_mm_s=1000*slip,palm_x_mm=1000*local[0],palm_y_mm=1000*local[1],palm_z_mm=1000*local[2]))
        if elapsed<300.-1e-8:self.points.extend(current)
        self.impulse+=model.opt.timestep*force
        self.angular_impulse+=model.opt.timestep*moment
        dof=model.joint("flight").dofadr[0]
        object_velocity=velocity[dof:dof+3]
        jo,jp=np.empty((3,model.nv)),np.empty((3,model.nv))
        mujoco.mj_jac(model,data,jo,None,object_body.xpos,object_body.id)
        mujoco.mj_jac(model,data,jp,None,object_body.xpos,palm.id)
        relative_velocity=rotation.T@((jo-jp)@velocity)
        travel=palm.xpos-self.palm_origin
        along=float(travel@self.incoming)
        turn=float(np.rad2deg(np.arccos(np.clip((np.trace(self.rotation_origin.T@rotation)-1)/2,-1,1))))
        row=dict(**self.tags,step_start_s=t,contact_ms=elapsed,dt_ms=1000*model.opt.timestep,
            command_fraction=0. if close is None else float(np.clip((t-close)/.14,0,1)),
            finger_contact=int(any(geometry[1:])),finger_loaded=int(any(loads[1:])),palm_loaded=int(loads[0]>0),
            opposed=int(self.closure.holding),palm_travel_mm=1000*np.linalg.norm(travel),
            palm_along_incoming_mm=1000*along,palm_sideways_mm=1000*np.sqrt(max(0.,travel@travel-along**2)),
            palm_rotation_deg=turn,relative_motion_mm=1000*np.linalg.norm(relative-self.relative_origin),
            object_palm_x_mm=1000*relative[0],object_palm_y_mm=1000*relative[1],object_palm_z_mm=1000*relative[2],
            max_slip_mm_s=1000*max_slip,near_sliding_load_N=near_slide_load,
            object_normal_velocity_m_s=float(object_velocity@(-self.rotation_origin[:,2])),
            object_along_incoming_velocity_m_s=float(object_velocity@self.incoming),
            relative_vx_mm_s=1000*relative_velocity[0],relative_vy_mm_s=1000*relative_velocity[1],relative_vz_mm_s=1000*relative_velocity[2],
            net_impulse_norm_Ns=float(np.linalg.norm(self.impulse)),angular_impulse_norm_Nms=float(np.linalg.norm(self.angular_impulse)),
            **{f"net_impulse_{a}_Ns":float(self.impulse[i]) for i,a in enumerate("xyz")},
            **{p+"_N":float(loads[i]) for i,p in enumerate(PARTS)},
            **{p+"_contact":int(geometry[i]) for i,p in enumerate(PARTS)})
        self.rows.append(row)

    def summary(self,outcome,prior):
        early=[r for r in self.rows if r["contact_ms"]<300.-1e-8]
        first100=[r for r in self.rows if r["contact_ms"]<100.-1e-8]
        late=[r for r in self.rows if r["contact_ms"]>=300.-1e-8]
        first_parts="|".join(p for p in PARTS if self.rows[0][p+"_contact"])
        stable=first_time(early,lambda r:r["opposed"])
        after=[r for r in early if stable is not None and r["contact_ms"]>=stable]
        at100=next((r for r in self.rows if r["contact_ms"]>=100.-1e-8),None)
        return dict(**self.tags,held=outcome["held"],historical_held=int(prior["held"]) if prior else None,first_parts=first_parts,
            first_opposed_ms=stable,opposed_first300_ms=sum(r["opposed"]*r["dt_ms"] for r in early),
            longest_support_loss_after_opposed_ms=longest_absence(after,"opposed",2.) if stable is not None else None,
            max_palm_travel_first300_mm=max(r["palm_travel_mm"] for r in early),
            max_palm_sideways_first300_mm=max(r["palm_sideways_mm"] for r in early),
            max_palm_rotation_first300_deg=max(r["palm_rotation_deg"] for r in early),
            relative_motion_first300_mm=max(r["relative_motion_mm"] for r in early),
            net_hand_impulse_first100_Ns=first100[-1]["net_impulse_norm_Ns"],
            angular_impulse_first100_Nms=first100[-1]["angular_impulse_norm_Nms"],
            first100_normal_load_integral_Ns=sum(sum(r[p+"_N"] for p in PARTS)*.002 for r in first100),
            first100_near_slide_load_integral_Ns=sum(r["near_sliding_load_N"]*.002 for r in first100),
            first100_slip_peak_mm_s=max(r["max_slip_mm_s"] for r in first100),
            first300_max_outgoing_normal_m_s=max(r["object_normal_velocity_m_s"] for r in early),
            object_normal_at100_m_s=at100["object_normal_velocity_m_s"] if at100 else None,
            relative_vz_at100_mm_s=at100["relative_vz_mm_s"] if at100 else None,
            longest_hold_finger_contact_gap_ms=longest_absence(late,"finger_contact",2.) if late else None,
            **{p+"_first_load_ms":first_time(early,lambda r,p=p:r[p+"_N"]>0) for p in PARTS},
            **{p+"_mean_first100_N":float(np.mean([r[p+"_N"] for r in first100])) for p in PARTS})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=801)
    parser.add_argument("--fresh",action="store_true",help="Use measured timing on a fresh trajectory set")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]/"results/catching_mpc_validation"
    prior=[];schedules=[]
    if not args.fresh:
        if args.seed!=801:parser.error("Recorded replay requires seed 801")
        with (root/"trials.csv").open(encoding="utf-8-sig",newline="") as f:prior=list(csv.DictReader(f))
        with (root/"solver_times.csv").open(encoding="utf-8-sig",newline="") as f:schedules=list(csv.DictReader(f))
    model=build_scene("ball");data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    summary,steps,points,replays=[],[],[],[]
    for scenario,kind,p,v in scenarios(center,model.opt.gravity,seed=args.seed):
        if kind!="near":continue
        for mode in ("static_contact","mpc_stop","mpc_match"):
            original=next((r for r in prior if int(r["scenario"])==scenario and r["mode"]==mode),None)
            schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in schedules
                      if int(r["scenario"])==scenario and r["mode"]==mode}
            record=ContactRecord(model,scenario,mode,args.seed,not args.fresh)
            row,_,_=run_case(model,p,v,scenario,kind,args.seed,mode,step_observer=record.observe,
                             solver_schedule=None if args.fresh else schedule)
            if original is not None:
                for key in ("held","first_contact_s","close_s","drift_mm","peak_normal_first100ms_N"):
                    if not np.isclose(float(row[key]),float(original[key]),atol=1e-5,rtol=1e-8):
                        raise ValueError(f"Replay differs {scenario} {mode} {key}: {row[key]} vs {original[key]}")
            summary.append(record.summary(row,original));steps.extend(record.rows);points.extend(record.points);replays.append(row)
        print("Measured" if args.fresh else "Replayed",scenario,flush=True)
    write_rows(args.output_dir/"summary.csv",summary)
    write_rows(args.output_dir/"time_series.csv",steps)
    write_rows(args.output_dir/"contact_points.csv",points)
    write_rows(args.output_dir/"replayed_trials.csv",replays)
    for mode in ("static_contact","mpc_stop","mpc_match"):
        for held in (0,1):
            group=[r for r in summary if r["mode"]==mode and r["held"]==held]
            if group:print(mode,held,len(group),"palm turn median",median(r["max_palm_rotation_first300_deg"] for r in group),
                "sideways median",median(r["max_palm_sideways_first300_mm"] for r in group),
                "opposed ms median",median(r["opposed_first300_ms"] for r in group))
