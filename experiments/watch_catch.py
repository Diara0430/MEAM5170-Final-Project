"""交互观察真实MuJoCo试验：先求解并记录物理状态，再慢速播放，避免绘图改变MPC时序。

默认复现seed=805/case=0的四组缓冲对照，使用已有实验的规划超时决策。
Space暂停；N/P切换策略；R重播；C跳到首触前；[和]逐步；T切换相机；-/=调速度。
所有文字均为英文。播放不重新求解动力学，不支持通过拖动物体改变本次试验。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import queue
import sys
import time

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,""):sys.path.insert(0,str(ROOT))
from controllers.contact_admittance import ContactAdmittance
from controllers.contact_closure import ContactClosure
from controllers.hand import LeapHand
from controllers.joint_aware_closure import ClosureConfig,JointAwareClosure
from controllers.friction_hold import FrictionHoldClosure,HoldConfig
from controllers.sphere_cage import CageConfig,SphereCageClosure
from controllers.discrete_arm_filter import DiscreteArmFilter
from controllers.measured_contact_reference import MeasuredContactReference
from controllers.scheduled_arm_tracker import ScheduledArmTracker
from controllers.coupled_hand_forecast import CoupledHandForecast
from experiments.buffer_ablation import VARIANTS
from experiments.finger_closure_ablation import FINGER_VARIANTS
from experiments.holding_control_ablation import HOLD_VARIANTS
from experiments.sphere_cage_ablation import CAGE_VARIANTS
from experiments.arm_filter_ablation import ARM_VARIANTS
from experiments.contact_delay_ablation import DELAY_VARIANTS,DELAY_REPLAY_NAMES
from experiments.hand_forecast_ablation import HAND_FORECAST_VARIANTS,HAND_REPLAY_NAMES
from experiments.intercept_candidates import scenarios
from experiments.intercept_tracking import initial
from experiments.mpc_ball_catch import run_case
from scene_setup.scene import build_scene

LABELS=("Palm","Finger 1","Finger 2","Finger 3","Thumb")
DEFAULT_RESULTS=ROOT/"results/contact_buffer_validation"
MAX_POINTS=32


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def source_signature(results_dir):
    # 缓存同时绑定控制器、场景/XML、回放器及库版本，修改代码后重新记录。
    digest=hashlib.sha256((mujoco.__version__+np.__version__).encode())
    files=list((ROOT/"controllers").glob("*.py"))+list((ROOT/"scene_setup").glob("*.py"))
    files+=list((ROOT/"scene_setup/models").rglob("*.xml"))
    files+=[Path(__file__),ROOT/"experiments/mpc_ball_catch.py",ROOT/"experiments/active_ball_catch.py",
            ROOT/"experiments/intercept_tracking.py",ROOT/"experiments/intercept_candidates.py",
            ROOT/"experiments/contact_delay_ablation.py",ROOT/"experiments/hand_forecast_ablation.py"]
    for path in sorted(files):digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
    for name in ("trials.csv","solver_times.csv","hand_planner.csv"):
        path=results_dir/name
        if path.exists():digest.update(path.read_bytes())
    return digest.hexdigest()


class StateRecorder:
    def __init__(self,model,duration):
        self.duration=duration
        self.support=ContactClosure(model,LeapHand(model))
        self.loads=np.zeros(5);self.points=np.zeros((MAX_POINTS,3));self.point_count=0
        self.normal=None;self.fraction=0.
        self.eta=0.;self.slip=0.
        self.time=[];self.qpos=[];self.qvel=[];self.ctrl=[];self.object_pos=[]
        self.diag=[];self.load_history=[];self.point_history=[];self.point_counts=[]
        self.self_points=[];self.self_counts=[];self.self_peak=None

    def contact_step(self,model,data,t,first,close,velocity,reference):
        # 此处接触力来自刚完成的物理步，不能用回放时mj_forward重算的力替代。
        if self.normal is None:self.normal=-data.body("hand/palm_lower").xmat.reshape(3,3)[:,2].copy()
        self.support.observe(model,data)
        self.loads[:]=0.;self.points[:]=0.;self.point_count=0;self.eta=0.;self.slip=0.
        self.fraction=0. if close is None else float(np.clip((t-close)/self.duration,0,1))
        for i,c in enumerate(data.contact):
            if self.support.projectile not in (c.geom1,c.geom2) or c.efc_address<0:continue
            other=c.geom1 if c.geom2==self.support.projectile else c.geom2
            group=self.support.group.get(other)
            if group is None:continue
            wrench=np.empty(6);mujoco.mj_contactForce(model,data,i,wrench)
            self.loads[group+1]+=max(0.,wrench[0])
            if wrench[0]>=.02 and self.point_count<MAX_POINTS:
                self.points[self.point_count]=c.pos;self.point_count+=1
            if wrench[0]>=.02:
                self.eta=max(self.eta,float(np.linalg.norm(wrench[1:3]/c.friction[:2])/wrench[0]))
                jo,jh=np.empty((3,model.nv)),np.empty((3,model.nv))
                mujoco.mj_jac(model,data,jo,None,c.pos,model.body("projectile").id)
                mujoco.mj_jac(model,data,jh,None,c.pos,model.geom_bodyid[other])
                self.slip=max(self.slip,float(np.linalg.norm(c.frame.reshape(3,3)[1:]@((jo-jh)@velocity))))

    def frame(self,model,data,info):
        first=info["first_contact_s"]
        normal_speed=0. if self.normal is None else float(data.joint("flight").qvel[:3]@self.normal)
        self.time.append(float(data.time));self.qpos.append(data.qpos.copy());self.qvel.append(data.qvel.copy())
        self.ctrl.append(data.ctrl.copy());self.object_pos.append(data.body("projectile").xpos.copy())
        self_depth=0.;self_points=np.zeros((8,3));self_count=0
        for contact in data.contact:
            a,b=self.support.group.get(contact.geom1),self.support.group.get(contact.geom2)
            if a is None or b is None or 3 not in (a,b):continue
            depth=max(0.,-float(contact.dist));self_depth=max(self_depth,depth)
            if depth>.001 and self_count<8:self_points[self_count]=contact.pos;self_count+=1
            if self.self_peak is None or 1000.*depth>self.self_peak["depth_mm"]:
                self.self_peak=dict(depth_mm=1000.*depth,step_start_s=max(0.,float(data.time)-model.opt.timestep),
                    body1=model.body(model.geom_bodyid[contact.geom1]).name,
                    body2=model.body(model.geom_bodyid[contact.geom2]).name)
        self.self_points.append(self_points);self.self_counts.append(self_count)
        self.diag.append([ -1. if first is None else 1000.*(data.time-first),self.fraction,
            float(self.loads.sum()),int(self.support.holding),int(info["finger_contact"]),int(info["floor"]),
            info["drift_mm"],normal_speed,info["plans_applied"],self.eta,1000.*self.slip,1000.*self_depth])
        self.load_history.append(self.loads.copy());self.point_history.append(self.points.copy());self.point_counts.append(self.point_count)

    def arrays(self):
        return {"time":np.asarray(self.time),"qpos":np.asarray(self.qpos),"qvel":np.asarray(self.qvel),
            "ctrl":np.asarray(self.ctrl),"object_pos":np.asarray(self.object_pos),"diag":np.asarray(self.diag),
            "loads":np.asarray(self.load_history),"points":np.asarray(self.point_history),"point_counts":np.asarray(self.point_counts),
            "self_points":np.asarray(self.self_points),"self_counts":np.asarray(self.self_counts)}


def make_clip(model,args,name,signature):
    path=args.cache_dir/f"seed{args.seed}_case{args.scenario}_{name}.npz"
    if path.exists() and not args.refresh:
        with np.load(path,allow_pickle=False) as archive:clip={k:archive[k] for k in archive.files}
        meta=json.loads(str(clip.pop("metadata")))
        if meta["signature"]==signature:
            if clip["qpos"].shape[1]!=model.nq:raise ValueError("Cached model size mismatch")
            print(f"Loaded {name}: {len(clip['time'])} physics frames",flush=True)
            return clip,meta
    # 已有种子严格使用历史实验的求解超时决策；缺失时明确采用新的实测时序。
    rows=read_csv(args.results_dir/"solver_times.csv") if (args.results_dir/"solver_times.csv").exists() else []
    schedule_rows=[r for r in rows if int(r["seed"])==args.seed and int(r["scenario"])==args.scenario and r["strategy"]==name]
    schedule={round(float(r["time_s"])/model.opt.timestep):bool(int(r["overrun"])) for r in schedule_rows} if schedule_rows else None
    data=initial(model);palm=data.body("hand/palm_lower")
    center=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.055,-.08])
    selection=next((x for x in scenarios(center,model.opt.gravity,seed=args.seed) if x[0]==args.scenario),None)
    if selection is None:raise ValueError("Scenario must be an existing index 0-19")
    case,kind,p,v=selection
    hand_factory=None;buffered=False;duration=.14;tracker_factory=None;planning_limit=4.;reference_factory=None
    if args.experiment=="hand":
        _,delay_steps,feedback,limit=next(x for x in HAND_FORECAST_VARIANTS if x[0]==name)
        hand_rows=read_csv(args.results_dir/"hand_planner.csv") if (args.results_dir/"hand_planner.csv").exists() else []
        hand_schedule={int(r["plan_step"]):bool(int(r["plan_overrun"])) for r in hand_rows
            if int(r["seed"])==args.seed and int(r["scenario"])==case and r["strategy"]==name}
        hand_factory=lambda m,h:SphereCageClosure(m,h,plan_schedule=hand_schedule if schedule is not None else None)
        tracker_factory=lambda m:CoupledHandForecast(m,delay_steps,feedback,limit)
        planning_limit=3.
    elif args.experiment=="delay":
        _,measured,delay_steps,predictive=next(x for x in DELAY_VARIANTS if x[0]==name)
        hand_factory=lambda m,h:SphereCageClosure(m,h)
        tracker_factory=lambda m:ScheduledArmTracker(m,delay_steps,predictive)
        planning_limit=3.
        if measured:reference_factory=MeasuredContactReference
    elif args.experiment=="arm":
        _,filtered,buffered,planning_limit=next(x for x in ARM_VARIANTS if x[0]==name)
        hand_factory=lambda m,h:SphereCageClosure(m,h)
        if filtered:tracker_factory=DiscreteArmFilter
    elif args.experiment=="cage":
        mode=next(x[1] for x in CAGE_VARIANTS if x[0]==name)
        config=CageConfig()
        if schedule is not None:
            prior=next(r for r in read_csv(args.results_dir/"trials.csv") if int(r["seed"])==args.seed and int(r["scenario"])==case and r["strategy"]==name)
            config=CageConfig(**{key:float(prior[key]) for key in CageConfig.__dataclass_fields__})
        hand_factory=(lambda m,h:JointAwareClosure(m,h,"compliant")) if mode is None else (lambda m,h:SphereCageClosure(m,h,mode,config))
    elif args.experiment=="holding":
        mode=next(x[1] for x in HOLD_VARIANTS if x[0]==name)
        config=HoldConfig()
        if schedule is not None:
            prior=next(r for r in read_csv(args.results_dir/"trials.csv") if int(r["seed"])==args.seed and int(r["scenario"])==case and r["strategy"]==name)
            config=HoldConfig(**{key:float(prior[key]) for key in HoldConfig.__dataclass_fields__})
        hand_factory=(lambda m,h:JointAwareClosure(m,h,"compliant")) if mode is None else (lambda m,h:FrictionHoldClosure(m,h,mode,config))
    elif args.experiment=="finger":
        mode=next(x[1] for x in FINGER_VARIANTS if x[0]==name)
        config=ClosureConfig()
        if schedule is not None:
            prior=next(r for r in read_csv(args.results_dir/"trials.csv") if int(r["seed"])==args.seed and int(r["scenario"])==case and r["strategy"]==name)
            config=ClosureConfig(**{key:float(prior[key]) for key in ClosureConfig.__dataclass_fields__})
        if mode is not None:hand_factory=lambda m,h:JointAwareClosure(m,h,mode,config)
    else:
        _,buffered,duration=next(x for x in VARIANTS if x[0]==name)
    record=StateRecorder(model,duration)
    print(f"Simulating {name}, seed {args.seed}, case {case}; timing={'recorded' if schedule else 'new measured'} ...",flush=True)
    result,_,_=run_case(model,p,v,case,kind,args.seed,"mpc_match",step_observer=record.contact_step,
        solver_schedule=schedule,buffer_factory=reference_factory if reference_factory is not None else ContactAdmittance if buffered else None,
        closure_duration=duration,frame_observer=record.frame,hand_controller_factory=hand_factory,
        tracker_factory=tracker_factory,mpc_acceleration_limit=planning_limit)
    verified=False
    if schedule is not None:
        originals=read_csv(args.results_dir/"trials.csv")
        original=next(r for r in originals if int(r["seed"])==args.seed and int(r["scenario"])==case and r["strategy"]==name)
        for key in ("held","qualified","first_contact_s","close_s","drift_mm","peak_normal_first100ms_N","peak_accel"):
            if result[key] is None and not original[key]:continue
            if not np.isclose(float(result[key]),float(original[key]),atol=1e-5,rtol=1e-8):
                raise ValueError(f"Historical replay mismatch {name}/{key}: {result[key]} vs {original[key]}")
        verified=True
    clip=record.arrays()
    if not np.allclose(np.diff(clip["time"]),model.opt.timestep,atol=1e-10):raise ValueError("Nonuniform physics frames")
    meta=dict(signature=signature,strategy=name,seed=args.seed,scenario=case,historical_verified=verified,
              timing="recorded" if schedule is not None else "measured",outcome=result,self_contact_peak=record.self_peak)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**clip,metadata=json.dumps(meta))
    print(f"Saved {name}: held={result['held']} qualified={result['qualified']} drift={result['drift_mm']:.1f} mm; historical match={verified}",flush=True)
    return clip,meta


def apply_frame(model,data,clip,index):
    # 回放只恢复已记录状态；不再mj_step，不让绘图或暂停改变试验结果。
    data.time=float(clip["time"][index]);data.qpos[:]=clip["qpos"][index]
    data.qvel[:]=clip["qvel"][index];data.ctrl[:]=clip["ctrl"][index]
    mujoco.mj_forward(model,data)


def add_markers(scene,clip,index):
    # 橙色轨迹为实际记录的球心，绿色点为上一物理步承载接触位置。
    points=[(clip["object_pos"][j],.0018,[1.,.48,.08,.6]) for j in range(max(0,index-150),index,6)]
    points += [(point,.0028,[.15,1.,.35,1.]) for point in clip["points"][index,:int(clip["point_counts"][index])]]
    points += [(point,.0035,[1.,.1,.1,1.]) for point in clip["self_points"][index,:int(clip["self_counts"][index])]]
    for pos,radius,color in points:
        if scene.ngeom>=scene.maxgeom:break
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius,0.,0.]),pos,np.eye(3).ravel(),np.array(color,dtype=np.float32))
        scene.ngeom+=1


def camera_pose(camera,focus,overview=False):
    camera.lookat[:]=[.3,0,.35] if overview else focus
    camera.distance=1.7 if overview else .52
    camera.azimuth=115;camera.elevation=-25


def configure_display(model):
    # 仅改变材质/灯光以看清黑色手指；不修改碰撞、质量、摩擦或控制参数。
    palette=np.array([[.55,.60,.66,1.],[.35,.65,.95,1.],[.4,.8,.65,1.],
                      [.75,.55,.88,1.],[.9,.73,.4,1.]])
    groups=ContactClosure(model,LeapHand(model)).group
    for geom,part in groups.items():
        model.geom_matid[geom]=-1;model.geom_rgba[geom]=palette[part+1]
    model.vis.headlight.active=1
    model.vis.headlight.ambient[:]=[.5,.5,.5]


def snapshot(model,args,clip,meta,focus):
    from PIL import Image
    data=mujoco.MjData(model)
    first=meta["outcome"]["first_contact_s"]
    at=(first if first is not None else .5)+.12
    index=min(len(clip["time"])-1,int(np.searchsorted(clip["time"],at)))
    apply_frame(model,data,clip,index)
    camera=mujoco.MjvCamera();camera_pose(camera,focus)
    # 只调整离屏画布大小，不改变物理模型参数。
    model.vis.global_.offwidth=1000;model.vis.global_.offheight=720
    with mujoco.Renderer(model,height=720,width=1000) as renderer:
        renderer.update_scene(data,camera=camera);add_markers(renderer.scene,clip,index)
        pixels=renderer.render().copy()
    args.snapshot.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(pixels).save(args.snapshot)
    print(f"Snapshot: {args.snapshot}",flush=True)


def play(model,args,clips,metas,focus):
    import mujoco.viewer
    events=queue.SimpleQueue();data=mujoco.MjData(model)
    selected=0;cursor=0.;paused=args.paused;speed=args.speed;overview=False
    previous=time.perf_counter();opened=previous
    synced=0
    with mujoco.viewer.launch_passive(model,data,key_callback=events.put,show_left_ui=False,show_right_ui=False) as window:
        camera_pose(window.cam,focus)
        while window.is_running():
            before=time.perf_counter();elapsed=before-previous;previous=before
            while not events.empty():
                key=events.get()
                if key==32:paused=not paused
                elif key in (ord('N'),ord('P')):selected=(selected+(1 if key==ord('N') else -1))%len(clips)
                elif ord('1')<=key<=ord('4') and key-ord('1')<len(clips):selected=key-ord('1')
                elif key==ord('R'):cursor=0.
                elif key==ord('C'):cursor=max(0.,(metas[selected]["outcome"]["first_contact_s"] or .5)-.04)
                elif key in (ord('['),ord(']')):paused=True;cursor+=model.opt.timestep*(1 if key==ord(']') else -1)
                elif key==ord('-'):speed=max(.05,speed/2.)
                elif key==ord('='):speed=min(2.,speed*2.)
                elif key==ord('T'):overview=not overview;camera_pose(window.cam,focus,overview)
            clip,meta=clips[selected],metas[selected]
            if not paused:cursor+=elapsed*speed
            end=float(clip["time"][-1])
            if cursor>end:
                if args.loop:cursor=0.
                else:cursor=end;paused=True
            cursor=max(0.,cursor)
            index=min(len(clip["time"])-1,max(0,int(np.searchsorted(clip["time"],cursor,side="right"))-1))
            diag=clip["diag"][index];outcome=meta["outcome"]
            contact_ms=diag[0]
            phase="Flight" if contact_ms<0 else "Impact" if contact_ms<60 else "Capture / closure" if contact_ms<300 else "Hold evaluation"
            step_accel=0.
            if index:
                dofs=[model.joint(f"joint{i}").dofadr[0] for i in range(1,8)]
                step_accel=float(max(abs((clip["qvel"][index,dofs]-clip["qvel"][index-1,dofs])/
                                        (clip["time"][index]-clip["time"][index-1]))))
            closure_label="Other joints closure" if args.experiment in ("finger","holding","cage","delay","hand") else "Closure command"
            labels=f"Strategy\nSeed / case\nPlayback\nTime / contact age\nPhase\n{closure_label}\nOpposed support*\nSum normal load*\nFriction utilization*\nContact slip*\nThumb self-depth*\nOutward ball speed\nActual joint accel\nHold drift\nTrial result\nConstraint-qualified"
            values=(f'{meta["strategy"]}\n{args.seed} / {args.scenario}\n'
                f'{"PAUSED" if paused else "Playing"} {speed:.2f}x\n{clip["time"][index]:.3f} s / '
                f'{"--" if contact_ms<0 else f"{contact_ms:.0f} ms"}\n{phase}\n{100.*diag[1]:.0f}%\n'
                f'{"YES" if diag[3] else "No"}\n{diag[2]:.1f} N\n{diag[9]:.3f}\n{diag[10]:.2f} mm/s\n{diag[11]:.2f} mm\n{diag[7]:+.3f} m/s\n{step_accel:.2f} rad/s2\n'
                f'{"--" if contact_ms<300 else f"{diag[6]:.1f} mm"}\n'
                f'{"HELD 2 s" if outcome["held"] else "NOT HELD"}\n{"YES" if outcome["qualified"] else "No"}')
            instructions=f"Space: pause   [ / ]: one physics step\nN / P or 1-{len(clips)}: strategy   R: restart\nC: jump near contact   T: camera\n- / =: playback speed\nOrange: ball trail; Green: ball contacts*\nRed: thumb-hand self penetration >1 mm*\n*Contact measurements: previous 2 ms physics step\nRecorded physics replay; mouse drag does not change outcome"
            if args.experiment=="arm":instructions+="\nModel-based experiment: control computation delays uncharged"
            if args.experiment=="delay":
                latency=next(r[2] for r in DELAY_VARIANTS if r[0]==meta["strategy"])*model.opt.timestep*1000.
                instructions+=f"\nInjected arm command delay: {latency:.0f} ms; fingers update each 2 ms\nRuntime overruns need additional validation"
            if args.experiment=="hand":
                _,latency,feedback,limit=next(r for r in HAND_FORECAST_VARIANTS if r[0]==meta["strategy"])
                instructions+=f"\nArm delay: {latency*2} ms; hand forecast: {'feedback' if feedback else 'frozen'}\nFilter target: {limit:.1f} rad/s2; actual criterion: 4.0"
            load_text="\n".join(f"{label}: {value:.2f} N" for label,value in zip(LABELS,clip["loads"][index]))
            if args.experiment in ("finger","holding","cage","arm","delay","hand"):
                load_text+="\n\nMiddle joints: angle / speed"
                for label,joint in zip(("F1","F2","F3","Thumb"),(2,6,10,14)):
                    j=model.joint(f"hand/{joint}")
                    angle=np.rad2deg(clip["qpos"][index,j.qposadr[0]])
                    speed_joint=clip["qvel"][index,j.dofadr[0]]
                    load_text+=f"\n{label}: {angle:.1f} deg / {speed_joint:+.2f} rad/s"
            with window.lock():
                apply_frame(model,data,clip,index)
                window.user_scn.ngeom=0;add_markers(window.user_scn,clip,index)
            window.set_texts([(None,mujoco.mjtGridPos.mjGRID_TOPLEFT,labels,values),
                              (None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,instructions,""),
                              (None,mujoco.mjtGridPos.mjGRID_TOPRIGHT,load_text,"")])
            window.sync()
            synced+=1
            if args.smoke_seconds is not None and before-opened>=args.smoke_seconds:break
            time.sleep(max(0.,1./60.-(time.perf_counter()-before)))
    print(f"Viewer closed normally after {synced} frame syncs.",flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=805)
    parser.add_argument("--scenario",type=int,default=0)
    parser.add_argument("--experiment",choices=["buffer","finger","holding","cage","arm","delay","hand"],default="buffer")
    parser.add_argument("--strategy",choices=sorted(set(["compare"]+[x[0] for x in VARIANTS]+[x[0] for x in FINGER_VARIANTS]+[x[0] for x in HOLD_VARIANTS]+[x[0] for x in CAGE_VARIANTS]+[x[0] for x in ARM_VARIANTS]+[x[0] for x in DELAY_VARIANTS]+[x[0] for x in HAND_FORECAST_VARIANTS])),default="compare")
    parser.add_argument("--speed",type=float,default=.25)
    parser.add_argument("--paused",action="store_true")
    parser.add_argument("--loop",action="store_true")
    parser.add_argument("--record-only",action="store_true",help="Record/verify physics without opening the viewer")
    parser.add_argument("--refresh",action="store_true",help="Rebuild recorded states after changing inputs/assets")
    parser.add_argument("--results-dir",type=Path)
    parser.add_argument("--cache-dir",type=Path)
    parser.add_argument("--snapshot",type=Path,help="Save an offscreen contact-stage PNG instead of opening a window")
    parser.add_argument("--smoke-seconds",type=float,help=argparse.SUPPRESS)
    args=parser.parse_args()
    result_folders=dict(buffer="contact_buffer_validation",finger="finger_closure_validation",holding="holding_control_validation",cage="sphere_cage_clearance_validation",arm="arm_filter_validation",delay="contact_delay_final")
    replay_folders=dict(buffer="interactive_replays",finger="finger_replays",holding="holding_replays",cage="cage_clearance_replays",arm="arm_filter_replays",delay="contact_delay_replays")
    result_folders["hand"]="hand_forecast_validation";replay_folders["hand"]="hand_forecast_replays"
    if args.results_dir is None:args.results_dir=ROOT/"results"/result_folders[args.experiment]
    if args.cache_dir is None:args.cache_dir=ROOT/"results"/replay_folders[args.experiment]
    if not np.isfinite(args.speed) or args.speed<=0:parser.error("Speed must be positive")
    if not 0<=args.scenario<20:parser.error("Scenario must be between 0 and 19")
    variants=HAND_FORECAST_VARIANTS if args.experiment=="hand" else DELAY_VARIANTS if args.experiment=="delay" else ARM_VARIANTS if args.experiment=="arm" else CAGE_VARIANTS if args.experiment=="cage" else HOLD_VARIANTS if args.experiment=="holding" else FINGER_VARIANTS if args.experiment=="finger" else VARIANTS
    allowed_seeds=(811,812) if args.experiment=="hand" else (810,811) if args.experiment=="delay" else (809,810) if args.experiment=="arm" else (807,809) if args.experiment=="cage" else (806,807) if args.experiment=="holding" else (805,806) if args.experiment=="finger" else (803,805)
    if args.strategy=="compare" and args.seed not in allowed_seeds:
        parser.error(f"Recorded paired comparison supports seeds {allowed_seeds}. For a new launch choose one --strategy explicitly.")
    if args.strategy!="compare" and args.strategy not in [x[0] for x in variants]:parser.error("Strategy does not belong to this experiment")
    model=build_scene("ball");rest=initial(model)
    palm=rest.body("hand/palm_lower");focus=palm.xpos+palm.xmat.reshape(3,3)@np.array([0.,-.04,-.04])
    names=list(HAND_REPLAY_NAMES) if args.strategy=="compare" and args.experiment=="hand" else list(DELAY_REPLAY_NAMES) if args.strategy=="compare" and args.experiment=="delay" else [x[0] for x in variants] if args.strategy=="compare" else [args.strategy]
    clips=[];metas=[];signature=source_signature(args.results_dir)
    for name in names:
        clip,meta=make_clip(model,args,name,signature);clips.append(clip);metas.append(meta)
    if args.strategy=="compare":
        # 轻度预成形改变接触前手形，只在拥有相同预成形的组之间核对首触。
        comparisons=[] if args.experiment in ("delay","hand") else [(1,2)] if args.experiment in ("cage","arm") else [(0,1),(2,3)] if args.experiment=="finger" else [(0,i) for i in range(1,len(metas))]
        for left,right in comparisons:
            for key in ("first_contact_s","first_normal_relative_speed_m_s","plans_applied"):
                if not np.isclose(metas[right]["outcome"][key],metas[left]["outcome"][key],atol=1e-9,rtol=1e-9):
                    raise ValueError(f"Paired pre-contact mismatch: {key}")
        print("Historical delayed trials verified; policies can change first contact." if args.experiment in ("delay","hand") else "Paired pre-contact checks passed (arm filter, planning headroom and preshape can intentionally change pre-contact motion).",flush=True)
    configure_display(model)
    if args.snapshot:snapshot(model,args,clips[0],metas[0],focus)
    elif not args.record_only:play(model,args,clips,metas,focus)


if __name__=="__main__":main()
