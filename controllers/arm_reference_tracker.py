"""通过模型现有的位置执行器跟踪七轴参考，补偿量同样受执行器力矩限制。"""
import mujoco
import numpy as np


class ArmReferenceTracker:
    def __init__(self,model,mode="inverse"):
        if mode not in ("pd_gravity","inverse"):
            raise ValueError("Unknown reference tracker")
        self.model=model
        self.mode=mode
        self.ids=[model.actuator(f"act{i}").id for i in range(1,8)]
        joints=[model.joint(f"joint{i}") for i in range(1,8)]
        self.qadr=[j.qposadr[0] for j in joints]
        self.dofs=[j.dofadr[0] for j in joints]
        self.kp=model.actuator_gainprm[self.ids,0]
        self.kd=-model.actuator_biasprm[self.ids,2]
        if np.any(self.kp<=0) or not np.allclose(model.actuator_gear[self.ids,0],1.):
            raise ValueError("Expected direct unit-gear position actuators")
        self.inverse=mujoco.MjData(model)
        self.clipped_steps=0

    def command(self,data,plan,t):
        q,v,a=plan.reference(t)
        if self.mode=="inverse":
            # 用参考状态求逆动力学；速度前馈抵消位置执行器自身的速度阻尼。
            self.inverse.qpos[:]=data.qpos
            self.inverse.qvel[:]=data.qvel
            self.inverse.qacc[:]=0.
            self.inverse.qpos[self.qadr]=q
            self.inverse.qvel[self.dofs]=v
            self.inverse.qacc[self.dofs]=a
            mujoco.mj_inverse(self.model,self.inverse)
            torque=self.inverse.qfrc_inverse[self.dofs]
            target=q+(torque+self.kd*v)/self.kp
        else:
            target=q+data.qfrc_bias[self.dofs]/self.kp
        limits=self.model.actuator_ctrlrange[self.ids]
        clipped=np.clip(target,limits[:,0],limits[:,1])
        self.clipped_steps+=int(np.any(clipped!=target))
        # 重力等补偿全部经执行器施加，便于统计真实的执行器力矩饱和。
        data.qfrc_applied[self.dofs]=0.
        data.ctrl[self.ids]=clipped
