import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# -------------------------------------------------------------
# Resolve path to MJCF file
# -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "tintin_thrust.xml")

print("Loading:", MJCF_PATH)

model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data  = mujoco.MjData(model)

# Body + joint + actuator IDs
rocket_bid     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
yaw_joint_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thruster_yaw")
pitch_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thruster_pitch")

yaw_act_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
pitch_act_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor")
thrust_act_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")


# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def get_pos():
    return data.xpos[rocket_bid].copy()

def get_vel():
    return data.cvel[rocket_bid][:3].copy()

def get_pointing_dir():
    R = data.xmat[rocket_bid].reshape(3, 3)
    return R[:, 2].copy()   # +Z axis of rocket in world frame

def get_gimbal_angles():
    yaw_qpos   = data.qpos[model.jnt_qposadr[yaw_joint_id]]
    pitch_qpos = data.qpos[model.jnt_qposadr[pitch_joint_id]]
    return np.degrees(yaw_qpos), np.degrees(pitch_qpos)

def get_thrust():
    return data.ctrl[thrust_act_id]


# -------------------------------------------------------------
# Initial controls
# -------------------------------------------------------------
data.ctrl[yaw_act_id]   = 0.0
data.ctrl[pitch_act_id] = 0.0
data.ctrl[thrust_act_id] = 120.0

print("\nStarting simulation...\n")

# -------------------------------------------------------------
# Viewer + logging
# -------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():
        mujoco.mj_step(model, data)

        # Logging
        pos = get_pos()
        vel = get_vel()
        pointing = get_pointing_dir()
        yaw_deg, pitch_deg = get_gimbal_angles()
        thrust = get_thrust()

        print(f"Step {step:04d}")
        print(f"  Position       : {pos}")
        print(f"  Velocity       : {vel}")
        print(f"  Pointing Dir   : {pointing}")
        print(f"  Gimbal Yaw     : {yaw_deg:.2f}°")
        print(f"  Gimbal Pitch   : {pitch_deg:.2f}°")
        print(f"  Thrust (N)     : {thrust:.1f}\n")

        step += 1

        viewer.sync()
        time.sleep(0.01)
