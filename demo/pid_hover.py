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
# PID controller parameters
# -------------------------------------------------------------
TARGET_ALT = 10.0        # hover at 10 meters
Kp = 250.0               # proportional gain
Ki = 20.0                # integral gain
Kd = 120.0               # derivative gain

integral_err = 0.0
last_err = 0.0

# Approximate mass of rocket (from your XML: 100 + 20 + 30 = 150 kg)
MASS = 150.0
GRAVITY = 9.81

# MuJoCo timestep
DT = model.opt.timestep


# -------------------------------------------------------------
# Viewer + PID hover control
# -------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():
        # -----------------------------------------------------
        # Read rocket state
        # -----------------------------------------------------
        pos = get_pos()
        vel = get_vel()
        z = pos[2]
        vz = vel[2]

        # -----------------------------------------------------
        # PID altitude control
        # -----------------------------------------------------
        err = TARGET_ALT - z              # altitude error
        derr = (err - last_err) / DT      # derivative
        integral_err += err * DT          # accumulate integral

        # anti-windup
        integral_err = np.clip(integral_err, -50, 50)

        # PID output (force)
        pid_output = Kp*err + Ki*integral_err + Kd*derr

        # gravity compensation
        hover_force = MASS * GRAVITY

        # final thrust command
        thrust_cmd = hover_force + pid_output

        # clamp to actuator limits
        thrust_cmd = np.clip(thrust_cmd, 0, 2000)

        # -----------------------------------------------------
        # Apply actuator commands
        # -----------------------------------------------------
        data.ctrl[thrust_act_id] = thrust_cmd

        # Keep gimbals centered for now
        data.ctrl[yaw_act_id] = 0.0
        data.ctrl[pitch_act_id] = 0.0

        # -----------------------------------------------------
        # Step simulation
        # -----------------------------------------------------
        mujoco.mj_step(model, data)

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        print(f"Step {step:04d}")
        print(f"  Altitude       : {z:.3f} m")
        print(f"  Vertical vel   : {vz:.3f} m/s")
        print(f"  Error          : {err:.3f}")
        print(f"  Thrust command : {thrust_cmd:.1f} N\n")

        last_err = err
        step += 1

        viewer.sync()
        time.sleep(0.01)
