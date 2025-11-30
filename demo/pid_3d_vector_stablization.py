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
    return R[:, 2].copy()

def get_gimbal_angles():
    yaw_qpos   = data.qpos[model.jnt_qposadr[yaw_joint_id]]
    pitch_qpos = data.qpos[model.jnt_qposadr[pitch_joint_id]]
    return np.degrees(yaw_qpos), np.degrees(pitch_qpos)

def get_thrust():
    return data.ctrl[thrust_act_id]


# -------------------------------------------------------------
# Fuel & Mass Parameters
# -------------------------------------------------------------
DRY_MASS = model.body_mass[rocket_bid]  # read from XML

INITIAL_FUEL_MASS = 20.0                # kg
fuel_mass = INITIAL_FUEL_MASS
total_mass = DRY_MASS + fuel_mass

# Save original inertia to scale later
orig_inertia = model.body_inertia[rocket_bid].copy()
initial_total_mass = total_mass

# Rocket engine performance
ISP = 250.0
G0 = 9.81
DT = model.opt.timestep

# -------------------------------------------------------------
# Target Position for 3-Axis Control
# -------------------------------------------------------------
TARGET_POS = np.array([2.0, 2.0, 10.0])    # (x, y, z)

# Position PID gains
Kp_pos = np.array([3.0, 3.0, 6.0])
Ki_pos = np.array([0.0, 0.0, 1.0])
Kd_pos = np.array([4.0, 4.0, 8.0])

pos_integral = np.zeros(3)
last_pos_err = np.zeros(3)

# Attitude stabilization gains
Kp_att = 2.0      # PD for orientation
Kd_att = 1.5

# Gimbal actuator limits (deg → rad)
MAX_GIMBAL = np.radians(30.0)


# -------------------------------------------------------------
# Thrust Vectoring Controller
# -------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():

        # -----------------------------------------------------
        # State Extraction
        # -----------------------------------------------------
        pos = get_pos()
        vel = get_vel()

        z = pos[2]
        pos_err = TARGET_POS - pos
        vel_err = -vel

        # Integrate error for vertical channel only
        pos_integral[2] += pos_err[2] * DT
        pos_integral[2] = np.clip(pos_integral[2], -10, 10)

        # D-term
        d_err = (pos_err - last_pos_err) / DT
        last_pos_err = pos_err.copy()

        # -----------------------------------------------------
        # DESIRED ACCELERATION (outer-loop PID)
        # -----------------------------------------------------
        acc_cmd = (
            Kp_pos * pos_err +
            Ki_pos * pos_integral +
            Kd_pos * d_err
        )

        # Vertical channel adds gravity
        acc_cmd[2] += G0

        # -----------------------------------------------------
        # DESIRED ORIENTATION
        # -----------------------------------------------------
        # The rocket thrust vector points along body +Z.
        # So desired thrust direction = normalize(acc_cmd)
        if np.linalg.norm(acc_cmd) < 1e-6:
            thrust_dir = np.array([0, 0, 1])
        else:
            thrust_dir = acc_cmd / np.linalg.norm(acc_cmd)

        # Current orientation (body frame rotation)
        R = data.xmat[rocket_bid].reshape(3, 3)
        body_z = R[:, 2]  # current thrust direction

        # -----------------------------------------------------
        # Orientation error (axis-angle)
        # -----------------------------------------------------
        axis = np.cross(body_z, thrust_dir)
        axis_norm = np.linalg.norm(axis)
        angle_err = np.arccos(np.clip(np.dot(body_z, thrust_dir), -1, 1))

        if axis_norm < 1e-6:
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = axis / axis_norm

        # PD attitude correction → angular velocity command
        ang_vel_cmd = Kp_att * angle_err * axis - Kd_att * data.qvel[3:6]

        # -----------------------------------------------------
        # Convert angular velocity command → gimbal commands
        # -----------------------------------------------------
        # For small angles, gimbal controls approximate:
        # yaw_gimbal ~ ang_vel_cmd[1]
        # pitch_gimbal ~ -ang_vel_cmd[0]
        yaw_cmd = np.clip(ang_vel_cmd[1], -MAX_GIMBAL, MAX_GIMBAL)
        pitch_cmd = np.clip(-ang_vel_cmd[0], -MAX_GIMBAL, MAX_GIMBAL)

        # -----------------------------------------------------
        # THRUST COMPUTATION (inner-loop vertical control)
        # -----------------------------------------------------
        required_force = total_mass * acc_cmd[2]
        thrust_cmd = np.clip(required_force, 0, 2000)

        # FUEL CUTOFF
        if fuel_mass <= 0:
            thrust_cmd = 0.0

        # -----------------------------------------------------
        # Apply Controls
        # -----------------------------------------------------
        data.ctrl[thrust_act_id] = thrust_cmd
        data.ctrl[yaw_act_id] = yaw_cmd
        data.ctrl[pitch_act_id] = pitch_cmd

        # -----------------------------------------------------
        # Fuel Burn
        # -----------------------------------------------------
        if fuel_mass > 0:
            mass_flow_rate = -thrust_cmd / (ISP * G0)
            fuel_mass += mass_flow_rate * DT
            fuel_mass = max(0, fuel_mass)

        total_mass = DRY_MASS + fuel_mass

        # -----------------------------------------------------
        # Update MuJoCo Mass & Inertia
        # -----------------------------------------------------
        mass_ratio = total_mass / initial_total_mass
        model.body_mass[rocket_bid] = total_mass
        model.body_inertia[rocket_bid] = orig_inertia * mass_ratio

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        print(f"Step {step:04d}")
        print(f"  Pos           : {pos}")
        print(f"  Target        : {TARGET_POS}")
        print(f"  Gimbal Yaw    : {np.degrees(yaw_cmd):.2f}°")
        print(f"  Gimbal Pitch  : {np.degrees(pitch_cmd):.2f}°")
        print(f"  Thrust (N)    : {thrust_cmd:.1f}")
        print(f"  Mass (kg)     : {total_mass:.2f}\n")

        step += 1
        viewer.sync()
        time.sleep(0.01)
