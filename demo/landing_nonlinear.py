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

# -------------------------------------------------------------
# IDs
# -------------------------------------------------------------
rocket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

yaw_joint_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thruster_yaw")
pitch_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "thruster_pitch")

yaw_act_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
pitch_act_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor")
thrust_act_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")

# -------------------------------------------------------------
# SET INITIAL STATE (tilted and moving)
# -------------------------------------------------------------
free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
qpos_adr = model.jnt_qposadr[free_joint_id]
qvel_adr = model.jnt_dofadr[free_joint_id]

# --- Initial Position ---
data.qpos[qpos_adr + 0] = 0.0       # x
data.qpos[qpos_adr + 1] = 0.0       # y
data.qpos[qpos_adr + 2] = 25.0      # start higher

# --- Initial Orientation (tilted forward 20° nose-down) ---
tilt_angle = np.radians(20)
# rotation around Y axis = nose down
qy = np.sin(tilt_angle/2)
qw = np.cos(tilt_angle/2)
qx = 0
qz = 0
data.qpos[qpos_adr + 3] = qw
data.qpos[qpos_adr + 4] = qx
data.qpos[qpos_adr + 5] = qy
data.qpos[qpos_adr + 6] = qz

# --- Initial Velocity ---
data.qvel[qvel_adr + 0] =  .0    # forward
data.qvel[qvel_adr + 1] =  0.0
data.qvel[qvel_adr + 2] = -0.0    # descending

data.qvel[qvel_adr + 3] = 0.0     # no angular velocity yet
data.qvel[qvel_adr + 4] = 0.0
data.qvel[qvel_adr + 5] = 0.0

mujoco.mj_forward(model, data)


# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def get_pos():
    return data.xpos[rocket_bid].copy()

def get_vel():
    # linear velocity in BODY frame from composite velocity
    v_body = data.cvel[rocket_bid][3:]

    # rotation matrix world-from-body
    R = data.xmat[rocket_bid].reshape(3, 3)

    # convert to WORLD frame
    return (R @ v_body).copy()

def get_gimbal_angles():
    yaw_q = data.qpos[model.jnt_qposadr[yaw_joint_id]]
    pitch_q = data.qpos[model.jnt_qposadr[pitch_joint_id]]
    return np.degrees(yaw_q), np.degrees(pitch_q)

# -------------------------------------------------------------
# Fuel & Mass Parameters
# -------------------------------------------------------------
DRY_MASS = model.body_mass[rocket_bid]

INITIAL_FUEL_MASS = 20.0
fuel_mass = INITIAL_FUEL_MASS
total_mass = DRY_MASS + fuel_mass

orig_inertia = model.body_inertia[rocket_bid].copy()
initial_total_mass = total_mass

ISP = 250.0
G0 = 9.81
DT = model.opt.timestep

# -------------------------------------------------------------
# LANDING CONTROLLER PARAMETERS (PD)
# -------------------------------------------------------------
# -------------------------------------------------------------
# FULL NONLINEAR POSITION → FORCE → TVC CONTROLLER
# -------------------------------------------------------------
Kp = np.diag([1.5, 1.5, 3.0])   # tunable
Kv = np.diag([2.5, 2.5, 4.0])

TARGET_POS = np.array([0.0, 0.0, 0.5])   # landing pad
TARGET_VEL = np.array([0.0, 0.0, 0.0])
g_vec = np.array([0, 0, -G0])            # gravity in world frame

def nonlinear_controller(pos, vel, mass, Rwb):
    # Position and velocity errors
    e_p = TARGET_POS - pos
    e_v = TARGET_VEL - vel

    # Desired acceleration (outer loop)
    a_cmd = Kp @ e_p + Kv @ e_v

    # Add gravity compensation
    F_cmd = mass * (a_cmd - g_vec)

    # ---------------------------------------------------------
    # THRUST MAGNITUDE
    # ---------------------------------------------------------
    T = np.linalg.norm(F_cmd)
    T = np.clip(T, 0.0, 2200.0)

    if T < 1e-6:
        return 0.0, 0.0, 0.0

    # ---------------------------------------------------------
    # THRUST DIRECTION → GIMBAL ANGLES
    # ---------------------------------------------------------
    thrust_dir_world = F_cmd / T
    thrust_dir_body  = Rwb.T @ thrust_dir_world

    tx, ty, tz = thrust_dir_body

    # pitch = atan2(-ty, tz)
    pitch_cmd = np.arctan2(-ty, tz)

    # yaw = atan2(tx, sqrt(ty^2 + tz^2))
    yaw_cmd = np.arctan2(tx, np.sqrt(ty**2 + tz**2))

    # Saturate to motor limits
    MAX_GIMBAL = np.radians(20)
    pitch_cmd = np.clip(pitch_cmd, -MAX_GIMBAL, MAX_GIMBAL)
    yaw_cmd   = np.clip(yaw_cmd,   -MAX_GIMBAL, MAX_GIMBAL)

    return T, yaw_cmd, pitch_cmd


# -------------------------------------------------------------
# SIMULATION LOOP
# -------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():

        # state
        pos = get_pos()
        vel = get_vel()
        z  = pos[2]
        vz = vel[2]
        Rwb = data.xmat[rocket_bid].reshape(3, 3)

        # nonlinear TVC controller
        thrust_cmd, yaw_cmd, pitch_cmd = nonlinear_controller(pos, vel, total_mass, Rwb)

        # fuel usage
        if fuel_mass > 0.0:
            mass_flow_rate = -thrust_cmd / (ISP * G0)
            fuel_mass += mass_flow_rate * DT
            fuel_mass = max(fuel_mass, 0.0)

        total_mass = DRY_MASS + fuel_mass

        # update mass
        mass_ratio = total_mass / initial_total_mass
        model.body_mass[rocket_bid] = total_mass
        model.body_inertia[rocket_bid] = orig_inertia * mass_ratio



        # -----------------------------------------------------
        # Fuel consumption
        # -----------------------------------------------------
        if fuel_mass > 0.0:
            mass_flow_rate = -thrust_cmd / (ISP * G0)
            fuel_mass += mass_flow_rate * DT
            fuel_mass = max(fuel_mass, 0.0)

        if fuel_mass <= 0.0:
            thrust_cmd = 0.0
            integral_z = 0.0

        total_mass = DRY_MASS + fuel_mass

        # update mass & inertia
        mass_ratio = total_mass / initial_total_mass
        model.body_mass[rocket_bid] = total_mass
        model.body_inertia[rocket_bid] = orig_inertia * mass_ratio

        # apply controls
        data.ctrl[thrust_act_id] = thrust_cmd
        data.ctrl[yaw_act_id]    = yaw_cmd
        data.ctrl[pitch_act_id]  = pitch_cmd

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        # -----------------------------------------------------
        # LOGGING
        # -----------------------------------------------------
        print(f"Step {step:04d}")
        print(f"  Altitude (z)      : {z:8.3f} m")
        print(f"  Vertical vel (vz) : {vz:8.3f} m/s")
        print(f"  Velocity vector   : [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}] m/s")
        print(f"  Fuel Mass         : {fuel_mass:8.3f} kg")
        print(f"  Total Mass        : {total_mass:8.3f} kg")
        print(f"  Thrust Command    : {thrust_cmd:8.1f} N\n")

        step += 1
        viewer.sync()
        time.sleep(0.01)
