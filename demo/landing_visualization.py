import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# -------------------------------------------------------------
# Resolve path to MJCF file
# -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "tintin_thrust_added_flame.xml")

print("Loading:", MJCF_PATH)

model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data  = mujoco.MjData(model)

# -------------------------------------------------------------
# IDs
# -------------------------------------------------------------
rocket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")

yaw_act_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
pitch_act_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor")
thrust_act_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")

# plume geom id
plume_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plume_geom")

# free joint addresses
qpos_adr = model.jnt_qposadr[free_joint_id]
qvel_adr = model.jnt_dofadr[free_joint_id]

# -------------------------------------------------------------
# INITIAL STATE
# -------------------------------------------------------------
data.qpos[qpos_adr + 0] = 0.0
data.qpos[qpos_adr + 1] = 0.0
data.qpos[qpos_adr + 2] = 20.0

data.qpos[qpos_adr + 3] = 1.0
data.qpos[qpos_adr + 4] = 0.0
data.qpos[qpos_adr + 5] = 0.0
data.qpos[qpos_adr + 6] = 0.0

data.qvel[qvel_adr + 0] = 0.0
data.qvel[qvel_adr + 1] = 0.0
data.qvel[qvel_adr + 2] = -0.5  # falling

data.qvel[qvel_adr + 3] = 0.0
data.qvel[qvel_adr + 4] = 0.0
data.qvel[qvel_adr + 5] = 0.0

mujoco.mj_forward(model, data)

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def get_pos():
    return data.xpos[rocket_bid].copy()

def get_vel():
    v_body = data.cvel[rocket_bid][3:]
    R = data.xmat[rocket_bid].reshape(3, 3)
    return (R @ v_body).copy()

# -------------------------------------------------------------
# Mass & Fuel
# -------------------------------------------------------------

def set_plume_length(length):
    # minimum safe half-length
    MIN_HALF = 0.005
    half_len = max(length * 0.5, MIN_HALF)

    # Set geom half-length
    model.geom_size[plume_geom_id][1] = half_len

    # Position center halfway down the plume
    model.geom_pos[plume_geom_id][2] = -0.25 - half_len

    # Pointing down (-Z): quaternion (180° rotation about X axis)
    model.geom_quat[plume_geom_id] = np.array([0, 1, 0, 0])


DRY_MASS = model.body_mass[rocket_bid]
INITIAL_FUEL_MASS = 10.0
fuel_mass = INITIAL_FUEL_MASS
total_mass = DRY_MASS + fuel_mass

orig_inertia = model.body_inertia[rocket_bid].copy()
initial_total_mass = total_mass

ISP = 250.0
G0 = 9.81
DT = model.opt.timestep

# -------------------------------------------------------------
# LANDING PD GAINS
# -------------------------------------------------------------
Kp_z = 1.0
Kd_v = 3.0
Ki_z = 0.0
integral_z = 0.0

TARGET_Z  = 0.5
TARGET_VZ = 0.0

# -------------------------------------------------------------
# SIMULATION LOOP
# -------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():

        pos = get_pos()
        vel = get_vel()

        z  = pos[2]
        vz = vel[2]

        err_z = TARGET_Z - z
        err_v = TARGET_VZ - vz

        # altitude integral (if Ki_z > 0)
        integral_z += err_z * DT
        integral_z = np.clip(integral_z, -10, 10)

        # desired vertical acceleration
        a_des = Kp_z * err_z + Kd_v * err_v + Ki_z * integral_z

        # thrust required
        thrust_cmd = total_mass * (G0 + a_des)
        thrust_cmd = np.clip(thrust_cmd, 0.0, 2200.0)

        # -----------------------------------------------------
        # Update plume rendering
        # -----------------------------------------------------
        norm_thrust = np.clip(thrust_cmd / 2200.0, 0.0, 1.0)

        # max plume = 0.5m
        plume_len = 0.5 * norm_thrust

        # add minimal non-zero length to avoid MuJoCo warnings
        MIN_LEN = 0.01

        if plume_len < MIN_LEN:
            plume_len = MIN_LEN


        start = np.array([0.0, 0.0, -0.25])
        end   = np.array([0.0, 0.0, -0.25 - plume_len])

        # normalized thrust → plume length
        norm = np.clip(thrust_cmd / 2200.0, 0.0, 1.0)
        plume_length = 0.5 * norm   # max 0.5m

        set_plume_length(plume_length)


        # -----------------------------------------------------
        # Fuel consumption
        # -----------------------------------------------------
        if fuel_mass > 0:
            mass_flow_rate = -thrust_cmd / (ISP * G0)
            fuel_mass += mass_flow_rate * DT
            fuel_mass = max(fuel_mass, 0.0)

        if fuel_mass <= 0:
            thrust_cmd = 0.0
            integral_z = 0.0

        total_mass = DRY_MASS + fuel_mass

        model.body_mass[rocket_bid] = total_mass
        model.body_inertia[rocket_bid] = orig_inertia * (total_mass / initial_total_mass)

        # apply controls
        data.ctrl[thrust_act_id] = thrust_cmd
        data.ctrl[yaw_act_id] = 0.0
        data.ctrl[pitch_act_id] = 0.0

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        # logging
        print(f"Step {step:04d}")
        print(f"  Altitude (z)      : {z:8.3f} m")
        print(f"  Vertical vel (vz) : {vz:8.3f} m/s")
        print(f"  Fuel              : {fuel_mass:8.3f} kg")
        print(f"  Thrust Command    : {thrust_cmd:8.1f} N\n")

        step += 1
        viewer.sync()
        time.sleep(0.01)
