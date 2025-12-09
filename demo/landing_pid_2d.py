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
# SET INITIAL STATE (position = 20m, velocity = -0.5m/s)
# -------------------------------------------------------------
free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
qpos_adr = model.jnt_qposadr[free_joint_id]
qvel_adr = model.jnt_dofadr[free_joint_id]

# initial position
data.qpos[qpos_adr + 0] = 0.0
data.qpos[qpos_adr + 1] = 0.0
data.qpos[qpos_adr + 2] = 20.0   # altitude

# upright quaternion
data.qpos[qpos_adr + 3] = 1.0
data.qpos[qpos_adr + 4] = 0.0
data.qpos[qpos_adr + 5] = 0.0
data.qpos[qpos_adr + 6] = 0.0

# initial velocity
data.qvel[qvel_adr + 0] = 0.0
data.qvel[qvel_adr + 1] = 0.0
data.qvel[qvel_adr + 2] = -0.5     # falling slowly

# no rotation
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

INITIAL_FUEL_MASS = 10.0
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
Kp_z = 2.0
Kd_v = 7.0
Ki_z = 0.0
integral_z = 0.0

TARGET_Z  = 0.5    # rocket stands around z=0.5 at rest
TARGET_VZ = 0.0

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

        # errors
        err_z = TARGET_Z - z
        err_v = TARGET_VZ - vz

        # altitude integral (off unless Ki_z > 0)
        integral_z += err_z * DT
        integral_z = np.clip(integral_z, -10, 10)

        # -----------------------------------------------------
        # LANDING BURN CONTROLLER
        # -----------------------------------------------------
        # desired vertical acceleration
        a_des = (
            Kp_z * err_z +
            Kd_v * err_v +
            Ki_z * integral_z
        )

        # thrust = mass * (gravity + desired_accel)
        thrust_cmd = total_mass * (G0 + a_des)
        thrust_cmd = np.clip(thrust_cmd, 0.0, 2200.0)

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

        # apply to sim
        data.ctrl[thrust_act_id] = thrust_cmd
        data.ctrl[yaw_act_id] = 0.0
        data.ctrl[pitch_act_id] = 0.0

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
