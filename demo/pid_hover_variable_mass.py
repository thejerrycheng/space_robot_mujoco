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
# PID controller parameters
# -------------------------------------------------------------
TARGET_ALT = 10.0
Kp = 300.0
Ki = 20.0
Kd = 150.0

integral_err = 0.0
last_err = 0.0


# -------------------------------------------------------------
# Viewer + PID + MASS DECAY + FUEL CUTOFF
# -------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    while viewer.is_running():

        # --- Read rocket state ---
        pos = get_pos()
        vel = get_vel()
        z = pos[2]
        vz = vel[2]

        # --- PID error ---
        err = TARGET_ALT - z
        derr = (err - last_err) / DT
        integral_err += err * DT
        integral_err = np.clip(integral_err, -50, 50)

        # --- Gravity compensation ---
        hover_force = total_mass * G0

        # --- PID thrust (before fuel cutoff) ---
        pid_out = Kp*err + Ki*integral_err + Kd*derr
        thrust_cmd = hover_force + pid_out
        thrust_cmd = np.clip(thrust_cmd, 0, 2000)

        # -----------------------------------------------------
        # FUEL CUTOFF LOGIC
        # -----------------------------------------------------
        if fuel_mass <= 0:
            thrust_cmd = 0.0              # NO THRUST WHEN OUT OF FUEL
            integral_err = 0.0            # stop accumulating integral term

        # Apply thrust
        data.ctrl[thrust_act_id] = thrust_cmd
        data.ctrl[yaw_act_id] = 0.0
        data.ctrl[pitch_act_id] = 0.0

        # -----------------------------------------------------
        # Fuel consumption
        # -----------------------------------------------------
        if fuel_mass > 0:
            mass_flow_rate = -thrust_cmd / (ISP * G0)
            fuel_mass += mass_flow_rate * DT
            fuel_mass = max(0.0, fuel_mass)

        # Update total mass
        total_mass = DRY_MASS + fuel_mass

        # -----------------------------------------------------
        # Update MuJoCo mass + inertia
        # -----------------------------------------------------
        mass_ratio = total_mass / initial_total_mass

        model.body_mass[rocket_bid] = total_mass
        model.body_inertia[rocket_bid] = orig_inertia * mass_ratio

        mujoco.mj_forward(model, data)

        # Step physics
        mujoco.mj_step(model, data)

        # --- Logging ---
        print(f"Step {step:04d}")
        print(f"  Altitude     : {z:.3f} m")
        print(f"  Total Mass   : {total_mass:.3f} kg")
        print(f"  Fuel Mass    : {fuel_mass:.3f} kg")
        print(f"  Thrust (N)   : {thrust_cmd:.1f}\n")

        last_err = err
        step += 1

        viewer.sync()
        time.sleep(0.01)
