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
    return data.xpos[rocket_bid].copy()  # world COM position

def get_body_rot():
    return data.xmat[rocket_bid].reshape(3, 3)  # body->world

def get_gimbal_state():
    yaw_qpos   = data.qpos[model.jnt_qposadr[yaw_joint_id]]
    pitch_qpos = data.qpos[model.jnt_qposadr[pitch_joint_id]]
    yaw_qvel   = data.qvel[model.jnt_dofadr[yaw_joint_id]]
    pitch_qvel = data.qvel[model.jnt_dofadr[pitch_joint_id]]
    return yaw_qpos, pitch_qpos, yaw_qvel, pitch_qvel


# -------------------------------------------------------------
# Fuel & Mass Parameters
# -------------------------------------------------------------
DRY_MASS = model.body_mass[rocket_bid]  # from XML "ball" inertial

INITIAL_FUEL_MASS = 50.0
fuel_mass = INITIAL_FUEL_MASS
total_mass = DRY_MASS + fuel_mass

orig_inertia = model.body_inertia[rocket_bid].copy()
initial_total_mass = total_mass

ISP = 250.0
G0 = 9.81
DT = model.opt.timestep
F_MAX = 2000.0
g_vec = np.array([0.0, 0.0, -G0])


# -------------------------------------------------------------
# Target and Position Controller
# -------------------------------------------------------------
TARGET_POS = np.array([0.5, 0.5, 10.0])  # goal (x,y,z)

# Lateral gains
Kp_xy = 1
Kd_xy = 0.2

# Vertical gains
Kp_z = 2.0
Kd_z = 1.0
Ki_z = 0.01
z_int = 0.0

# Gimbal PD
Kp_gimbal = 6.0
Kd_gimbal = 1.0

MAX_GIMBAL_ANGLE  = np.radians(30.0)
MAX_GIMBAL_TORQUE = 500.0

# for numerical velocity
last_pos = None


# -------------------------------------------------------------
# Viewer + Nonlinear Thrust-Vector Control
# -------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0

    # make sure initial forward kinematics is computed
    mujoco.mj_forward(model, data)
    last_pos = get_pos()

    while viewer.is_running():
        # -----------------------------------------------------
        # State
        # -----------------------------------------------------
        pos = get_pos()
        vel = (pos - last_pos) / DT
        last_pos = pos.copy()

        R = get_body_rot()  # body->world

        pos_err = TARGET_POS - pos
        vel_err = -vel

        # -----------------------------------------------------
        # Vertical control (z)
        # -----------------------------------------------------
        z_err = pos_err[2]
        z_vel = vel[2]

        z_int += z_err * DT
        z_int = np.clip(z_int, -10.0, 10.0)

        a_z_cmd = Kp_z * z_err + Ki_z * z_int - Kd_z * z_vel

        # Desired vertical force: Fz = m * (a_z_cmd + g)
        Fz = total_mass * (a_z_cmd + G0)
        Fz = max(0.0, Fz)  # no negative thrust

        # -----------------------------------------------------
        # Lateral control (x,y)
        # -----------------------------------------------------
        a_lat_cmd = np.zeros(2)
        a_lat_cmd[0] = Kp_xy * pos_err[0] - Kd_xy * vel[0]
        a_lat_cmd[1] = Kp_xy * pos_err[1] - Kd_xy * vel[1]

        Fxy = total_mass * a_lat_cmd
        Fxy_norm = np.linalg.norm(Fxy)

        # -----------------------------------------------------
        # Respect thrust limit: sqrt(Fz^2 + |Fxy|^2) <= F_MAX
        # -----------------------------------------------------
        if Fz > F_MAX:
            # Too much vertical demand; cap and kill lateral
            Fz = F_MAX
            Fxy[:] = 0.0
        else:
            # Max allowed lateral magnitude given Fz
            Fxy_max = np.sqrt(max(F_MAX**2 - Fz**2, 0.0))
            if Fxy_norm > 1e-6:
                scale = min(1.0, Fxy_max / Fxy_norm)
                Fxy *= scale

        # Final desired force in world frame
        F_des_world = np.array([Fxy[0], Fxy[1], Fz])
        F_mag = np.linalg.norm(F_des_world)

        if fuel_mass <= 0.0 or F_mag < 1e-6:
            thrust_cmd = 0.0
            yaw_des = 0.0
            pitch_des = 0.0
        else:
            # Thrust magnitude (still capped by F_MAX)
            thrust_cmd = min(F_mag, F_MAX)

            # Desired thrust direction (world)
            u_world = F_des_world / thrust_cmd

            # Convert thrust direction to body frame
            u_body = R.T @ u_world
            ux, uy, uz = u_body

            # ---- CORRECT inverse gimbal kinematics ----
            # u_body = Rx(yaw) * Ry(pitch) * e_z  ⇒
            # ux = sin(pitch)
            # uy = -sin(yaw)*cos(pitch)
            # uz =  cos(yaw)*cos(pitch)

            pitch_des = np.arcsin(np.clip(ux, -1.0, 1.0))
            cos_pitch = np.cos(pitch_des)

            if abs(cos_pitch) < 1e-6:
                yaw_des = 0.0
            else:
                yaw_des = np.arctan2(-uy, uz)

            # Limit gimbal angles
            yaw_des   = np.clip(yaw_des,   -MAX_GIMBAL_ANGLE, MAX_GIMBAL_ANGLE)
            pitch_des = np.clip(pitch_des, -MAX_GIMBAL_ANGLE, MAX_GIMBAL_ANGLE)

        # -----------------------------------------------------
        # Gimbal PD control
        # -----------------------------------------------------
        yaw_angle, pitch_angle, yaw_vel, pitch_vel = get_gimbal_state()

        yaw_err   = yaw_des   - yaw_angle
        pitch_err = pitch_des - pitch_angle

        yaw_torque   = Kp_gimbal * yaw_err   - Kd_gimbal * yaw_vel
        pitch_torque = Kp_gimbal * pitch_err - Kd_gimbal * pitch_vel

        yaw_torque   = np.clip(yaw_torque,   -MAX_GIMBAL_TORQUE, MAX_GIMBAL_TORQUE)
        pitch_torque = np.clip(pitch_torque, -MAX_GIMBAL_TORQUE, MAX_GIMBAL_TORQUE)

        # -----------------------------------------------------
        # Apply controls
        # -----------------------------------------------------
        data.ctrl[thrust_act_id] = thrust_cmd
        data.ctrl[yaw_act_id]    = yaw_torque
        data.ctrl[pitch_act_id]  = pitch_torque

        # -----------------------------------------------------
        # Fuel burn & mass/inertia update
        # -----------------------------------------------------
        if fuel_mass > 0.0 and thrust_cmd > 0.0:
            mdot = -thrust_cmd / (ISP * G0)
            fuel_mass += mdot * DT
            fuel_mass = max(0.0, fuel_mass)

        total_mass = DRY_MASS + fuel_mass
        mass_ratio = total_mass / initial_total_mass

        model.body_mass[rocket_bid]    = total_mass
        model.body_inertia[rocket_bid] = orig_inertia * mass_ratio

        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------
        print(f"Step {step:04d}")
        print(f"  Pos           : {pos}")
        print(f"  Target        : {TARGET_POS}")
        print(f"  Thrust (N)    : {thrust_cmd:.1f}")
        print(f"  Fuel Mass (kg): {fuel_mass:.3f}")
        print(f"  Gimbal yaw    : {np.degrees(yaw_angle):.2f} deg (des {np.degrees(yaw_des):.2f})")
        print(f"  Gimbal pitch  : {np.degrees(pitch_angle):.2f} deg (des {np.degrees(pitch_des):.2f})\n")

        step += 1
        viewer.sync()
        time.sleep(0.01)
