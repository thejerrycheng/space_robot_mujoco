#!/usr/bin/env python3
"""
Rocket PID trajectory tracking (MuJoCo)
- Tracks a simple parabola in X-Z from the initial position.
- Controls thrust magnitude (actuator 'thrust') and gimbal yaw/pitch (motors).
- Logs position and velocity over time and plots them at the end.
"""

import math
import time
from pathlib import Path
import numpy as np
import mujoco as mj
import mujoco.viewer
import matplotlib.pyplot as plt
from pathlib import Path
import os
# Prefer a safe backend on macOS terminals
import os
import matplotlib

# Try Qt first, then Tk, else headless
for backend in ("QtAgg", "TkAgg", "Agg"):
    try:
        matplotlib.use(backend)
        break
    except Exception:
        continue



# pid.py is next to the folder "mjcf", and the XML you’re using is "ball_thrust.xml"
XML_PATH = (Path(__file__).resolve().parent / "ball_thrust.xml")
print("Loading XML ->", XML_PATH)



# ----------------------------
# Config
# ----------------------------
SIM_DT    = 0.01
CTRL_HZ   = 100          # control at every step (since dt=0.01)
SIM_TIME  = 15.0         # seconds

# Rocket (approx). You can refine using model masses if desired.
ROCKET_MASS = 1.10       # kg (body + mount + gimbal approx)
THRUST_MAX  = 20.0       # from XML ctrlrange

# Outer-loop PD on position
Kp_pos = np.array([6.0, 6.0, 10.0])
Kd_pos = np.array([3.5, 3.5, 6.0])

# Inner-loop PD for gimbal joints (torque motors)
Kp_gim = 8.0
Kd_gim = 0.3

# Gimbal angle limits (rad) — keep in sync with XML ±70 deg
GIMBAL_LIM = math.radians(70.0)


# ----------------------------
# Helpers
# ----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])

def body_rot_mat(data, body_name):
    """World-from-body rotation (3x3) for a body."""
    bid = mj.mj_name2id(data.model, mj.mjtObj.mjOBJ_BODY, body_name)
    R = np.array(data.xmat[bid]).reshape(3, 3)  # row-major
    return R

def body_pos(data, body_name):
    bid = mj.mj_name2id(data.model, mj.mjtObj.mjOBJ_BODY, body_name)
    return np.array(data.xpos[bid])

def body_linvel_world(data, body_name):
    """Linear velocity in world frame."""
    bid = mj.mj_name2id(data.model, mj.mjtObj.mjOBJ_BODY, body_name)
    return np.array(data.cvel[bid][3:])  # cvel = [angVel, linVel] in *world*
    # If your MuJoCo build doesn't have cvel, use: data.body_xvelp[bid]

def joint_q_qd(data, joint_name):
    jid = mj.mj_name2id(data.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    adr = data.model.jnt_qposadr[jid]
    vadr = data.model.jnt_dofadr[jid]
    q = data.qpos[adr]
    qd = data.qvel[vadr]
    return float(q), float(qd)

def actuator_id(model, name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)


# ----------------------------
# Reference trajectory (parabola in X-Z)
# x(t) = x0 + Vx * t
# z(t) = z0 + a*t - b*t^2
# y(t) = y0  (keep centered)
# ----------------------------
def make_parabola(x0, y0, z0):
    Vx = 1.0       # m/s forward
    a  = 2.0       # upward initial slope (m/s)
    b  = 0.3       # curvature

    def r_des(t):
        return np.array([x0 + Vx*t, y0, z0 + a*t - b*t*t])

    def v_des(t):
        return np.array([Vx, 0.0, a - 2*b*t])

    return r_des, v_des


# ----------------------------
# Main
# ----------------------------
def main():
    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)

    # Quick sanity: gravity vector from model
    gvec = np.array(model.opt.gravity)  # e.g., [0, 0, -9.81]

    # IDs
    thrust_id = actuator_id(model, "thrust")
    yaw_id    = actuator_id(model, "yaw_motor")
    pitch_id  = actuator_id(model, "pitch_motor")

    # joint names (from your XML)
    J_YAW   = "thruster_yaw"
    J_PITCH = "thruster_pitch"

    # Bodies
    B_ROCKET = "ball"              # your parent rocket body name
    B_MOUNT  = "thruster_mount"    # parent frame for gimbal yaw

    # Initial reference anchored at initial pose
    p0 = body_pos(data, B_ROCKET)
    r_des_fun, v_des_fun = make_parabola(p0[0], p0[1], p0[2])

    # Logging
    T = int(SIM_TIME / SIM_DT)
    t_log   = np.zeros(T)
    p_log   = np.zeros((T, 3))
    v_log   = np.zeros((T, 3))
    rref_log = np.zeros((T, 3))
    vref_log = np.zeros((T, 3))
    u_thrust_log = np.zeros(T)
    yaw_cmd_log  = np.zeros(T)
    pit_cmd_log  = np.zeros(T)

    # Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -15

        t = 0.0
        for k in range(T):
            # --- Read state ---
            p = body_pos(data, B_ROCKET)
            v = body_linvel_world(data, B_ROCKET)

            # --- Reference ---
            r_ref = r_des_fun(t)
            v_ref = v_des_fun(t)

            # --- Outer-loop PD to get desired world acceleration (no integral for simplicity) ---
            pos_err = r_ref - p
            vel_err = v_ref - v
            a_cmd = Kp_pos * pos_err + Kd_pos * vel_err   # m/s^2

            # Force needed (account for gravity)
            F_world = ROCKET_MASS * (a_cmd - gvec)

            # Clamp to achievable magnitude
            F_mag = np.linalg.norm(F_world)
            if F_mag < 1e-9:
                F_dir_world = np.array([0.0, 0.0, 1.0])
                F_mag = 0.0
            else:
                F_dir_world = F_world / F_mag

            # Desired thrust magnitude (limited by actuator ctrlrange)
            u_thrust = clamp(F_mag, 0.0, THRUST_MAX)

            # Map desired world force direction to mount frame
            R_mount = body_rot_mat(data, B_MOUNT)       # world-from-mount
            v_mount = R_mount.T @ F_dir_world           # in mount local frame

            # Compute desired yaw/pitch so that local +Z aligns with v_mount.
            # Using d = Rz(yaw)*Ry(pitch)*[0,0,1] = [cos(yaw)*sin(pitch), sin(yaw)*sin(pitch), cos(pitch)]
            v_hat = unit(v_mount)
            dz = clamp(v_hat[2], -1.0, 1.0)
            pitch_des = math.acos(dz)                   # in [0,pi]
            # Handle near-zero sin(pitch) robustly
            s = math.sqrt(max(1.0 - dz*dz, 0.0))
            if s < 1e-6:
                yaw_des = 0.0
            else:
                yaw_des = math.atan2(v_hat[1], v_hat[0])

            # Wrap to limits
            yaw_des   = clamp(yaw_des,  -GIMBAL_LIM, GIMBAL_LIM)
            pitch_des = clamp(pitch_des, -GIMBAL_LIM, GIMBAL_LIM)

            # Inner-loop PD on gimbal joints -> motor torques
            q_yaw,   qd_yaw   = joint_q_qd(data, J_YAW)
            q_pitch, qd_pitch = joint_q_qd(data, J_PITCH)

            tau_yaw   = Kp_gim * (yaw_des - q_yaw)   - Kd_gim * qd_yaw
            tau_pitch = Kp_gim * (pitch_des - q_pitch) - Kd_gim * qd_pitch

            # Respect actuator ctrl ranges for motors (±2 N·m per your XML default)
            motor_max = 2.0
            tau_yaw   = clamp(tau_yaw,   -motor_max, motor_max)
            tau_pitch = clamp(tau_pitch, -motor_max, motor_max)

            # --- Apply controls: [yaw_motor, pitch_motor, thrust] (order by actuator id) ---
            data.ctrl[yaw_id]   = tau_yaw
            data.ctrl[pitch_id] = tau_pitch
            data.ctrl[thrust_id]= u_thrust

            # Step sim
            mj.mj_step(model, data)

            # Log
            t_log[k] = t
            p_log[k, :] = p
            v_log[k, :] = v
            rref_log[k, :] = r_ref
            vref_log[k, :] = v_ref
            u_thrust_log[k] = u_thrust
            yaw_cmd_log[k]  = yaw_des
            pit_cmd_log[k]  = pitch_des

            # Render
            viewer.sync()
            t += SIM_DT

    # ----------------------------
    # Plot logs
    # ----------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(t_log, p_log[:, 0], label="x")
    axs[0].plot(t_log, p_log[:, 1], label="y")
    axs[0].plot(t_log, p_log[:, 2], label="z")
    axs[0].plot(t_log, rref_log[:, 0], "--", label="x_ref")
    axs[0].plot(t_log, rref_log[:, 1], "--", label="y_ref")
    axs[0].plot(t_log, rref_log[:, 2], "--", label="z_ref")
    axs[0].set_ylabel("Position (m)")
    axs[0].legend(loc="best")
    axs[0].grid(True)

    axs[1].plot(t_log, v_log[:, 0], label="vx")
    axs[1].plot(t_log, v_log[:, 1], label="vy")
    axs[1].plot(t_log, v_log[:, 2], label="vz")
    axs[1].plot(t_log, vref_log[:, 0], "--", label="vx_ref")
    axs[1].plot(t_log, vref_log[:, 1], "--", label="vy_ref")
    axs[1].plot(t_log, vref_log[:, 2], "--", label="vz_ref")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend(loc="best")
    axs[1].grid(True)

    axs[2].plot(t_log, u_thrust_log, label="thrust (N)")
    axs[2].plot(t_log, np.degrees(yaw_cmd_log), label="yaw_cmd (deg)")
    axs[2].plot(t_log, np.degrees(pit_cmd_log), label="pitch_cmd (deg)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Inputs")
    axs[2].legend(loc="best")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
