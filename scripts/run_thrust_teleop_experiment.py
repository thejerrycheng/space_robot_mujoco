#!/usr/bin/env python3
# scripts/run_thrust_teleop_experiment.py
import argparse, math
from dataclasses import dataclass
from typing import List

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------- CLI ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--xml", type=str, default="assets/mjcf/tintin_thrust.xml")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--kp", type=float, default=8.0)
parser.add_argument("--kd", type=float, default=0.5)
parser.add_argument("--dthrust", type=float, default=0.5)
parser.add_argument("--dangle", type=float, default=1.0)  # deg
parser.add_argument("--thrustmax", type=float, default=20.0)
args = parser.parse_args()

# -------------- Teleop state --------------
@dataclass
class Teleop:
    thrust: float = 0.0
    pitch_d_deg: float = 0.0   # + => pitch up about local Y
    yaw_d_deg: float = 0.0     # + => yaw right about local X
    paused: bool = False
tele = Teleop()

# -------------- Load model/data --------------
model = mujoco.MjModel.from_xml_path(args.xml)
data  = mujoco.MjData(model)
model.opt.timestep = args.dt

# -------------- name2id (portable) --------------
def aid(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
def jid(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
def bid(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

aid_thrust = aid("thrust")
aid_yaw    = aid("yaw_motor")
aid_pitch  = aid("pitch_motor")

jid_yaw    = jid("thruster_yaw")
jid_pitch  = jid("thruster_pitch")
ball_bid   = bid("ball")

# qpos/qvel addresses for PD
adr_yaw_q     = model.jnt_qposadr[jid_yaw]
adr_pitch_q   = model.jnt_qposadr[jid_pitch]
adr_yaw_dq    = model.jnt_dofadr[jid_yaw]
adr_pitch_dq  = model.jnt_dofadr[jid_pitch]

THRUST_MIN, THRUST_MAX = 0.0, min(args.thrustmax, model.actuator_ctrlrange[aid_thrust, 1])

HELP = """
Teleop:
  Thrust:   '=' increase, '-' decrease, '0' zero
  Gimbal:   ↑ pitch+, ↓ pitch-, ← yaw-, → yaw+
  Reset:    r       (zero pitch/yaw)
  Pause:    space
  Quit:     q (or close window)
"""

# -------------- Robust keyboard handler --------------
def _inc_thrust(d):  tele.thrust = float(np.clip(tele.thrust + d, THRUST_MIN, THRUST_MAX))
def _inc_pitch(d):   tele.pitch_d_deg += d
def _inc_yaw(d):     tele.yaw_d_deg   += d

def on_key(keycode: int):
    """Works with ASCII and mjtKey (GLFW) codes."""
    # ASCII path (letters, symbols)
    if 32 <= (keycode & 0xFF) <= 126:
        ch = chr(keycode & 0xFF).lower()
        if ch == '=': _inc_thrust(+args.dthrust)
        elif ch == '-': _inc_thrust(-args.dthrust)
        elif ch == '0': tele.thrust = 0.0
        elif ch == 'r': tele.pitch_d_deg = 0.0; tele.yaw_d_deg = 0.0
        elif ch == 'q': raise SystemExit
        elif ch == ' ': tele.paused = not tele.paused

    # GLFW-style special keys (arrows, space, minus, equal)
    k = mujoco.mjtKey
    if keycode == k.mjKEY_UP:       _inc_pitch(+args.dangle)
    elif keycode == k.mjKEY_DOWN:   _inc_pitch(-args.dangle)
    elif keycode == k.mjKEY_LEFT:   _inc_yaw(-args.dangle)
    elif keycode == k.mjKEY_RIGHT:  _inc_yaw(+args.dangle)
    elif keycode == k.mjKEY_MINUS:  _inc_thrust(-args.dthrust)
    elif keycode == k.mjKEY_EQUAL:  _inc_thrust(+args.dthrust)
    elif keycode == k.mjKEY_0:      tele.thrust = 0.0
    elif keycode == k.mjKEY_SPACE:  tele.paused = not tele.paused
    elif keycode == k.mjKEY_ESCAPE: raise SystemExit

# -------------- Logging --------------
ts:   List[float] = []
pos:  List[np.ndarray] = []
thrusts: List[float] = []
pitch_deg_log: List[float] = []
yaw_deg_log:   List[float] = []

# -------------- Sim loop --------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Bind to either attribute name to cover MuJoCo viewer variants
    if hasattr(viewer, "user_key_callback"):
        viewer.user_key_callback = on_key
    elif hasattr(viewer, "user_keyboard_callback"):
        viewer.user_keyboard_callback = on_key

    print(HELP)

    while viewer.is_running():
        # if not tele.paused:
        #     # --- PD control on gimbals ---
        #     q_pitch = float(data.qpos[adr_pitch_q])
        #     q_yaw   = float(data.qpos[adr_yaw_q])
        #     dq_pitch= float(data.qvel[adr_pitch_dq])
        #     dq_yaw  = float(data.qvel[adr_yaw_dq])

        #     pitch_d = math.radians(tele.pitch_d_deg)
        #     yaw_d   = math.radians(tele.yaw_d_deg)

        #     data.ctrl[aid_pitch] = args.kp*(pitch_d - q_pitch) - args.kd*dq_pitch
        #     data.ctrl[aid_yaw]   = args.kp*(yaw_d   - q_yaw)   - args.kd*dq_yaw

        #     # +Z thrust in the thrust_site frame (per your XML gear="0 0 1 0 0 0")
        #     data.ctrl[aid_thrust] = tele.thrust

        #     mujoco.mj_step(model, data)

        #     ts.append(data.time)
        #     pos.append(data.xpos[ball_bid].copy())   # world position of body
        #     thrusts.append(tele.thrust)
        #     pitch_deg_log.append(tele.pitch_d_deg)
        #     yaw_deg_log.append(tele.yaw_d_deg)

        viewer.title = (
            f"Thrust={tele.thrust:.2f} N | "
            f"Pitch={tele.pitch_d_deg:.1f}°  Yaw={tele.yaw_d_deg:.1f}° | "
            f"{'PAUSED' if tele.paused else 'RUN'}"
        )
        viewer.sync()

# -------------- Plots --------------
if len(ts) >= 2:
    pos = np.vstack(pos)
    dt   = np.diff(ts)
    dpos = np.diff(pos, axis=0)
    speed = np.r_[0.0, np.linalg.norm(dpos / dt[:, None], axis=1)]

    fig = plt.figure(figsize=(11, 9))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1])

    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=2)
    ax3d.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], s=30, label="start")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.set_title("Rocket COM trajectory"); ax3d.legend(loc="best")

    ax1 = fig.add_subplot(gs[1, 0]); ax1.plot(ts, thrusts)
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("Thrust [N]"); ax1.set_title("Thrust command")

    ax2 = fig.add_subplot(gs[1, 1]); ax2.plot(ts, speed)
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("Speed [m/s]"); ax2.set_title("‖v‖ (finite diff)")

    ax3 = fig.add_subplot(gs[2, 0]); ax3.plot(ts, pitch_deg_log)
    ax3.set_xlabel("t [s]"); ax3.set_ylabel("Pitch [deg]"); ax3.set_title("Pitch setpoint")

    ax4 = fig.add_subplot(gs[2, 1]); ax4.plot(ts, yaw_deg_log)
    ax4.set_xlabel("t [s]"); ax4.set_ylabel("Yaw [deg]"); ax4.set_title("Yaw setpoint")

    fig.tight_layout(); plt.show()
else:
    print("No data logged.")
