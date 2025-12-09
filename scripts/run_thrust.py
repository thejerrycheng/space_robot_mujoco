import sys
import time
import math
import select
import numpy as np
import mujoco as mj
import mujoco.viewer
from pathlib import Path


XML_PATH = Path("assets/mjcf/tintin_view.xml")
model = mj.MjModel.from_xml_path(str(XML_PATH))

# ---------- Controls (you can change these live from the terminal) ----------
# Angles in degrees for human friendliness; we convert to radians internally.
state = {
    "magnitude": 5.0,   # Newtons
    "pitch_deg": 0.0,   # rotation about world Y (up/down)
    "yaw_deg": 0.0,     # rotation about world Z (left/right)
    "scale_N_to_len": 0.05,  # viz: meters of cylinder per Newton (tweak freely)
    "running": True,
}

HELP = """
Controls (type commands and press Enter; they apply immediately):

  m <value>        set magnitude in N            (e.g.,  m 12)
  dm <delta>       add to magnitude              (e.g.,  dm -1.5)

  p <deg>          set pitch  (degrees, +up)     (e.g.,  p 15)
  dp <delta>       add to pitch                   (e.g.,  dp -5)

  y <deg>          set yaw    (degrees, +CCW)     (e.g.,  y 45)
  dy <delta>       add to yaw                      (e.g.,  dy 10)

  s <k>            set viz scale (m per N)       (e.g.,  s 0.08)

  r                reset ball pose/vel to origin
  h                show this help
  q                quit

Notes:
- Pitch is rotation about world Y, yaw is about world Z.
- Thrust cylinder points along +Z of the viz mocap; we align it to (yaw,pitch).
- Cylinder half-length = max(0.05, scale_N_to_len * magnitude / 2).
"""

def deg2rad(d):
    return d * math.pi / 180.0

def yaw_pitch_to_unitvec(yaw_deg, pitch_deg):
    """Return a unit vector in world frame from yaw (Z) and pitch (Y), roll=0."""
    yaw = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    # Direction (x,y,z) using aerospace-style Z then Y
    # x = cos(pitch)*cos(yaw)
    # y = cos(pitch)*sin(yaw)
    # z = sin(pitch)
    cp = math.cos(pitch)
    return np.array([cp*math.cos(yaw), cp*math.sin(yaw), math.sin(pitch)], dtype=float)

def yaw_pitch_to_quat(yaw_deg, pitch_deg):
    """Quaternion (w,x,y,z) for Rz(yaw) * Ry(pitch) with roll=0."""
    yaw = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    # roll=0 -> cr=1, sr=0; compose q = qz * qy
    # Using scalar-first convention (w,x,y,z)
    w = cy*cp
    x = 0.0
    y = cy*sp
    z = sy*cp
    # Normalize for safety
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)

def apply_thrust(model, data, force_world, body_name="ball"):
    """Apply a spatial force (N) to the body in world coordinates at its center."""
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    # xfrc_applied is in world frame for each body (force[0:3], torque[3:6])
    data.xfrc_applied[body_id, :3] = force_world
    data.xfrc_applied[body_id, 3:] = 0.0

def update_thrust_visual(model, data, force_world, yaw_deg, pitch_deg):
    """Move the mocap body to ball position, orient along thrust, and set cylinder length."""
    # Where is the ball?
    ball_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "ball")
    ball_pos_w = data.xpos[ball_bid].copy()

    # Set mocap position/orientation
    mocap_id = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "thrust_viz")]
    data.mocap_pos[mocap_id] = ball_pos_w
    data.mocap_quat[mocap_id] = yaw_pitch_to_quat(yaw_deg, pitch_deg)

    # Adjust site half-length based on |force|
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "thrust_site")
    site_geomid = model.site_geomadr[site_id]   # the (auto) geom backing the site

    mag = np.linalg.norm(force_world)
    half_len = max(0.05, state["scale_N_to_len"] * mag * 0.5)  # keep a minimum so it's visible
    # site size = (radius, half-length). We'll keep radius fixed (index 0), change index 1.
    # Sites share the same size array layout as geoms in the model.
    model.site_size[site_id, 1] = half_len

def parse_command(line):
    """Parse a single-line command from stdin."""
    parts = line.strip().split()
    if not parts:
        return
    cmd = parts[0].lower()
    try:
        if cmd == "m" and len(parts) == 2:
            state["magnitude"] = float(parts[1])
        elif cmd == "dm" and len(parts) == 2:
            state["magnitude"] += float(parts[1])
        elif cmd == "p" and len(parts) == 2:
            state["pitch_deg"] = float(parts[1])
        elif cmd == "dp" and len(parts) == 2:
            state["pitch_deg"] += float(parts[1])
        elif cmd == "y" and len(parts) == 2:
            state["yaw_deg"] = float(parts[1])
        elif cmd == "dy" and len(parts) == 2:
            state["yaw_deg"] += float(parts[1])
        elif cmd == "s" and len(parts) == 2:
            state["scale_N_to_len"] = float(parts[1])
        elif cmd == "r":
            # Reset ball pose/vel
            qpos_addr = slice(0, 7)   # free joint starts at 0, len 7 (quat+pos)
            qvel_addr = slice(0, 6)
            data.qpos[qpos_addr] = np.array([1, 0, 0, 0, 0, 0, 0])  # unit quat + zero pos
            data.qvel[qvel_addr] = 0.0
            mj.mj_forward(model, data)
        elif cmd == "h":
            print(HELP, flush=True)
        elif cmd == "q":
            state["running"] = False
        else:
            print("Unrecognized command. Type 'h' for help.", flush=True)
    except ValueError:
        print("Bad number format. Type 'h' for help.", flush=True)

def nonblocking_readline():
    """Return a line from stdin if available, else None (POSIX/macOS)."""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline()
    return None


if __name__ == "__main__":
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)

    print(HELP)
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and state["running"]:
            # Clear any previously applied external forces (MuJoCo accumulates per step)
            data.xfrc_applied[:] = 0.0

            # Compute thrust (world)
            dir_w = yaw_pitch_to_unitvec(state["yaw_deg"], state["pitch_deg"])
            force_w = dir_w * state["magnitude"]

            # Apply to ball
            apply_thrust(model, data, force_w, body_name="ball")

            # Update visualization
            update_thrust_visual(model, data, force_w, state["yaw_deg"], state["pitch_deg"])

            # Step sim
            mj.mj_step(model, data)

            # Render
            viewer.sync()

            # Print status at ~2 Hz
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                print(f"[mag={state['magnitude']:.2f} N | yaw={state['yaw_deg']:.1f}° | pitch={state['pitch_deg']:.1f}° | scale={state['scale_N_to_len']:.3f} m/N]",
                      end="\r", flush=True)

            # Handle user input (non-blocking)
            line = nonblocking_readline()
            if line is not None and line.strip() != "":
                parse_command(line)

    print("\nExiting.")
