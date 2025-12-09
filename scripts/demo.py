import os
import platform
import time
from pathlib import Path
import numpy as np
import mujoco, mujoco.viewer

# ---------- GLOBAL TUNABLES ----------
START_Z           = 150.0        # initial COM height (m)
THRUST_CONST_N    = 1.0e7        # constant thrust (N)
THRUST_MAX_N      = 5.0e6        # clamp (N)
THRUST_DIR_BODY   = np.array([0, 0, 1], dtype=float)  # thrust direction in BODY frame
FLAME_RADIUS      = 1.5          # visual cylinder radius (m)
FLAME_SCALE       = 1e-5         # flame length (m) per Newton
LOG_EVERY_S       = 0.50         # print state/action at this cadence
# Nozzle site baseline (matches XML): pos.z=170, half_len=50 -> TOP=220 (kept constant)
NOZZLE_SITE_NAME  = "nozzle_site"
NOZZLE_TOP_Z      = 170.0 + 50.0
NOZZLE_RADIUS     = 20.0
NOZZLE_HALF_LEN   = 50.0         # change at runtime; pos.z auto-moves so top stays fixed
# ------------------------------------


def clamp_thrust(v: float) -> float:
    return float(min(max(v, 0.0), THRUST_MAX_N))


def set_nozzle_size_and_pos(model, radius, half_len):
    """Keep nozzle TOP fixed at NOZZLE_TOP_Z when height (half_len) changes."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, NOZZLE_SITE_NAME)
    model.site_size[sid, 0] = float(radius)
    model.site_size[sid, 1] = float(half_len)
    model.site_pos[sid, 2]  = NOZZLE_TOP_Z - float(half_len)  # center moves down as it gets taller


def body_pose(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    R = data.xmat[bid].reshape(3, 3).copy()
    p_com = data.xipos[bid].copy()
    return p_com, R


def apply_nozzle_thrust(model, data, throttleN: float, body_name="rocket"):
    throttle = clamp_thrust(throttleN)
    if throttle <= 0.0:
        return
    p_com, R = body_pose(model, data, body_name)
    dir_w = R @ THRUST_DIR_BODY
    F = throttle * dir_w

    # nozzle point from site (follows size/pos automatically)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, NOZZLE_SITE_NAME)
    p_noz = data.site_xpos[sid].copy()
    r = p_noz - p_com

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data.xfrc_applied[bid, 0:3] += F
    data.xfrc_applied[bid, 3:6] += np.cross(r, F)


def draw_flame(viewer, start, dir_w, thrustN):
    """Draw a connector cylinder from nozzle site along -thrust direction.
    Compatible with mujoco.mjv_connector(geom, type, width, from, to).
    Uses float64 contiguous arrays (more robust on macOS).
    """
    scn = viewer.user_scn
    L = float(thrustN * FLAME_SCALE)

    # Clear any previous user geoms when no flame
    if L <= 1e-6:
        scn.ngeom = 0
        return

    start = np.ascontiguousarray(np.asarray(start, dtype=np.float64).reshape(3))
    end   = np.ascontiguousarray(start - L * np.ascontiguousarray(np.asarray(dir_w, dtype=np.float64).reshape(3)))

    g = scn.geoms[0]
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        float(FLAME_RADIUS),
        start,
        end,
    )
    # styling (after connector call so it doesn't get overwritten)
    g.rgba[:] = (0.2, 0.4, 1.0, 0.7)
    scn.ngeom = 1


def main():
    # macOS viewer hint: ensure we're using GLFW (Metal-backed on macOS)
    if platform.system() == "Darwin":
        os.environ.setdefault("MUJOCO_GL", "glfw")

    xml_path = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "tintin_view.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)

    # Initial pose (free-fall start height)
    data.qpos[0:3] = (0.0, 0.0, START_Z)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    # Ensure nozzle site is at baseline values (top fixed = 220 m by default)
    set_nozzle_size_and_pos(model, NOZZLE_RADIUS, NOZZLE_HALF_LEN)
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    thrustN = clamp_thrust(THRUST_CONST_N)
    print(
        f"[init] gravity={model.opt.gravity[2]:.3f} m/s^2 | dt={dt:.4f}s | THRUST_CONST_N={thrustN:.1f} N"
    )

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 135, -15, max(START_Z, 150.0) * 1.6

        t0 = last = time.perf_counter()
        t_last_log = t0

        while v.is_running():
            # clear forces
            data.xfrc_applied[:] = 0.0

            # apply off-COM thrust at nozzle (constant action)
            apply_nozzle_thrust(model, data, thrustN)

            # step physics
            mujoco.mj_step(model, data)

            # flame from site (which tracks height automatically)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, NOZZLE_SITE_NAME)
            start = data.site_xpos[sid].copy()
            p_com, R = body_pose(model, data, "rocket")
            dir_w = R @ THRUST_DIR_BODY
            draw_flame(v, start, dir_w, thrustN)

            # real-time pacing
            last += dt
            nap = last - time.perf_counter()
            if nap > 0:
                time.sleep(nap)

            # logs: state (pos, vel, attitude rough via quaternion) + action (thrust)
            now = time.perf_counter()
            if now - t_last_log >= LOG_EVERY_S:
                sim = data.time
                wall = now - t0
                x, y, z = data.qpos[0:3]
                vx, vy, vz = data.qvel[0:3]
                qw, qx, qy, qz = data.qpos[3:7]
                print(
                    f"sim={sim:6.3f}s wall={wall:6.3f}s drift={sim-wall:+.4f}s | "
                    f"thrust={thrustN:8.1f} N | pos=({x:7.2f},{y:7.2f},{z:7.2f}) | "
                    f"vel=({vx:7.3f},{vy:7.3f},{vz:7.3f}) | quat=({qw:+.4f},{qx:+.4f},{qy:+.4f},{qz:+.4f})"
                )
                t_last_log = now

            v.sync()


if __name__ == "__main__":
    main()
