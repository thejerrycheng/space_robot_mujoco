import time, math, threading
from pathlib import Path
import numpy as np
from pynput import keyboard
import mujoco, mujoco.viewer

# ---------- GLOBAL TUNABLES ----------
START_Z          = 150.0*0.1        # initial COM height (m)
THRUST_STEP_N    = 1e5*0.01            # W/S increment (N)
THRUST_MAX_N     = 5e6*0.01            # clamp (N)
THRUST_DIR_BODY  = np.array([0,0,1], dtype=float)  # thrust direction in BODY frame
FLAME_RADIUS     = 1.5*0.01       # visual cylinder radius (m)
FLAME_SCALE      = 1e-5*0.01           # flame length (m) per Newton
# Nozzle site baseline (matches XML): pos.z=170, half_len=50 -> TOP=220 (kept constant)
NOZZLE_SITE_NAME = "nozzle_site"
NOZZLE_TOP_Z     = (170.0 + 50.0)*0.01
NOZZLE_RADIUS    = 20.0*0.01
NOZZLE_HALF_LEN  = 50.0*0.01           # change at runtime; pos.z auto-moves so top stays fixed
# ------------------------------------

class Teleop:
    def __init__(self): self.throttle=0.0; self.pitch=0.0; self.yaw=0.0; self._lock=threading.Lock()
    def clamp(self):
        with self._lock:
            self.throttle = min(max(self.throttle, 0.0), THRUST_MAX_N)
    def get(self):
        with self._lock: return self.throttle, self.pitch, self.yaw
    def set_throttle(self, v):
        with self._lock: self.throttle = v

def start_keyboard_listener(tele: Teleop, reset_cb):
    def on_press(key):
        try:
            c = key.char
            if c in ('w','W'): tele.throttle += THRUST_STEP_N
            elif c in ('s','S'): tele.throttle -= THRUST_STEP_N
            elif c in ('x','X'): tele.throttle  = 0.0
            elif c in ('r','R'): reset_cb()
        except AttributeError:
            # arrow keys reserved if you later add gimbal
            pass
        tele.clamp()
    lst = keyboard.Listener(on_press=on_press)
    lst.daemon = True; lst.start()
    return lst

def set_nozzle_size_and_pos(model, radius, half_len):
    """Keep nozzle TOP fixed at NOZZLE_TOP_Z when height (half_len) changes."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, NOZZLE_SITE_NAME)
    model.site_size[sid, 0] = float(radius)
    model.site_size[sid, 1] = float(half_len)
    model.site_pos[sid, 2]  = NOZZLE_TOP_Z - float(half_len)  # center moves down as it gets taller

def body_pose(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    R = data.xmat[bid].reshape(3,3).copy()
    p_com = data.xipos[bid].copy()
    return p_com, R

def apply_nozzle_thrust(model, data, tele: Teleop, body_name="rocket"):
    throttle, _, _ = tele.get()
    if throttle <= 0.0: return
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
    scn = viewer.user_scn
    scn.ngeom = 0
    L = float(thrustN * FLAME_SCALE)
    if L <= 1e-6: return
    end = start - L * dir_w
    g = scn.geoms[0]
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size[:] = (FLAME_RADIUS, FLAME_RADIUS, 0.0)
    g.rgba[:] = (0.2, 0.4, 1.0, 0.7)
    mujoco.mjv_connector(
        scn, mujoco.mjtGeom.mjGEOM_CYLINDER, float(FLAME_RADIUS),
        float(start[0]), float(start[1]), float(start[2]),
        float(end[0]),   float(end[1]),   float(end[2])
    )

def main():
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
    print(f"[init] gravity={model.opt.gravity[2]:.3f} m/s^2 | dt={dt:.4f}s | Keys: W/S (+/- thrust), X (zero), R (reset)")

    # Teleop and keyboard listener (GLOBAL, always-on)
    tele = Teleop()
    def reset_cb(): tele.set_throttle(0.0)
    start_keyboard_listener(tele, reset_cb)

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 135, -15, max(START_Z,150.0)*1.6

        t0 = last = time.perf_counter()
        t_last_log = t0

        while v.is_running():
            # clear forces
            data.xfrc_applied[:] = 0.0

            # apply off-COM thrust at nozzle
            apply_nozzle_thrust(model, data, tele)

            # step physics
            mujoco.mj_step(model, data)

            # flame from site (which tracks height automatically)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, NOZZLE_SITE_NAME)
            start = data.site_xpos[sid].copy()
            p_com, R = body_pose(model, data, "rocket")
            dir_w = R @ THRUST_DIR_BODY
            draw_flame(v, start, dir_w, tele.get()[0])

            # real-time pacing
            last += dt
            nap = last - time.perf_counter()
            if nap > 0: time.sleep(nap)

            # logs
            now = time.perf_counter()
            if now - t_last_log >= 0.5:
                sim = data.time; wall = now - t0
                x, y, z = data.qpos[0:3]
                thr = tele.get()[0]
                print(f"sim={sim:6.3f}s wall={wall:6.3f}s drift={sim-wall:+.4f}s | thrust={thr:8.1f} N | pos=({x:7.2f},{y:7.2f},{z:7.2f})")
                t_last_log = now

            v.sync()

if __name__ == "__main__":
    main()
