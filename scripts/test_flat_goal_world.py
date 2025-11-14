#!/usr/bin/env python3
import os, time, math, numpy as np
from pynput import keyboard
import mujoco, mujoco.viewer

# ===== sim constants =====
G0 = 9.80665
DRY_MASS   = 1000.0
FUEL_MASS0 = 1800.0
ISP        = 320.0
T_MIN      = 0.0
T_MAX      = 500000.0
GIMBAL_MAX_DEG = 10.0

THROTTLE_STEP   = 0.02
GIMBAL_STEP_DEG = 0.5

# ---------- teleop ----------
class Teleop:
    def __init__(self):
        self.throttle = 0.0
        self.pitch = 0.0
        self.yaw   = 0.0
        self.gimbal_lim = math.radians(GIMBAL_MAX_DEG)
    def clamp(self):
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        self.pitch = float(np.clip(self.pitch, -self.gimbal_lim, self.gimbal_lim))
        self.yaw   = float(np.clip(self.yaw,   -self.gimbal_lim, self.gimbal_lim))

def start_keyboard_listener(tele: Teleop, reset_cb, move_goal_cb):
    def on_press(key):
        try:
            c = key.char
            if c in ('w','W'): tele.throttle += THROTTLE_STEP
            elif c in ('s','S'): tele.throttle -= THROTTLE_STEP
            elif c in ('x','X'): tele.throttle  = 0.0
            elif c in ('z','Z'): tele.pitch = tele.yaw = 0.0
            elif c in ('r','R'): reset_cb()
            elif c in ('g','G'): move_goal_cb()
        except AttributeError:
            if key == keyboard.Key.up:    tele.pitch += math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.down:  tele.pitch -= math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.left:  tele.yaw   += math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.right: tele.yaw   -= math.radians(GIMBAL_STEP_DEG)
        tele.clamp(); return True
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True; listener.start(); return listener

# ---------- thrust direction helpers ----------
def thrust_force_dir_body(pitch, yaw):
    """
    FORCE direction on the rocket in BODY frame.
    Convention:
      - Body +Z is the rocket nose.
      - pitch = Rx(pitch), yaw = Ry(yaw).
      - With pitch=yaw=0, force_dir = +Z (exhaust is -Z).
    """
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    # R_x @ R_y @ e_z  => [ sp*cy,  sy,  cp*cy ]
    fx =  sp * cy
    fy =  sy
    fz =  cp * cy
    v = np.array([fx, fy, fz], dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)

def thrust_exhaust_dir_body(pitch, yaw):
    """Exhaust (plume) direction in body frame (opposite of force)."""
    return -thrust_force_dir_body(pitch, yaw)

# ---------- physics helpers ----------
def set_mass_and_inertia(m, bid, new_mass, ref_mass=None):
    if ref_mass is None: ref_mass = float(m.body_mass[bid])
    scale = float(new_mass / max(ref_mass, 1e-9))
    m.body_mass[bid] = float(new_mass)
    m.body_inertia[bid, :] *= scale

def apply_thrust(m, d, bid, sid, tele, mass_state):
    dt = m.opt.timestep
    T_cmd = float(np.clip(tele.throttle * T_MAX, T_MIN, T_MAX))
    mdot = T_cmd / (ISP * G0)
    fuel = max(0.0, mass_state['fuel_mass'] - mdot * dt)
    if fuel <= 0.0:
        T_cmd = 0.0; tele.throttle = 0.0
    new_mass = DRY_MASS + fuel
    if abs(new_mass - mass_state['mass']) > 1e-9:
        set_mass_and_inertia(m, bid, new_mass, ref_mass=mass_state['mass'])
        mujoco.mj_forward(m, d)
        mass_state['mass'] = new_mass
        mass_state['fuel_mass'] = fuel

    # --- directions in body & world ---
    force_dir_body   = thrust_force_dir_body(tele.pitch, tele.yaw)   # +nose at zero
    exhaust_dir_body = -force_dir_body                               # -nose at zero (for visuals)
    Rwb = d.xmat[bid].reshape(3,3)
    force_dir_world = Rwb @ force_dir_body

    # --- force & torque on the body ---
    F = force_dir_world * T_cmd
    p = d.site_xpos[sid]
    com = d.xipos[bid]
    tau = np.cross(p - com, F)

    d.xfrc_applied[bid, :] = 0.0
    d.xfrc_applied[bid, 0:3] = F
    d.xfrc_applied[bid, 3:6] = tau

    return force_dir_body, force_dir_world, exhaust_dir_body, T_cmd

# For a free joint root, world linear velocity = qvel[0:3]
def root_linvel_world(d): return d.qvel[0:3].copy()

# ---------- main ----------
if __name__ == "__main__":
    xml = os.path.join(os.path.dirname(__file__), "..", "assets", "mjcf", "new_world.xml")
    xml = os.path.abspath(xml)
    print("XML ->", xml)
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)

    rocket_bid = m.body('rocket_root').id
    engine_sid = m.site('engine_site').id
    goal_bid   = m.body('goal_marker').id

    mass_state = {'fuel_mass': FUEL_MASS0, 'mass': DRY_MASS + FUEL_MASS0}
    set_mass_and_inertia(m, rocket_bid, mass_state['mass'])
    mujoco.mj_forward(m, d)

    tele = Teleop()
    rng = np.random.default_rng(42)

    def reset_ic():
        qpos = d.qpos.copy(); qvel = d.qvel.copy()

        # --- Position: spawn high; allow horizontal offset up to 200 m ---
        x0 = rng.uniform(-200, 200)
        y0 = rng.uniform(-200, 200)
        z0 = rng.uniform(5000, 7000)       # meters AGL
        qpos[0:3] = np.array([x0, y0, z0])

        # --- Orientation: +90° about X so mesh +Y becomes +Z (nose up) ---
        # If your XML already rotates the body (euler="1.5708 0 0"), set that to 0 0 0.
        # half = math.sqrt(0.5)              # cos(pi/4) == sin(pi/4)
        qpos[3:7] = np.array([0.0, 0.0, 0.0, 0.0])  # [w,x,y,z]

        # --- Velocity: pure downward; no lateral, no spin ---
        vz0 = rng.uniform(-600.0, -300.0)
        qvel[0:3] = np.array([0.0, 0.0, vz0])
        qvel[3:6] = np.array([0.0, 0.0, 0.0])

        d.qpos[:] = qpos; d.qvel[:] = qvel; d.time = 0.0

        # reset mass & teleop
        tele.throttle = 0.0; tele.pitch = tele.yaw = 0.0
        mass_state['fuel_mass'] = FUEL_MASS0
        set_mass_and_inertia(m, rocket_bid, DRY_MASS + FUEL_MASS0, ref_mass=mass_state['mass'])
        mass_state['mass'] = DRY_MASS + FUEL_MASS0
        mujoco.mj_forward(m, d)

        print(f"[reset] start @ ({x0:+.1f},{y0:+.1f},{z0:.1f}) m, vz={vz0:.1f} m/s (pure down, nose +Z)")

    def move_goal():
        gx = rng.uniform(-500, 500)
        gy = rng.uniform(-500, 500)
        m.body_pos[goal_bid, 0:3] = np.array([gx, gy, 0.0])
        mujoco.mj_forward(m, d)
        print(f"[goal] moved to ({gx:.1f},{gy:.1f})")

    reset_ic()
    move_goal()
    start_keyboard_listener(tele, reset_ic, move_goal)

    with mujoco.viewer.launch_passive(m, d) as v:
        # Camera for km scale
        v.cam.lookat[:] = d.xipos[rocket_bid]
        v.cam.distance   = 3000.0
        v.cam.azimuth    = 130.0
        v.cam.elevation  = -35.0

        last_print = 0.0
        while v.is_running():
            step_start = time.time()

            # Apply thrust based on current teleop, then ADVANCE PHYSICS
            force_dir_body, force_dir_world, exhaust_dir_body, T_cmd = apply_thrust(
                m, d, rocket_bid, engine_sid, tele, mass_state
            )
            mujoco.mj_step(m, d)  # <-- MISSING BEFORE: advance the sim!

            if time.time() - last_print > 0.5:
                vz = float(root_linvel_world(d)[2])
                z  = float(d.xipos[rocket_bid,2])
                print(f"t={d.time:6.2f}s  thr={tele.throttle:0.2f}  "
                      f"pitch={math.degrees(tele.pitch):+5.1f}°  yaw={math.degrees(tele.yaw):+5.1f}°  "
                      f"T={T_cmd/1000:7.1f} kN  z={z:7.1f} m  vz={vz:6.1f} m/s")
                last_print = time.time()

            v.sync()
            dt = m.opt.timestep - (time.time() - step_start)
            if dt > 0: time.sleep(dt)

    print("Sim closed.")
