#!/usr/bin/env python3
import os, argparse, time, math
import numpy as np
from pynput import keyboard
import mujoco
import mujoco.viewer

# -- log out the data here ---- 
import csv, datetime
from collections import deque

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


G0 = 9.80665

# ----- Physical params -----
DRY_MASS   = 10000.0      # kg
FUEL_MASS0 = 18000.0      # kg  -> wet = 30 t
ISP        = 320.0        # s   (vac)
T_MIN      = 0.0          # N
T_MAX      = 500000.0    # N (120 kN)
GIMBAL_MAX_DEG = 10.0     # +/- deg
THROTTLE_STEP  = 0.02
GIMBAL_STEP_DEG = 0.5

# ----- Teleop -----
class Teleop:
    def __init__(self):
        self.throttle = 0.0
        self.pitch = 0.0   # rad
        self.yaw   = 0.0   # rad
    def clamp(self):
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        lim = math.radians(GIMBAL_MAX_DEG)
        self.pitch = float(np.clip(self.pitch, -lim, lim))
        self.yaw   = float(np.clip(self.yaw,   -lim, lim))

def start_keyboard_listener(tele: Teleop, reset_cb):
    def on_press(key):
        try:
            if key.char in ('w','W'): tele.throttle += THROTTLE_STEP
            elif key.char in ('s','S'): tele.throttle -= THROTTLE_STEP
            elif key.char in ('x','X'): tele.throttle  = 0.0
            elif key.char in ('z','Z'): tele.pitch = tele.yaw = 0.0
            elif key.char in ('r','R'): reset_cb()
        except AttributeError:
            if key == keyboard.Key.up:    tele.pitch += math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.down:  tele.pitch -= math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.left:  tele.yaw   += math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.right: tele.yaw   -= math.radians(GIMBAL_STEP_DEG)
        tele.clamp()
        return True
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener

# ----- IDs / Handles -----
def handles(m):
    return dict(
        rocket_bid = m.body('rocket_root').id,
        engine_sid = m.site('engine_site').id,
        ground_gid = m.geom('moon_ground').id,
        flame_core_gid  = m.geom('flame_core').id,
        flame_plume_gid = m.geom('flame_plume').id,
    )

# ----- Mass/Inertia -----
def set_mass_and_inertia(m, bid, new_mass, ref_mass=None):
    if ref_mass is None:
        ref_mass = float(m.body_mass[bid])
    scale = float(new_mass / max(ref_mass, 1e-9))
    m.body_mass[bid] = float(new_mass)
    m.body_inertia[bid, :] *= scale

# ----- Thrust math -----
def thrust_direction_body(pitch, yaw):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    # dir = R_x(pitch) @ R_y(yaw) @ [0,0,1]
    dx =  sp*cy
    dy =  sy
    dz =  cp*cy
    v = np.array([dx, dy, dz], dtype=float)
    return v / np.linalg.norm(v)

def apply_thrust(m, d, bid, sid, tele: Teleop, mass_state):
    dt = m.opt.timestep
    # commanded thrust
    T_cmd = np.clip(tele.throttle * T_MAX, T_MIN, T_MAX)
    # fuel burn
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

    # thrust vector in world
    dir_body  = thrust_direction_body(tele.pitch, tele.yaw)
    Rwb = d.xmat[bid].reshape(3,3)
    dir_world = Rwb @ dir_body
    F = dir_world * T_cmd

    # apply wrench at engine site
    p = d.site_xpos[sid]
    com = d.xipos[bid]
    tau = np.cross(p - com, F)

    d.xfrc_applied[bid, :] = 0.0
    d.xfrc_applied[bid, 0:3] = F
    d.xfrc_applied[bid, 3:6] = tau

    return dir_body, dir_world, T_cmd


# ----- Flame helpers -----
def rotm_from_z(z_body):
    """Build a body-local rotation matrix whose +Z aligns to z_body."""
    z = z_body / (np.linalg.norm(z_body) + 1e-12)
    # pick a non-collinear reference for x
    ref = np.array([1.0,0,0]) if abs(z[0]) < 0.9 else np.array([0,1.0,0])
    x = np.cross(ref, z); x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)  # columns are axes
    return R

def quat_from_rotm(R):
    """Convert 3x3 rotation matrix to [w,x,y,z] quaternion (MuJoCo order)."""
    t = np.trace(R)
    if t > 0:
        r = math.sqrt(1.0 + t)
        w = 0.5 * r
        s = 0.5 / r
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            r = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            x = 0.5 * r; s = 0.5 / r
            y = (R[0,1] + R[1,0]) * s
            z = (R[0,2] + R[2,0]) * s
            w = (R[2,1] - R[1,2]) * s
        elif i == 1:
            r = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            y = 0.5 * r; s = 0.5 / r
            x = (R[0,1] + R[1,0]) * s
            z = (R[1,2] + R[2,1]) * s
            w = (R[0,2] - R[2,0]) * s
        else:
            r = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            z = 0.5 * r; s = 0.5 / r
            x = (R[0,2] + R[2,0]) * s
            y = (R[1,2] + R[2,1]) * s
            w = (R[1,0] - R[0,1]) * s
    return np.array([w,x,y,z], dtype=float)

def update_flame_visual(m, d, h, dir_body, thrust_N):
    """
    Resize/recolor/reorient flame geoms based on thrust:
      - core  (ellipsoid): bright, short, tapered-ish
      - plume (cylinder) : longer translucent column
    Align +Z of each geom to the exhaust vector (-dir_body).
    """
    # throttle ratio + a tiny flicker
    r = float(thrust_N / max(T_MAX, 1e-9))
    flick = 0.92 + 0.16*np.random.random()

    # lengths/radii (meters)
    core_len   = (0.25 + 2.0 * r) * flick
    core_rad   = 0.06 + 0.14 * r
    plume_len  = (0.40 + 3.5 * r) * flick
    plume_rad  = 0.10 + 0.25 * r

    # MuJoCo sizes (all geoms have 3 slots):
    #   ellipsoid: [rx ry rz]  (half-axes)
    #   cylinder : [radius half_length unused]
    core_size  = np.array([core_rad, core_rad, 0.5*core_len], dtype=float)
    plume_size = np.array([plume_rad, 0.5*plume_len, 0.0],   dtype=float)

    # Alpha with throttle (0 = invisible at idle)
    core_rgba  = m.geom_rgba[h['flame_core_gid']].copy()
    plume_rgba = m.geom_rgba[h['flame_plume_gid']].copy()
    core_rgba[3]  = 0.0 if r <= 1e-4 else (0.25 + 0.70*r)
    plume_rgba[3] = 0.0 if r <= 1e-4 else (0.10 + 0.35*r)

    # Orientation: exhaust is -dir_body; align geom +Z to that
    z_body = -dir_body
    R = rotm_from_z(z_body)
    quat = quat_from_rotm(R)  # [w,x,y,z]

    # Anchor at engine site in the rocket frame, offset by half-length along +Z of geom
    site_local = m.site_pos[h['engine_sid']].copy()
    core_pos   = site_local + z_body * core_size[2]       # ellipsoid half-length is size[2]
    plume_pos  = site_local + z_body * plume_size[1]      # cylinder  half-length is size[1]

    # Write into the model
    m.geom_size[h['flame_core_gid'],  :] = core_size
    m.geom_size[h['flame_plume_gid'], :] = plume_size
    m.geom_rgba[h['flame_core_gid'],  :] = core_rgba
    m.geom_rgba[h['flame_plume_gid'], :] = plume_rgba
    m.geom_quat[h['flame_core_gid'],  :] = quat
    m.geom_quat[h['flame_plume_gid'], :] = quat
    m.geom_pos[h['flame_core_gid'],   :] = core_pos
    m.geom_pos[h['flame_plume_gid'],  :] = plume_pos

    mujoco.mj_forward(m, d)


# ----- logger -----
def body_z_world(d, bid):
    """World-space +Z axis of the rocket body frame."""
    Rwb = d.xmat[bid].reshape(3, 3)   # world-from-body
    return Rwb[:, 2]                  # body +Z (column 2)

def angle_of_attack_deg(d, bid):
    """
    AoA = angle between body +Z and relative wind (-velocity).
    Returns NaN if speed ~ 0.
    """
    v = d.xvelp[bid].copy()           # world linear velocity of body origin
    speed = np.linalg.norm(v)
    if speed < 1e-8:
        return float("nan")
    rw = -v / speed                   # relative wind
    bz = body_z_world(d, bid)
    c = float(np.clip(np.dot(bz, rw), -1.0, 1.0))
    return math.degrees(math.acos(c))

class DataLogger:
    def __init__(self, folder="data", basename="rocket_log"):
        self.folder = os.path.join(os.path.dirname(__file__), folder)
        os.makedirs(self.folder, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.folder, f"{basename}_{ts}.csv")
        self.fields = [
            "sim_time",
            "x","y","z",
            "qw","qx","qy","qz",
            "vx","vy","vz",
            "wx","wy","wz",
            "mass","fuel_mass",
            "throttle","pitch_deg","yaw_deg",
            "thrust_N",
            "dir_body_x","dir_body_y","dir_body_z",
            "dir_world_x","dir_world_y","dir_world_z",
            "aoa_deg",
        ]
        self.file = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()

    def log(self, d, bid, mass_state, tele: Teleop, thrust_N, dir_body, dir_world, aoa_deg):
        row = dict(
            sim_time=float(d.time),
            x=float(d.xipos[bid,0]), y=float(d.xipos[bid,1]), z=float(d.xipos[bid,2]),
            qw=float(d.xquat[bid,0]), qx=float(d.xquat[bid,1]),
            qy=float(d.xquat[bid,2]), qz=float(d.xquat[bid,3]),
            vx=float(d.xvelp[bid,0]), vy=float(d.xvelp[bid,1]), vz=float(d.xvelp[bid,2]),
            wx=float(d.xvelr[bid,0]), wy=float(d.xvelr[bid,1]), wz=float(d.xvelr[bid,2]),
            mass=float(mass_state["mass"]), fuel_mass=float(mass_state["fuel_mass"]),
            throttle=float(tele.throttle),
            pitch_deg=float(math.degrees(tele.pitch)),
            yaw_deg=float(math.degrees(tele.yaw)),
            thrust_N=float(thrust_N),
            dir_body_x=float(dir_body[0]), dir_body_y=float(dir_body[1]), dir_body_z=float(dir_body[2]),
            dir_world_x=float(dir_world[0]), dir_world_y=float(dir_world[1]), dir_world_z=float(dir_world[2]),
            aoa_deg=float(aoa_deg),
        )
        self.writer.writerow(row)
        # keep data on disk as we go (safer if you crash)
        self.file.flush()

    def close(self):
        try: self.file.close()
        except: pass


class RealTimePlotter:
    """
    Lightweight live plot for mass, thrust, and AoA. Updates every N steps.
    """
    def __init__(self, update_every=5, max_points=600):
        self.enabled = _HAVE_MPL
        if not self.enabled:
            print("[plot] matplotlib not available; live plots disabled.")
            return
        plt.ion()
        self.update_every = int(max(1, update_every))
        self.k = 0

        self.t  = deque(maxlen=max_points)
        self.m  = deque(maxlen=max_points)
        self.T  = deque(maxlen=max_points)
        self.aoa= deque(maxlen=max_points)

        self.fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        self.ax1, self.ax2, self.ax3 = ax1, ax2, ax3

        self.lm,  = ax1.plot([], [], 'tab:blue', lw=2, label='Mass [kg]')
        self.lT,  = ax2.plot([], [], 'tab:orange', lw=2, label='Thrust [N]')
        self.lA,  = ax3.plot([], [], 'tab:green', lw=2, label='AoA [deg]')

        ax1.set_ylabel("Mass [kg]");   ax1.grid(True); ax1.legend(loc="upper right")
        ax2.set_ylabel("Thrust [N]");  ax2.grid(True); ax2.legend(loc="upper right")
        ax3.set_ylabel("AoA [deg]");   ax3.set_xlabel("Sim time [s]"); ax3.grid(True); ax3.legend(loc="upper right")
        self.fig.tight_layout()

    def update(self, t, mass, thrust_N, aoa_deg):
        if not self.enabled:
            return
        self.k += 1
        self.t.append(float(t))
        self.m.append(float(mass))
        self.T.append(float(thrust_N))
        self.aoa.append(float(aoa_deg))

        if self.k % self.update_every != 0:
            return

        self.lm.set_data(self.t, self.m)
        self.lT.set_data(self.t, self.T)
        self.lA.set_data(self.t, self.aoa)

        for ax, y in [(self.ax1, self.m), (self.ax2, self.T), (self.ax3, self.aoa)]:
            if len(self.t) >= 2:
                ax.set_xlim(self.t[0], self.t[-1])
            ymin, ymax = (min(y), max(y)) if len(y) else (0, 1)
            if ymin == ymax:
                ymax = ymin + 1.0
            pad = 0.05*(ymax - ymin) if ymax > ymin else 1.0
            ax.set_ylim(ymin - pad, ymax + pad)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


# ----- Main -----
def main(xml_path: str):
    print("XML ->", os.path.abspath(xml_path))
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    h = handles(m)

    # Mass bookkeeping
    mass_state = {'fuel_mass': FUEL_MASS0, 'mass': DRY_MASS + FUEL_MASS0}
    set_mass_and_inertia(m, h['rocket_bid'], mass_state['mass'])
    mujoco.mj_forward(m, d)

    tele = Teleop()
    def reset_all():
        tele.throttle = 0.0; tele.pitch = tele.yaw = 0.0
        mass_state['fuel_mass'] = FUEL_MASS0
        set_mass_and_inertia(m, h['rocket_bid'], DRY_MASS + FUEL_MASS0)
        mass_state['mass'] = DRY_MASS + FUEL_MASS0
        mujoco.mj_forward(m, d)
        print("[reset] throttle=0, gimbal=0, fuel full.")
    start_keyboard_listener(tele, reset_all)

    # NEW: logger + live plots
    logger = DataLogger(folder="data", basename="rocket_log")
    plotter = RealTimePlotter(update_every=5, max_points=1200)

    try:
        with mujoco.viewer.launch_passive(m, d) as v:
            mujoco.mj_forward(m, d)
            v.cam.lookat[:] = d.xpos[h['rocket_bid']]
            v.cam.distance   = 180.0
            v.cam.azimuth    = 130.0
            v.cam.elevation  = -35.0

            last_print = 0.0
            while v.is_running():
                step_start = time.time()

                dir_body, dir_world, T_cmd = apply_thrust(
                    m, d, h['rocket_bid'], h['engine_sid'], tele, mass_state
                )
                # Update flame visuals to match thrust/gimbal
                update_flame_visual(m, d, h, dir_body, T_cmd)

                mujoco.mj_step(m, d)

                # Compute angle of attack using post-step velocity
                aoa_deg = angle_of_attack_deg(d, h['rocket_bid'])

                # Log one row to CSV
                logger.log(d, h['rocket_bid'], mass_state, tele, T_cmd, dir_body, dir_world, aoa_deg)

                # Update live plots
                plotter.update(d.time, mass_state['mass'], T_cmd, aoa_deg)

                now = time.time()
                if now - last_print > 0.1:
                    print(f"t={d.time:6.2f}s  throttle={tele.throttle:0.2f}  "
                          f"pitch={math.degrees(tele.pitch):+5.1f}°  "
                          f"yaw={math.degrees(tele.yaw):+5.1f}°  "
                          f"T={T_cmd/1000:6.1f} kN  "
                          f"mass={mass_state['mass']:.1f} kg  "
                          f"fuel={mass_state['fuel_mass']:.1f} kg  "
                          f"AoA={aoa_deg:5.1f}°")
                    last_print = now

                v.sync()
                dt = m.opt.timestep - (time.time() - step_start)
                if dt > 0: time.sleep(dt)

            print("Sim closed.")
    finally:
        logger.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_xml = os.path.join(os.path.dirname(__file__), "..", "assets", "mjcf", "world.xml")
    ap.add_argument("--xml", default=os.path.abspath(default_xml), help="Path to MJCF XML")
    args = ap.parse_args()
    main(args.xml)
