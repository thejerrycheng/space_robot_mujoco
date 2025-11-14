#!/usr/bin/env mjpython
# Run: mjpython scripts/new_run_viewer.py
import os, math, time, numpy as np, mujoco, mujoco.viewer
from pynput import keyboard

# ---------- Params ----------
DT           = 0.01
G0           = 9.80665
ISP          = 320.0
DRY_MASS     = 1.0e5        # kg
FUEL_MASS0   = 4.0e5        # kg
THRUST_XWT   = 10.0         # max thrust ~ 10x weight

THROTTLE_STEP = 0.02
GIMBAL_STEP   = math.radians(1.0)
GIMBAL_LIM    = math.radians(70.0)

N = dict(nozzle_site="nozzle_site", yaw="yaw", pitch="pitch",
         rocket_body="rocket_root", flame_core="flame_core", flame_plume="flame_plume")

FLAME_CORE_BASE  = np.array([1.2, 1.2, 6.0], dtype=float)   # ellipsoid radii
FLAME_PLUME_BASE = np.array([1.8, 10.0], dtype=float)       # cylinder [radius, half-length]

class Tele:
    def __init__(self):
        self.throttle = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.quit = False
        self.reset = False
    def clamp(self):
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        self.yaw   = float(np.clip(self.yaw,   -GIMBAL_LIM, GIMBAL_LIM))
        self.pitch = float(np.clip(self.pitch, -GIMBAL_LIM, GIMBAL_LIM))

def attach_keys(t):
    def on_press(key):
        try:
            if key == keyboard.Key.up:    t.throttle += THROTTLE_STEP
            elif key == keyboard.Key.down:t.throttle -= THROTTLE_STEP
            elif key == keyboard.Key.left:t.yaw += GIMBAL_STEP
            elif key == keyboard.Key.right:t.yaw -= GIMBAL_STEP
            elif key.char.lower() == 'w': t.pitch += GIMBAL_STEP
            elif key.char.lower() == 's': t.pitch -= GIMBAL_STEP
            elif key.char.lower() == 'r': t.reset = True
            elif key.char.lower() == 'q': t.quit = True
        except AttributeError:
            if key == keyboard.Key.esc: t.quit = True
        t.clamp()
    lst = keyboard.Listener(on_press=on_press); lst.daemon = True; lst.start(); return lst

def col_from_xmat(xmat, col):
    return np.array([xmat[0*3+col], xmat[1*3+col], xmat[2*3+col]], dtype=float)

def main(xml_path):
    # Debug: print/verify the file you’re about to load
    xml_abs = os.path.abspath(xml_path)
    print(f"[INFO] Loading XML: {xml_abs}")
    try:
        with open(xml_abs, "r") as f:
            head = "".join([next(f) for _ in range(12)])
        print("[INFO] First lines of XML:\n" + head)
    except Exception as e:
        print(f"[WARN] Could not preview XML: {e}")

    model = mujoco.MjModel.from_xml_path(xml_abs)
    data  = mujoco.MjData(model)
    model.opt.timestep = DT

    # Lookups
    sid_noz = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,  N["nozzle_site"])
    jid_yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, N["yaw"])
    jid_pit = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, N["pitch"])
    bid_rkt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  N["rocket_body"])
    gid_core= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,  N["flame_core"])
    gid_plu = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,  N["flame_plume"])

    # Correct addresses for joint qpos
    adr_yaw = model.jnt_qposadr[jid_yaw]
    adr_pit = model.jnt_qposadr[jid_pit]

    # Cache base mass/inertia for scaling
    base_mass0    = float(model.body_mass[bid_rkt])
    base_inertia0 = model.body_inertia[bid_rkt].copy()

    # Initialize mass to dry+fuel
    total0 = DRY_MASS + FUEL_MASS0
    model.body_mass[bid_rkt]    = total0
    model.body_inertia[bid_rkt] = base_inertia0 * (total0 / base_mass0)
    mujoco.mj_forward(model, data)

    # Start pose (free joint: xyz + quat(wxyz))
    data.qpos[:] = 0.0; data.qvel[:] = 0.0
    data.qpos[0:3] = np.array([0.0, 0.0, 150.0])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    tele = Tele(); attach_keys(tele)
    fuel = FUEL_MASS0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_last = time.time(); t_print = t_last
        while viewer.is_running() and not tele.quit:
            now = time.time()
            while now - t_last >= DT:
                t_last += DT

                if tele.reset:
                    data.qpos[:] = 0.0; data.qvel[:] = 0.0
                    data.qpos[0:3] = np.array([0.0, 0.0, 150.0])
                    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
                    tele.throttle = 0.0; tele.yaw = 0.0; tele.pitch = 0.0
                    fuel = FUEL_MASS0
                    total = DRY_MASS + fuel
                    model.body_mass[bid_rkt]    = total
                    model.body_inertia[bid_rkt] = base_inertia0 * (total / base_mass0)
                    mujoco.mj_forward(model, data)
                    tele.reset = False

                # Write gimbal joints
                data.qpos[adr_yaw] = tele.yaw
                data.qpos[adr_pit] = tele.pitch

                # Thrust & fuel
                total = DRY_MASS + fuel
                weight = total * G0
                thrust_max = THRUST_XWT * weight
                thrust = tele.throttle * thrust_max
                mdot = thrust / (ISP * G0) if fuel > 0 else 0.0
                fuel = max(0.0, fuel - mdot * DT)

                # Update mass/inertia
                total = DRY_MASS + fuel
                model.body_mass[bid_rkt]    = total
                model.body_inertia[bid_rkt] = base_inertia0 * (total / base_mass0)
                mujoco.mj_forward(model, data)

                # Thrust direction = -Z axis of nozzle (world frame)
                z_world = col_from_xmat(data.site_xmat[sid_noz], 2)
                dir_world = -z_world / (np.linalg.norm(z_world) + 1e-12)
                data.xfrc_applied[bid_rkt, 0:3] = thrust * dir_world
                data.xfrc_applied[bid_rkt, 3:6] = 0.0

                # Flame scale
                s = max(0.0, float(tele.throttle))
                model.geom_size[gid_core, 0:3] = FLAME_CORE_BASE * (0.3 + 1.0 * s)
                model.geom_size[gid_plu, 0]    = FLAME_PLUME_BASE[0] * (0.3 + 1.5 * s)
                model.geom_size[gid_plu, 1]    = FLAME_PLUME_BASE[1] * (0.3 + 1.5 * s)

                mujoco.mj_step(model, data)

            if now - t_print > 0.25:
                print(f"thr={tele.throttle:0.2f}  yaw={math.degrees(tele.yaw):5.1f}°  "
                      f"pitch={math.degrees(tele.pitch):5.1f}°  "
                      f"mass={total:,.0f} kg  thrust={thrust/1000:,.0f} kN")
                t_print = now
            viewer.sync()

if __name__ == "__main__":
    xml = "assets/mjcf/world.xml"
    main(xml)
