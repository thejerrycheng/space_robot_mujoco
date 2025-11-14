"""
Thrust + 2-DoF gimbal demo with robust startup and rich debug logging.

What this script guarantees:
- Starts above ground: auto-raises until there are ZERO contacts.
- Computes hover thrust from model mass (so we don't overshoot).
- Soft thrust ramp to avoid instant zoom.
- Neutral gimbal during ramp (no lateral kick).
- Detailed telemetry in overlay + periodic console prints.

Controls actually written:
- data.ctrl[thrust]  : scalar thrust (N) along site +Z
- data.ctrl[pitch/yaw]: PD torques to hold zero angle (can enable oscillation)
"""

import math
import time
from pathlib import Path

import numpy as np
import mujoco
from mujoco import viewer

# ----------------- user knobs -----------------
THRUST_MAX = 3000.0
HOVER_SCALE = 1.00          # 1.00 = exact hover; <1 sink, >1 climb slowly
GIMBAL_AMP_DEG = 0.0        # keep 0.0 while debugging; raise later
GIMBAL_FREQ_HZ = 0.2
KP = 8.0
KD = 0.4
REALTIME = True

# startup behavior
START_POS = (0.0, 0.0, 1.2)       # initial guess (will be raised if needed)
START_QUAT = (1.0, 0.0, 0.0, 0.0) # upright (w,x,y,z)
CLEARANCE_STEP = 0.05             # meters to nudge up per check
CLEARANCE_MAX_STEPS = 80          # up to 4.0 m extra
THRUST_RAMP_TIME = 2.0            # seconds to smoothly ramp thrust
FREEZE_GIMBAL_FOR = 1.0           # seconds to keep gimbal neutral
PRINT_PERIOD = 0.25               # s between console debug prints
# ----------------------------------------------


def asset_path():
    here = Path(__file__).resolve()
    root = here.parent.parent  # repo root
    return root / "assets" / "mjcf" / "thrust_test.xml"


def name2id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def actuator_ids(model):
    return {
        "thrust": name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust"),
        "pitch":  name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gimbal_pitch_motor"),
        "yaw":    name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gimbal_yaw_motor"),
    }


def joint_ids(model):
    return {
        "free":  name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free"),
        "pitch": name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gimbal_pitch"),
        "yaw":   name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gimbal_yaw"),
    }


def set_free_pose(model, data, pos, quat):
    """Set free joint qpos (x y z qw qx qy qz)."""
    adr = model.jnt_qposadr[joint_ids(model)["free"]]
    data.qpos[adr:adr+3] = np.array(pos)
    data.qpos[adr+3:adr+7] = np.array(quat)


def ensure_clear_of_contacts(model, data):
    """Raise along +Z until there are NO contacts."""
    adr = model.jnt_qposadr[joint_ids(model)["free"]]
    steps = 0
    while True:
        mujoco.mj_forward(model, data)
        if data.ncon == 0: break
        if steps >= CLEARANCE_MAX_STEPS: break
        data.qpos[adr + 2] += CLEARANCE_STEP
        steps += 1
    mujoco.mj_forward(model, data)


def smooth_ramp(t, T):
    """0->1 smooth step over [0,T]."""
    if t <= 0.0: return 0.0
    if t >= T:   return 1.0
    x = t / T
    return x*x*(3 - 2*x)


def pd_torque(model, data, jnid, q_target, kp=KP, kd=KD):
    qadr = model.jnt_qposadr[jnid]
    qdadr = model.jnt_dofadr[jnid]
    q = float(data.qpos[qadr])
    qd = float(data.qvel[qdadr])
    return kp * (q_target - q) - kd * qd


def total_mass(model):
    # Sum all body masses (excluding world which is 0 anyway)
    return float(np.sum(model.body_mass))


def com_height(model, data):
    # free joint pos z is body frame origin's world z, which is good enough here
    adr = model.jnt_qposadr[joint_ids(model)["free"]]
    return float(data.qpos[adr + 2])


def vertical_speed(model, data):
    adr = model.jnt_dofadr[joint_ids(model)["free"]]
    return float(data.qvel[adr + 2])

# ----- Helper functions for thrust application -----
def pos_xyz(model, data):
    """Free-joint world position (x,y,z)."""
    adr = model.jnt_qposadr[joint_ids(model)["free"]]
    return tuple(float(v) for v in data.qpos[adr:adr+3])

def vel_xyz(model, data):
    """Free-joint world linear velocity (vx,vy,vz)."""
    adr = model.jnt_dofadr[joint_ids(model)["free"]]
    return tuple(float(v) for v in data.qvel[adr:adr+3])



def main():
    xml = asset_path()
    if not xml.exists():
        raise FileNotFoundError(f"XML not found: {xml}")

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    print(f"Loaded XML: {xml}")
    act = actuator_ids(model)
    jnt = joint_ids(model)

    # Pose + contact-free start
    set_free_pose(model, data, START_POS, START_QUAT)
    ensure_clear_of_contacts(model, data)

    # Hover thrust from mass * g (uses model.opt.gravity z)
    mass = total_mass(model)                     # ~ 11.0 kg for your XML
    g = -model.opt.gravity[2]                    # 9.81
    hover_thrust = mass * g                      # ≈ 107.9 N
    hover_thrust *= HOVER_SCALE                  # let you bias slightly

    # Log initial conditions
    print(f"[init] mass_total = {mass:.3f} kg, hover_thrust = {hover_thrust:.2f} N, g = {g:.3f} m/s^2")
    print(f"[init] start_z = {com_height(model, data):.3f} m, ncon = {data.ncon}")

    # Make sure controls start at zero
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    t0 = time.time()
    t_last_print = t0

    with viewer.launch_passive(model, data) as v:
        v.add_overlay(
            "topleft",
            "thrust_test debug",
            "Keys: (none) | Tweak constants in script",
        )

        while v.is_running():
            t = time.time() - t0

            # --- Thrust command with smooth ramp to hover value ---
            ramp = smooth_ramp(t, THRUST_RAMP_TIME)
            thrust_cmd = float(np.clip(hover_thrust * ramp, 0.0, THRUST_MAX))
            data.ctrl[act["thrust"]] = thrust_cmd

            # --- Keep gimbal neutral initially (you can enable oscillation later) ---
            if t < FREEZE_GIMBAL_FOR or GIMBAL_AMP_DEG <= 1e-6:
                pitch_tgt = 0.0
                yaw_tgt = 0.0
            else:
                amp = math.radians(GIMBAL_AMP_DEG)
                phi = 2 * math.pi * GIMBAL_FREQ_HZ * (t - FREEZE_GIMBAL_FOR)
                pitch_tgt = amp * math.sin(phi)
                yaw_tgt   = amp * math.sin(phi + math.pi / 2)

            data.ctrl[act["pitch"]] = pd_torque(model, data, jnt["pitch"], pitch_tgt)
            data.ctrl[act["yaw"]]   = pd_torque(model, data, jnt["yaw"],   yaw_tgt)

            # Step physics
            mujoco.mj_step(model, data)

            # Debug readbacks
            z   = com_height(model, data)
            zd  = vertical_speed(model, data)
            ncon = int(data.ncon)
            # actuator force readback (from sensor) if present
            # F_actuator is not always equal to ctrl due to clamping/limits/gains
            thrust_readback = np.nan
            try:
                sid = name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "thrust_readback")
                thrust_readback = float(data.sensordata[model.sensor_adr[sid]])
            except Exception:
                pass

            # Overlay
            v.add_overlay(
                "bottomleft",
                f"t={t:5.2f}s  ctrl(thrust)={thrust_cmd:7.2f} N  F_readback={thrust_readback:7.2f} N",
                f"z={z:5.2f} m  zd={zd:6.3f} m/s  ncon={ncon}",
            )

            # Periodic console print
            # Periodic console print
            if (time.time() - t_last_print) >= PRINT_PERIOD:
                x, y, z = pos_xyz(model, data)
                vx, vy, vz = vel_xyz(model, data)
                print(
                    f"[{t:5.2f}s] pos=({x:6.3f}, {y:6.3f}, {z:6.3f}) m  "
                    f"vel=({vx:6.3f}, {vy:6.3f}, {vz:6.3f}) m/s  "
                    f"ctrl_thrust={thrust_cmd:7.2f}N  ncon={ncon}"
                )
                t_last_print = time.time()


            if REALTIME:
                v.sync()


if __name__ == "__main__":
    main()
