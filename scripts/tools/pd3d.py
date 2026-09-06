#!/usr/bin/env python3
"""
A 3-D cascaded PD landing controller that flies the real MuJoCo vehicle.

This replaces the planar PD study: the controller here closes the loop on the
full 6-DoF state of rocket_env/rocket_dr_env.py (the MJCF vehicle, RK4 at
100 Hz, variable mass, 2-DoF gimbal), not on a reduced planar model.

Structure
---------
Outer (translational) loop, in the world frame:

    a_des = Kp (p* - p) + Kd (v*(h) - v),      v*_z(h) = -clip(glide h, 0.6, 22)
    f_des = m (a_des + g e_z)                   the thrust vector we want

Attitude reference:  the vehicle's body z-axis must point along f_des.  With
u_b = R^T f_des / |f_des| the body-frame error vector is

    e = [0,0,1] x u_b = (-u_b_y,  u_b_x,  0)

Inner (attitude) loop:

    alpha_des = Kp_att e - Kd_att omega_body
    tau_des   = I(m) alpha_des

Gimbal inversion.  The engine sits at r = (0, 0, -L) in the body frame with
L = 30 m, and the two hinges give a thrust direction

    F_body = T ( sin(g_p),  -sin(g_y) cos(g_p),  cos(g_y) cos(g_p) )

so the torque it produces is tau = r x F = ( L T sin(g_y) cos(g_p) ... ), i.e.

    tau_x = -L T sin(g_y) cos(g_p)
    tau_y = -L T sin(g_p)
    tau_z = 0

which inverts to

    g_p = asin( -tau_y_des / (L T) )
    g_y = asin( -tau_x_des / (L T cos g_p) )

Yaw about the vehicle's own axis is unactuated, exactly as in the report.
"""

import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.modules.setdefault("tensorflow", None)

from rocket_env.rocket_dr_env import RocketLandingDREnv          # noqa: E402

L_GIMBAL = 30.0
G_MOON = 1.62

# Gains are sized to the authority the vehicle actually has.  At TWR 3 and a
# 25 deg tilt cap the lateral acceleration available is 3 * 1.62 * sin(25 deg)
# ~ 2.0 m/s^2, and the gimbal can produce at most
# tau = L T sin(30 deg) = 3.75e8 N m against I = 1.2e9 kg m^2, i.e. 0.31 rad/s^2.
# Gains above those numbers just saturate both loops and turn the cascade into a
# bang-bang oscillator, which is what a naively-tuned PD does here.
DEFAULT_GAINS = dict(kp_xy=0.048, kd_xy=0.95, kd_z=0.90, glide=0.55,
                     t_lag=3.0, v_lat_max=16.0, ki_xy=0.004, i_lim=0.45,
                     a_up_frac=0.55, flare=6.0, vz_max=26.0, touch_vz=1.2,
                     lat_hold=110.0,
                     kp_att=1.1, kd_att=2.41, tilt_max_deg=35.0,
                     # Lateral authority is set by the tilt cap, not by TWR.
                     # During a near-hover the thrust vector is about m*g, so
                     # a_lat <= g tan(theta_max): 0.65 m/s^2 at 22 deg and
                     # 1.13 m/s^2 at 35 deg.  The first version of this
                     # controller assumed 2.0 m/s^2, commanded approach speeds
                     # it could not arrest, and flew through the pad in a limit
                     # cycle it never left.
                     a_lat_max=1.1,
                     # The tilt cap has to sit BELOW the fin-strike boundary,
                     # not on it.  With theta <= 4 (h - 1.5) — the boundary
                     # itself — the vehicle simply arrives at whatever tilt the
                     # cap allows, which is by construction the largest tilt
                     # that still counts as contact rather than a strike.
                     contact_slope=2.0, contact_z0=2.0,
                     thrust_floor=1.0,
                     gate_alt=55.0, gate_lateral=8.0, gate_hspeed=0.8,
                     gate_tilt_deg=8.0)


def env_twr(env):
    """Thrust-to-weight the vehicle actually has right now."""
    return env.max_thrust / (env.model.body_mass[env.rocket_bid] * G_MOON)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PD3D:
    def __init__(self, env, **gains):
        self.env = env
        self.g = dict(DEFAULT_GAINS)
        self.g.update(gains)
        self.reset()

    def reset(self):
        # Integral state on the lateral channel.  A constant disturbance needs a
        # permanent tilt to cancel, and a pure PD has no way to hold one: it
        # settles at whatever offset makes its proportional term match.
        self.i_xy = np.zeros(2)

    def __call__(self, env):
        g = self.g
        m = env._metrics()
        mass = env.model.body_mass[env.rocket_bid]
        I = env.model.body_inertia[env.rocket_bid]
        p, v, q, om = m["pos"], m["vel"], m["quat"], m["omega"]
        alt = max(m["alt"], 0.0)

        # ---- outer loop -----------------------------------------------------
        # Hold altitude back while there is still downrange to kill: the
        # lateral channel is the slow one, and a descent that beats it to the
        # ground is the dominant failure of a fixed-gain cascade here.
        # Vertical guidance by stopping distance, the same shape the lateral
        # channel uses.  With TWR 3 the vehicle can decelerate upward at about
        # 3*g_moon - g_moon = 3.2 m/s^2; commanding
        #     v_z = -sqrt(2 * frac * a_up * (h - flare))
        # means the descent rate is always one the remaining altitude can
        # arrest.  Two earlier laws failed here: a fixed exponential glideslope
        # ran the 30 s clock out, and tying altitude to remaining downrange
        # parked the vehicle in a hover at the altitude where the two balanced.
        a_up = g["a_up_frac"] * (env_twr(env) * env.model.opt.gravity[2] * -1.0
                                 - -env.model.opt.gravity[2])
        h = max(alt - g["flare"], 0.0)
        # descend more gently while there is still a long way to go sideways
        lat_hold = np.clip(1.0 - m["lateral"] / g["lat_hold"], 0.30, 1.0)
        vz_ref = -min(np.sqrt(2.0 * a_up * h) * lat_hold, g["vz_max"])
        if alt > g["flare"]:
            vz_ref = min(vz_ref, -0.5)
        else:
            vz_ref = -g["touch_vz"]

        # Terminal gate.  The vehicle is 100 m long, so a tilted or still-
        # translating lander puts a fin into the ground with its centre of mass
        # metres up.  Below the gate altitude the controller refuses to descend
        # until it is centred, upright and no longer moving sideways.
        hspeed = float(np.linalg.norm(v[:2]))
        tilt_deg = m["tilt_deg"]
        composed = (m["lateral"] < g["gate_lateral"] and hspeed < g["gate_hspeed"]
                    and tilt_deg < g["gate_tilt_deg"])
        if alt < g["gate_alt"] and not composed:
            # Hold, do not climb: climbing wastes the clock and the propellant.
            vz_ref = max(vz_ref, -0.2)

        # Lateral guidance is velocity-shaped rather than a plain position PD:
        # with only ~2 m/s^2 of lateral authority a proportional term saturates
        # for the first ten seconds and then decays far too slowly to make the
        # descent.  Commanding a speed that the remaining distance can actually
        # arrest is the same trick the vertical channel already uses.
        d_xy = p[:2]
        dist = float(np.linalg.norm(d_xy))
        if dist > 1e-6:
            # Stopping distance with an attitude-lag allowance.  Braking
            # laterally means reversing the whole vehicle's tilt, which takes
            # about t_lag seconds during which there is no deceleration at all,
            # so the approach speed has to satisfy
            #     dist = v^2 / (2 a) + v t_lag
            # rather than the textbook v = sqrt(2 a d).  Ignoring the lag is
            # what made the earlier version fly through the pad at 5 m/s.
            a = g["a_lat_max"]
            tl = g["t_lag"]
            v_mag = min(a * (-tl + np.sqrt(tl * tl + 2.0 * dist / a)), g["v_lat_max"])
            v_ref_xy = -d_xy / dist * v_mag
        else:
            v_ref_xy = np.zeros(2)
        a_lat = g["kd_xy"] * (v_ref_xy - v[:2])
        self.i_xy = np.clip(self.i_xy + g["ki_xy"] * (-p[:2]) * env.DT * env.FRAME_SKIP,
                            -g["i_lim"], g["i_lim"])
        a_lat = a_lat + self.i_xy
        n = np.linalg.norm(a_lat)
        if n > g["a_lat_max"]:
            a_lat *= g["a_lat_max"] / n
        a_des = np.array([a_lat[0], a_lat[1], g["kd_z"] * (vz_ref - v[2])])

        f_des = mass * (a_des + np.array([0.0, 0.0, G_MOON]))
        if f_des[2] < 0.2 * mass * G_MOON:
            f_des[2] = 0.2 * mass * G_MOON          # never command thrust downward

        # You cannot steer while falling.  Lateral acceleration is
        # (a_z + g) tan(theta): if the vertical channel throttles back to a
        # near-free-fall descent then a_z + g -> 0 and the gimbal buys almost no
        # lateral authority at any tilt.  While there is still downrange to
        # kill, hold at least hover thrust so the tilt is worth something.
        if m["lateral"] > g["gate_lateral"]:
            f_des[2] = max(f_des[2], g["thrust_floor"] * mass * G_MOON)

        # NOTE.  Raising this floor unconditionally to give the gimbal authority
        # on a straight-down descent was tried and made things worse: it slows
        # the descent enough that the vehicle runs the clock out. The attitude
        # limit cycle documented on the website is therefore a real property of
        # this cascade on this vehicle, not a tuning oversight — the gimbal's
        # authority scales with the very thrust an efficient descent minimises.

        # Cap the commanded tilt by ALTITUDE, not just by a constant.  Measured
        # on this model by bisecting the resting height at a range of tilts, the
        # centre of mass touches down at
        #     z_contact = 1.00 + 0.2475 * tilt[deg]        (R^2 > 0.999)
        # because the vehicle is 100 m long with a ~17 m fin radius, so a tilted
        # lander puts a fin into the ground with its centre of mass ten metres
        # up.  Any tilt above 4 (alt - 1.5) degrees is therefore a guaranteed
        # fin strike, and a cascade that ignores this flies a textbook approach
        # straight into the ground at 16 deg and 4 m/s.
        tilt_cap = min(g["tilt_max_deg"],
                       max(0.0, g["contact_slope"] * (alt - g["contact_z0"])))
        tmax = np.deg2rad(tilt_cap)
        horiz = np.linalg.norm(f_des[:2])
        if horiz > f_des[2] * np.tan(tmax):
            f_des[:2] *= f_des[2] * np.tan(tmax) / max(horiz, 1e-9)

        T_des = float(np.linalg.norm(f_des))
        T_des = float(np.clip(T_des, 0.0, env.max_thrust))
        u_des = f_des / (np.linalg.norm(f_des) + 1e-9)

        # ---- attitude error in the body frame -------------------------------
        R = quat_to_mat(q)
        u_b = R.T @ u_des
        e = np.array([-u_b[1], u_b[0], 0.0])

        alpha = g["kp_att"] * e - g["kd_att"] * om
        tau = I * alpha

        # ---- gimbal inversion ------------------------------------------------
        lever = max(T_des, 0.04 * env.max_thrust) * L_GIMBAL
        sp = np.clip(-tau[1] / lever, -1.0, 1.0)
        gp = np.arcsin(sp)
        sy = np.clip(-tau[0] / (lever * max(np.cos(gp), 0.3)), -1.0, 1.0)
        gy = np.arcsin(sy)

        return np.array([
            np.clip(2.0 * T_des / env.max_thrust - 1.0, -1.0, 1.0),
            np.clip(gy / env.MAX_GIMBAL, -1.0, 1.0),
            np.clip(gp / env.MAX_GIMBAL, -1.0, 1.0)])


def rollout(env, ctrl, options=None, record=False, seed=None):
    env.reset(seed=seed, options=options)
    trace = [] if record else None
    info = {}
    while True:
        a = ctrl(env)
        if record:
            m = env._metrics()
            trace.append(np.concatenate([[env.step_count * env.DT * env.FRAME_SKIP], m["pos"], m["vel"],
                                         m["quat"], m["omega"], a,
                                         [env.fuel_mass / env.start_fuel]]))
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    m = env._metrics()
    res = dict(outcome=info.get("outcome"), success=bool(info.get("success")),
               t=env.step_count * env.DT * env.FRAME_SKIP, lateral=m["lateral"], speed=m["speed"],
               tilt_deg=m["tilt_deg"],
               vz=m["vz"], alt=m["alt"],
               fuel_frac=env.fuel_mass / env.start_fuel)
    if record:
        res["trace"] = np.array(trace)
    return res


def monte_carlo(n, seed=0, gains=None, dr=True, curriculum=False, record=0, options=None):
    env = RocketLandingDREnv(seed=seed, domain_randomize=dr, curriculum=curriculum)
    ctrl = PD3D(env, **(gains or {}))
    rows, traces = [], []
    for i in range(n):
        r = rollout(env, ctrl, options=options, record=(i < record))
        if "trace" in r:
            traces.append(r.pop("trace"))
        rows.append(r)
    return rows, traces


def summarise(rows):
    c = collections.Counter(r["outcome"] for r in rows)
    ok = [r for r in rows if r["success"]]
    def st(key, src=ok):
        v = np.array([abs(r[key]) for r in src]) if src else np.array([np.nan])
        return dict(median=float(np.nanmedian(v)), p95=float(np.nanpercentile(v, 95)),
                    max=float(np.nanmax(v)))
    return dict(trials=len(rows), success=len(ok), rate=len(ok) / max(len(rows), 1),
                outcomes={k: v for k, v in c.most_common()},
                lateral=st("lateral"), speed=st("speed"), tilt=st("tilt_deg"),
                fuel_used=dict(median=float(np.median([1 - r["fuel_frac"] for r in ok])) if ok else np.nan),
                flight=st("t"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--no-dr", action="store_true")
    a = ap.parse_args()
    rows, _ = monte_carlo(a.trials, dr=not a.no_dr)
    print(json.dumps(summarise(rows), indent=2))
