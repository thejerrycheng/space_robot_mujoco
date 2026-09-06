"""
RocketGym-planar: the planar (x, z, theta) slice of the 6-DoF RocketGym
environment used in the TINTIN report, re-implemented in pure NumPy so that
the baseline experiments and the browser demo on
https://thejerrycheng.github.io/tintin.html run from one definition of the
dynamics.

Physical parameters follow Table I of the TINTIN report (ROB-GY 7863,
Project 2): a 100 m lunar lander with a single gimbaled engine 30 m below the
centre of mass, wet mass 5.00e6 kg, dry-mass fraction 10 %, Isp = 400 s
(vacuum), lunar gravity 1.62 m/s^2, and an engine sized for a lunar
thrust-to-weight ratio of 3.

State  s = [x, z, vx, vz, theta, omega, m]
Action a = [u_T, u_g] in [-1, 1]^2, mapped to
           T     = T_max (u_T + 1) / 2
           theta_g = 30 deg * u_g            (gimbal deflection from body axis)

Everything below is deliberately dependency-free (NumPy only) so that the
same equations can be transliterated into the JavaScript demo.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Table I — rocket physical and propulsion parameters (lunar environment)
# ----------------------------------------------------------------------------
G_MOON = 1.62                      # m/s^2
G0 = 9.81                          # m/s^2, standard gravity for the Isp law
ISP = 400.0                        # s, vacuum
M_WET = 5.00e6                     # kg
DRY_FRACTION = 0.10
M_DRY = DRY_FRACTION * M_WET       # 5.00e5 kg
M_PROP = M_WET - M_DRY             # 4.50e6 kg
TWR = 3.0                          # lunar thrust-to-weight ratio at wet mass
T_MAX = TWR * M_WET * G_MOON       # 2.43e7 N
L_VEHICLE = 100.0                  # m
L_GIMBAL = 30.0                    # m, engine below the centre of mass
GIMBAL_MAX = np.deg2rad(30.0)
Z_TOUCHDOWN = 55.0                 # m, CoM height at which the episode ends
DT = 0.05                          # s, control step
MAX_STEPS = 2000

# Inertia of a uniform 100 m rod about its centre, scaled with mass:
#   I(t) = I0 * m(t) / m0
I0 = M_WET * L_VEHICLE ** 2 / 12.0

# Termination envelope (report, Sec. II)
MAX_LATERAL = 700.0                # m
MAX_SPEED = 200.0                  # m/s
MAX_TILT = np.deg2rad(100.0)

# Success criteria (report, Sec. II)
SUCCESS_RADIUS = 80.0              # m
SUCCESS_TILT = np.deg2rad(15.0)
SUCCESS_VZ = 20.0                  # m/s, downward magnitude


class RocketGymPlanar:
    """Planar variable-mass lander with a gimbaled engine."""

    def __init__(self, dt=DT, max_steps=MAX_STEPS, seed=None):
        self.dt = dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---------------- deployment ellipse (report, Sec. II) ----------------
    def reset(self, x0=None, z0=None, vx0=None, vz0=None, theta0=None):
        r = self.rng
        self.x = 500.0 + r.uniform(-20, 20) if x0 is None else x0
        self.z = 500.0 + r.uniform(-50, 50) if z0 is None else z0
        speed = r.uniform(0, 50) if (vx0 is None or vz0 is None) else None
        if speed is not None:
            ang = r.uniform(-np.pi, np.pi)
            self.vx = speed * np.cos(ang)
            self.vz = -abs(speed * np.sin(ang))
        else:
            self.vx, self.vz = vx0, vz0
        self.theta = r.uniform(-0.15, 0.15) if theta0 is None else theta0
        self.omega = 0.0
        self.m = M_WET
        self.t = 0.0
        self.steps = 0
        self.prop_used = 0.0
        return self.state()

    def state(self):
        return np.array([self.x, self.z, self.vx, self.vz,
                         self.theta, self.omega, self.m])

    # ------------------------------- dynamics -------------------------------
    def step(self, action):
        """One control step. action = [u_T, u_g] in [-1, 1]."""
        u_T, u_g = float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1))
        T = T_MAX * (u_T + 1.0) / 2.0
        gimbal = GIMBAL_MAX * u_g

        # No propellant -> no thrust.
        if self.m <= M_DRY:
            T = 0.0
            self.m = M_DRY

        # Thrust direction in the world frame. theta is the tilt of the body
        # axis from vertical; the nozzle adds `gimbal` on top of it.
        phi = self.theta + gimbal
        fx = T * np.sin(phi)
        fz = T * np.cos(phi)

        ax = fx / self.m
        az = fz / self.m - G_MOON

        # Torque about the CoM from the offset, gimbaled nozzle.
        inertia = I0 * self.m / M_WET
        torque = -T * np.sin(gimbal) * L_GIMBAL
        alpha = torque / inertia

        dt = self.dt
        self.vx += ax * dt
        self.vz += az * dt
        self.x += self.vx * dt
        self.z += self.vz * dt
        self.omega += alpha * dt
        self.theta += self.omega * dt

        # Mass depletion: mdot = -T / (Isp g0)
        mdot = T / (ISP * G0)
        burn = min(mdot * dt, max(self.m - M_DRY, 0.0))
        self.m -= burn
        self.prop_used += burn

        self.t += dt
        self.steps += 1
        return self.state(), *self._terminal(T)

    def _terminal(self, T):
        done, outcome = False, None
        speed = np.hypot(self.vx, self.vz)
        if self.z <= Z_TOUCHDOWN:
            done = True
            landed = (abs(self.x) <= SUCCESS_RADIUS
                      and abs(self.theta) <= SUCCESS_TILT
                      and 0.0 <= -self.vz <= SUCCESS_VZ)
            outcome = "success" if landed else self._why_hard()
        elif abs(self.x) > MAX_LATERAL:
            done, outcome = True, "drift"
        elif speed > MAX_SPEED:
            done, outcome = True, "overspeed"
        elif abs(self.theta) > MAX_TILT:
            done, outcome = True, "tilt"
        elif self.steps >= self.max_steps:
            done, outcome = True, "timeout"
        return done, outcome, {"thrust": T, "speed": speed}

    def _why_hard(self):
        """Classify a touchdown that missed the success box."""
        if abs(self.theta) > SUCCESS_TILT:
            return "tilt"
        if -self.vz > SUCCESS_VZ:
            return "impact"
        return "drift"


# ----------------------------------------------------------------------------
# PID / PD baseline (report, Sec. III)
# ----------------------------------------------------------------------------
DEFAULT_GAINS = dict(kp_x=0.004, kd_x=0.10, kd_z=0.70,
                     kp_att=0.16, kd_att=0.72, glide=0.065)


class PIDBaseline:
    """The report's cascaded baseline (Sec. III), written at report scale.

    Outer translational loop (accelerations):
        a_x^des = kp_x (0 - x) + kd_x (0 - vx)
        a_z^des = kd_z (vz_ref(h) - vz),   vz_ref = -clip(glide * h, 1, 40)
    Thrust vector:
        T_des   = m ||[a_x^des, a_z^des + g]||
        phi_des = atan2(a_x^des, a_z^des + g)          (tilt from vertical)
    Inner attitude loop.  In this model the nozzle torque is
    tau = -T sin(theta_g) L_gimbal, so a positive gimbal deflection rotates the
    body the *negative* way; the PD output is inverted accordingly and
    linearised through the current thrust and inertia:
        alpha^des = kp_att (phi_des - theta) - kd_att omega
        theta_g   = -asin(clip(alpha^des I(m) / (T L_gimbal), -1, 1))
    """

    def __init__(self, **gains):
        self.g = dict(DEFAULT_GAINS)
        self.g.update(gains)

    def __call__(self, s):
        x, z, vx, vz, theta, omega, m = s
        g = self.g

        # Glideslope reference: descend fast when high, flare near the pad.
        h = max(z - Z_TOUCHDOWN, 0.0)
        vz_ref = -np.clip(g["glide"] * h, 1.0, 40.0)

        ax_des = g["kp_x"] * (0.0 - x) + g["kd_x"] * (0.0 - vx)
        az_des = g["kd_z"] * (vz_ref - vz)

        # The engine can only push along +body-z, so the commanded vertical
        # acceleration is floored: free fall is the cheapest way down.
        ax_t = ax_des
        az_t = max(az_des + G_MOON, 0.0)
        T_des = m * float(np.hypot(ax_t, az_t))
        T_des = float(np.clip(T_des, 0.0, T_MAX))
        phi_des = float(np.arctan2(ax_t, max(az_t, 0.2 * G_MOON)))
        phi_des = float(np.clip(phi_des, -np.deg2rad(30), np.deg2rad(30)))

        alpha_des = g["kp_att"] * (phi_des - theta) - g["kd_att"] * omega
        inertia = I0 * m / M_WET
        lever = max(T_des, 0.05 * T_MAX) * L_GIMBAL
        theta_g = -float(np.arcsin(np.clip(alpha_des * inertia / lever, -1.0, 1.0)))

        u_T = float(np.clip(2.0 * T_des / T_MAX - 1.0, -1.0, 1.0))
        u_g = float(np.clip(theta_g / GIMBAL_MAX, -1.0, 1.0))
        return np.array([u_T, u_g])


def rollout(env, controller, x0=None, z0=None, vx0=None, vz0=None, theta0=None,
            record=False):
    """Run one episode; return the outcome dict (and the trace if requested)."""
    env.reset(x0=x0, z0=z0, vx0=vx0, vz0=vz0, theta0=theta0)
    trace = [] if record else None
    outcome, info = None, {}
    while True:
        s = env.state()
        a = controller(s)
        if record:
            trace.append(np.concatenate([[env.t], s, a]))
        s, done, outcome, info = env.step(a)
        if done:
            break
    res = dict(outcome=outcome, t=env.t, x=env.x, z=env.z,
               vx=env.vx, vz=env.vz, tilt=np.rad2deg(env.theta),
               speed=float(np.hypot(env.vx, env.vz)),
               prop_used=env.prop_used,
               prop_frac=env.prop_used / M_PROP,
               radius=abs(env.x))
    if record:
        res["trace"] = np.array(trace)
    return res
