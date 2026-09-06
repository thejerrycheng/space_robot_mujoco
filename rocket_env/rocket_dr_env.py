"""
RocketGym-DR — the 6-DoF MuJoCo lunar-landing environment with wide domain
randomisation.

Derived from rocket_env/rocket_3_env.py and the same MJCF
(assets/mjcf/realistic_param.xml): a 5,000-tonne, 100 m vehicle with a 2-DoF
gimbaled engine 30 m below the centre of mass, lunar gravity, RK4 at 100 Hz.

Two things are fixed relative to rocket_3_env:

1. TARGET_Z.  rocket_3_env was written against an older MJCF whose moon terrain
   sat at z = -56 and used TARGET_Z = -16.  With the current model the ground
   plane is at z = 0 and the vehicle comes to rest with its centre of mass at
   z = 0.995 m (measured by dropping it), so the touchdown reference is 1.0 m.
   With the old value the success test could never fire.
2. Domain randomisation.  rocket_3_env randomised only the initial pose,
   velocity and tilt.  Here every episode also resamples the vehicle and the
   engine — dry mass, propellant load, specific impulse, thrust authority,
   a persistent gimbal misalignment, a constant lateral acceleration
   disturbance, and observation noise — so a policy has to be robust to a
   vehicle it has not flown before, not just to a starting state it has not
   seen.

The curriculum from rocket_3_env is kept: the initial-condition envelope widens
as the policy succeeds, but the vehicle randomisation is always on.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "realistic_param.xml")

# Measured: drop the vehicle in this MJCF and its centre of mass settles here.
REST_Z = 0.995


class RocketLandingDREnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    # ---- nominal vehicle (MJCF + rocket_3_env) ----------------------------
    DRY_MASS = 5_000_000.0
    START_FUEL = 1_000_000.0
    ISP = 250.0
    G0 = 1.62
    MAX_THRUST = 25_000_000.0
    MAX_GIMBAL = np.deg2rad(30.0)

    TARGET_Z = 1.0
    # The policy acts at 20 Hz while MuJoCo integrates at 100 Hz.  Without the
    # frame skip an episode is 4,500 decisions long, and at any usable discount
    # factor the landing bonus at the end is invisible: gamma = 0.985 discounts
    # it by 1e-30.  Holding each action for five physics steps makes an episode
    # 900 decisions, which a discount of 0.995 can actually see across, and it
    # makes training five times faster into the bargain.
    FRAME_SKIP = 5
    MAX_STEPS = 900                  # 45 s at dt = 0.05 control step
    MAX_LATERAL_DIST = 150.0
    MAX_VELOCITY = 100.0
    LANDING_TOLERANCE = 10.0         # m, radius of the accepted pad
    LANDING_SPEED = 3.0              # m/s
    LANDING_TILT = 0.10              # 1 - |q_w|, about 25 deg

    # ---- domain randomisation ranges ---------------------------------------
    DR = dict(
        mass_scale=(0.85, 1.15),      # dry mass
        fuel_scale=(0.70, 1.30),      # propellant load
        isp_scale=(0.90, 1.10),
        thrust_scale=(0.85, 1.15),    # engine authority
        gimbal_bias_deg=2.0,          # persistent misalignment, per axis
        wind_acc=0.25,                # m/s^2, constant lateral disturbance
        obs_pos_noise=0.5,            # m
        obs_vel_noise=0.15,           # m/s
        obs_quat_noise=0.004,
    )

    def __init__(self, render_mode=None, model_path=MJCF_PATH,
                 domain_randomize=True, curriculum=True, seed=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.np_random_ = np.random.default_rng(seed)

        self.rocket_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.qpos_adr = self.model.jnt_qposadr[jid]
        self.qvel_adr = self.model.jnt_dofadr[jid]
        self.yaw_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
        self.pitch_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor")
        self.thrust_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
        self.DT = self.model.opt.timestep
        # Touchdown is detected from contact, not from a height threshold: the
        # vehicle is 100 m long, so a tilted lander touches its fins down with
        # the centre of mass still several metres up, and a fixed height test
        # simply never fires.
        self.ground_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.ring_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "landing_ring")
        self.rocket_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tintin_visual")

        self.domain_randomize = domain_randomize
        self.use_curriculum = curriculum
        self.curriculum_level = 0
        self.max_curriculum_level = 40
        self.curriculum_params = {
            "altitude":    (30.0, 5.5, 250.0),    # m above the pad
            "lateral":     (0.0, 2.0, 80.0),      # m
            "vel_std":     (0.0, 0.4, 15.0),      # m/s
            "tilt_deg":    (0.0, 1.2, 45.0),
        }
        self.success_history = []

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self._sample_vehicle()
        self.reset()

    # ======================================================================
    #  vehicle sampling
    # ======================================================================
    def _sample_vehicle(self):
        r = self.np_random_
        D = self.DR
        if self.domain_randomize:
            self.dry_mass = self.DRY_MASS * r.uniform(*D["mass_scale"])
            self.start_fuel = self.START_FUEL * r.uniform(*D["fuel_scale"])
            self.isp = self.ISP * r.uniform(*D["isp_scale"])
            self.max_thrust = self.MAX_THRUST * r.uniform(*D["thrust_scale"])
            b = np.deg2rad(D["gimbal_bias_deg"])
            self.gimbal_bias = r.uniform(-b, b, 2)
            ang = r.uniform(0, 2 * np.pi)
            w = r.uniform(0, D["wind_acc"])
            self.wind = np.array([w * np.cos(ang), w * np.sin(ang), 0.0])
        else:
            self.dry_mass = self.DRY_MASS
            self.start_fuel = self.START_FUEL
            self.isp = self.ISP
            self.max_thrust = self.MAX_THRUST
            self.gimbal_bias = np.zeros(2)
            self.wind = np.zeros(3)
        # inertia scales with the mass the vehicle actually has
        self.base_inertia = np.array([1.2e9, 1.2e9, 3.0e7])

    # ======================================================================
    #  step
    # ======================================================================
    def step(self, action):
        self.step_count += 1
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        thrust_cmd = (a[0] + 1.0) * 0.5 * self.max_thrust
        yaw_cmd = a[1] * self.MAX_GIMBAL + self.gimbal_bias[0]
        pitch_cmd = a[2] * self.MAX_GIMBAL + self.gimbal_bias[1]

        # propellant
        if self.fuel_mass > 0.0:
            mdot = thrust_cmd / (self.isp * self.G0)
            used = mdot * self.DT * self.FRAME_SKIP
            if used > self.fuel_mass:
                thrust_cmd *= self.fuel_mass / max(used, 1e-9)
                self.fuel_mass = 0.0
            else:
                self.fuel_mass -= used
            m_now = self.dry_mass + self.fuel_mass
            self.model.body_mass[self.rocket_bid] = m_now
            self.model.body_inertia[self.rocket_bid] = self.base_inertia * (m_now / self.wet_mass)
        else:
            thrust_cmd = 0.0

        self.data.ctrl[self.thrust_act] = thrust_cmd
        self.data.ctrl[self.yaw_act] = np.clip(yaw_cmd, -self.MAX_GIMBAL, self.MAX_GIMBAL)
        self.data.ctrl[self.pitch_act] = np.clip(pitch_cmd, -self.MAX_GIMBAL, self.MAX_GIMBAL)

        # constant lateral disturbance, applied as a body force
        if self.wind.any():
            self.data.xfrc_applied[self.rocket_bid, :3] = self.wind * self.model.body_mass[self.rocket_bid]

        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            if self._in_contact():
                break

        m = self._metrics()
        terminated, truncated, success = self._terminate(m)
        reward, parts = self._reward(m, thrust_cmd, terminated, success)
        if terminated or truncated:
            self._update_curriculum(success)

        info = dict(success=success, fuel_remaining=self.fuel_mass,
                    altitude=m["alt"], outcome=self.outcome, **parts)
        return self._obs(), reward, terminated, truncated, info

    # ======================================================================
    #  reward
    # ======================================================================
    def _reward(self, m, thrust, terminated, success):
        """Shaping is non-positive by construction.

        The first version of this reward gave a +0.3 per-step "alive" bonus on
        top of a +4 upright term.  With a 4,500-step limit that made hovering to
        the time limit worth ~19,000 against ~8,300 for a successful landing,
        and SAC duly learned to hover: success peaked at 20 % around 50 k steps
        and decayed to 6 % by 250 k.  Every shaping term below is therefore a
        penalty, so the only way to score is to finish, and to finish quickly.
        """
        r = {}
        alt = max(m["alt"], 0.0)

        # attitude: 0 upright, -2 on its side
        r["upright"] = -2.0 * (1.0 - m["quat_w"] ** 2)

        # glideslope tracking, gated on being roughly upright so the policy is
        # not rewarded for diving at the pad nose-first
        gate = 1.0 if m["quat_w"] > 0.85 else 0.0
        target_vz = -np.clip(0.35 * alt, 0.6, 22.0)
        # weight the glideslope error more heavily near the pad, where getting
        # it wrong is the difference between a landing and a crater
        near = 1.0 + 2.0 * np.exp(-alt / 15.0)
        r["descent"] = -1.5 * near * gate * min(abs(m["vz"] - target_vz), 12.0) / 12.0

        r["lateral"] = -1.0 * min(m["lateral"], 120.0) / 120.0
        r["drift"] = -0.5 * min(float(np.linalg.norm(m["vel"][:2])), 20.0) / 20.0
        r["spin"] = -0.4 * min(float(np.linalg.norm(m["omega"])), 2.0) / 2.0
        r["fuel"] = -0.6 * (thrust / self.MAX_THRUST)
        r["time"] = -0.25

        r["terminal"] = 0.0
        if terminated or self.outcome == "timeout":
            if success:
                frac = self.fuel_mass / max(self.start_fuel, 1e-9)
                r["terminal"] = 3000.0 + 500.0 * frac
            elif self.outcome == "hard":
                # Graded, but always negative.  An earlier version used
                # -300 + 250*q with q in [0, 3], so a tidy-looking crash scored
                # +450 and the policy learned to arrive fast rather than flare.
                q = (max(0.0, 1.0 - m["lateral"] / 40.0)
                     + max(0.0, 1.0 - m["speed"] / 15.0)
                     + max(0.0, 1.0 - m["tilt"] / 0.3)) / 3.0
                r["terminal"] = -400.0 + 300.0 * q
            elif self.outcome == "timeout":
                r["terminal"] = -500.0
            else:
                r["terminal"] = -600.0
        return float(sum(r.values())), r

    # ======================================================================
    #  termination
    # ======================================================================
    def _terminate(self, m):
        self.outcome = None
        terminated = truncated = success = False

        if m["lateral"] > self.MAX_LATERAL_DIST:
            terminated, self.outcome = True, "drift"
        elif m["speed"] > self.MAX_VELOCITY:
            terminated, self.outcome = True, "overspeed"
        elif m["quat_w"] < 0.5:                       # past ~120 deg
            terminated, self.outcome = True, "tumble"
        elif self._in_contact():
            # The episode ends at first contact and the touchdown state is
            # judged there.  Holding contact for a fixed number of steps does
            # not work on this model: a 5,000 t mesh landing on the pad bounces
            # in and out of contact for several tenths of a second, so a
            # consecutive-contact counter never completes.
            terminated = True
            success = (m["speed"] < self.LANDING_SPEED
                       and m["tilt"] < self.LANDING_TILT
                       and m["lateral"] < self.LANDING_TOLERANCE)
            self.outcome = "success" if success else "hard"

        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated, self.outcome = True, "timeout"
        return terminated, truncated, success

    def _in_contact(self):
        """True when the vehicle mesh is touching the pad or the ground."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if self.rocket_gid in pair and (self.ground_gid in pair or self.ring_gid in pair):
                return True
        return False

    # ======================================================================
    #  reset
    # ======================================================================
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random_ = np.random.default_rng(seed)
        super().reset(seed=seed)
        self.step_count = 0
        self.touchdown_steps = 0
        self.outcome = None

        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0
        self._sample_vehicle()
        self.fuel_mass = self.start_fuel
        self.wet_mass = self.dry_mass + self.start_fuel
        self.model.body_mass[self.rocket_bid] = self.wet_mass
        self.model.body_inertia[self.rocket_bid] = self.base_inertia

        r = self.np_random_
        alt = self._level("altitude")
        lat = self._level("lateral")
        vstd = self._level("vel_std")
        tilt = self._level("tilt_deg")
        if options:
            alt = options.get("altitude", alt)
            lat = options.get("lateral", lat)
            vstd = options.get("vel_std", vstd)
            tilt = options.get("tilt_deg", tilt)

        rad = np.sqrt(r.uniform(0.0, 1.0)) * lat
        th = r.uniform(0, 2 * np.pi)
        self.data.qpos[self.qpos_adr:self.qpos_adr + 3] = [
            rad * np.cos(th), rad * np.sin(th), self.TARGET_Z + max(alt, 5.0)]

        if tilt > 0:
            t = np.deg2rad(r.uniform(0, tilt))
            ax = r.uniform(0, 2 * np.pi)
            self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = [
                np.cos(t / 2), np.cos(ax) * np.sin(t / 2), np.sin(ax) * np.sin(t / 2), 0.0]
        else:
            self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = [1, 0, 0, 0]

        self.data.qvel[self.qvel_adr:self.qvel_adr + 3] = r.normal(0, vstd, 3)
        self.data.qvel[self.qvel_adr + 2] -= 2.0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def _level(self, name):
        start, step, limit = self.curriculum_params[name]
        if not self.use_curriculum:
            return limit
        return min(start + self.curriculum_level * step, limit)

    def _update_curriculum(self, success):
        if not self.use_curriculum:
            return
        self.success_history.append(int(success))
        if len(self.success_history) > 60:
            self.success_history.pop(0)
        if len(self.success_history) >= 30:
            w = float(np.mean(self.success_history))
            if w > 0.6 and self.curriculum_level < self.max_curriculum_level:
                self.curriculum_level += 1
                self.success_history = []
            elif w < 0.15 and self.curriculum_level > 0:
                self.curriculum_level -= 1
                self.success_history = []

    # ======================================================================
    #  observation and metrics
    # ======================================================================
    def _metrics(self):
        pos = self.data.xpos[self.rocket_bid].copy()
        vel = self.data.qvel[self.qvel_adr:self.qvel_adr + 3].copy()
        quat = self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7].copy()
        omega = self.data.qvel[self.qvel_adr + 3:self.qvel_adr + 6].copy()
        return dict(pos=pos, vel=vel, quat=quat, omega=omega,
                    z=pos[2], alt=pos[2] - self.TARGET_Z, vz=vel[2],
                    quat_w=abs(quat[0]),
                    lateral=float(np.linalg.norm(pos[:2])),
                    speed=float(np.linalg.norm(vel)),
                    tilt=1.0 - abs(quat[0]))

    def _obs(self):
        m = self._metrics()
        r = self.np_random_
        D = self.DR
        pos = m["pos"].copy()
        pos[2] = m["alt"]
        vel = m["vel"].copy()
        quat = m["quat"].copy()
        if self.domain_randomize:
            pos = pos + r.normal(0, D["obs_pos_noise"], 3)
            vel = vel + r.normal(0, D["obs_vel_noise"], 3)
            quat = quat + r.normal(0, D["obs_quat_noise"], 4)
            quat = quat / (np.linalg.norm(quat) + 1e-9)
        return np.concatenate([
            pos / 100.0,
            vel / 50.0,
            quat,
            m["omega"] / 2.0,
            [self.fuel_mass / max(self.start_fuel, 1e-9)],
            [self.model.body_mass[self.rocket_bid] / self.WET_NOMINAL],
            [np.linalg.norm(pos[:2]) / 100.0],
            [np.clip(m["alt"], 0, 400) / 100.0],
        ]).astype(np.float32)

    WET_NOMINAL = DRY_MASS + START_FUEL

    def render(self):
        if self.render_mode == "human":
            import mujoco.viewer
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
