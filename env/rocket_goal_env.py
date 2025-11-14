# envs/rocket_goal_env.py
import math, os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# ====== constants (your original values) ======
G0 = 9.80665
DRY_MASS   = 10000.0
FUEL_MASS0 = 18000.0
ISP        = 320.0
T_MIN      = 0.0
T_MAX      = 500000.0
GIMBAL_MAX_DEG = 10.0

# ====== utility / from your script ======
def set_mass_and_inertia(m, bid, new_mass, ref_mass=None):
    if ref_mass is None:
        ref_mass = float(m.body_mass[bid])
    scale = float(new_mass / max(ref_mass, 1e-9))
    m.body_mass[bid] = float(new_mass)
    m.body_inertia[bid, :] *= scale

def thrust_direction_body(pitch, yaw):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    v = np.array([sp*cy, sy, cp*cy], dtype=float)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def angle_of_attack_deg(d, bid):
    v = d.xvelp[bid].copy()
    spd = np.linalg.norm(v)
    if spd < 1e-6: return 0.0
    rw = -v / spd
    Rwb = d.xmat[bid].reshape(3,3)
    bz = Rwb[:,2]
    c = float(np.clip(np.dot(bz, rw), -1.0, 1.0))
    return math.degrees(math.acos(c))

def handles(m):
    return dict(
        rocket_bid = m.body('rocket_root').id,
        engine_sid = m.site('engine_site').id,
        ground_gid = m.geom('moon_ground').id,
        rocket_gid = m.geom('rocket_mesh').id,
        flame_core_gid  = m.geom('flame_core').id,
        flame_plume_gid = m.geom('flame_plume').id,
        goal_sid  = m.site('goal_site').id if 'goal_site' in [m.site_id2name(i) for i in range(m.nsite)] else None,
    )

def update_flame_visual(m, d, h, dir_body, thrust_N):
    r = float(np.clip(thrust_N / max(T_MAX,1e-9), 0.0, 1.0))
    flick = 0.92 + 0.16*np.random.random()
    core_len   = (0.25 + 2.0 * r) * flick
    core_rad   = 0.06 + 0.14 * r
    plume_len  = (0.40 + 3.5 * r) * flick
    plume_rad  = 0.10 + 0.25 * r
    core_size  = np.array([core_rad, core_rad, 0.5*core_len])
    plume_size = np.array([plume_rad, 0.5*plume_len, 0.0])

    def rotm_from_z(z):
        z = z / (np.linalg.norm(z)+1e-12)
        ref = np.array([1.0,0,0]) if abs(z[0]) < 0.9 else np.array([0,1.0,0])
        x = np.cross(ref, z); x /= (np.linalg.norm(x)+1e-12)
        y = np.cross(z, x)
        return np.stack([x,y,z], axis=1)

    def quat_from_rotm(R):
        t = np.trace(R)
        if t > 0:
            r = math.sqrt(1.0 + t); w = 0.5*r; s = 0.5/r
            x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
        else:
            i = np.argmax([R[0,0],R[1,1],R[2,2]])
            if i==0:
                r=math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2]); x=0.5*r; s=0.5/r
                y=(R[0,1]+R[1,0])*s; z=(R[0,2]+R[2,0])*s; w=(R[2,1]-R[1,2])*s
            elif i==1:
                r=math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2]); y=0.5*r; s=0.5/r
                x=(R[0,1]+R[1,0])*s; z=(R[1,2]+R[2,1])*s; w=(R[0,2]-R[2,0])*s
            else:
                r=math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1]); z=0.5*r; s=0.5/r
                x=(R[0,2]+R[2,0])*s; y=(R[1,2]+R[2,1])*s; w=(R[1,0]-R[0,1])*s
        return np.array([w,x,y,z], dtype=float)

    z_body = -dir_body
    R = rotm_from_z(z_body); quat = quat_from_rotm(R)
    site_local = m.site_pos[h['engine_sid']].copy()
    core_pos   = site_local + z_body * core_size[2]
    plume_pos  = site_local + z_body * plume_size[1]
    m.geom_size[h['flame_core_gid'],  :] = core_size
    m.geom_size[h['flame_plume_gid'], :] = plume_size
    m.geom_quat[h['flame_core_gid'],  :] = quat
    m.geom_quat[h['flame_plume_gid'], :] = quat
    m.geom_pos[h['flame_core_gid'],   :] = core_pos
    m.geom_pos[h['flame_plume_gid'],  :] = plume_pos
    mujoco.mj_forward(m, d)

# ====== goal-conditioned env ======
class RocketGoalEnv(gym.Env):
    """
    Action: [throttle, pitch, yaw] with pitch/yaw in radians (gimbal)
    Observation (18-dim):
      rel_pos(3), rel_vel(3), quat(4), ang_vel(3), mass_frac(1), thrust_N(1), , aoa_deg(1)
    """
    metadata = {"render_modes": ["none"], "render_fps": 100}

    def __init__(self,
                 xml_path: str,
                 frame_skip: int = 5,
                 gimbal_max_deg: float = GIMBAL_MAX_DEG,
                 time_limit_s: float = 120.0,
                 # goal / success parameters
                 goal_xy: Optional[Tuple[float,float]] = None,
                 goal_radius: float = 2.0,
                 success_vz: float = 0.75,     # |vz| <= m/s
                 success_vxy: float = 1.0,     # sqrt(vx^2+vy^2) <= m/s
                 success_tilt_deg: float = 10.0,
                 success_hold_time: float = 0.5,  # seconds continuous
                 # reset randomization
                 init_z_range=(60.0, 120.0),        # meters AGL
                 init_down_vz_range=(-14.0, -6.0),  # m/s (negative)
                 init_xy_spread=10.0,               # m
                 init_vxy_spread=1.0,               # m/s
                 seed: Optional[int] = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.h = handles(self.model)
        self.frame_skip = int(frame_skip)
        self.dt = float(self.model.opt.timestep)
        self.ctrl_dt = self.frame_skip * self.dt
        self.time_limit_s = float(time_limit_s)
        self.gimbal_max = math.radians(gimbal_max_deg)

        # action space
        low  = np.array([0.0, -self.gimbal_max, -self.gimbal_max], dtype=np.float32)
        high = np.array([1.0,  self.gimbal_max,  self.gimbal_max], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # observation space
        high_obs = np.array([np.inf]*3 + [np.inf]*3 + [1,1,1,1] + [np.inf]*3 +
                            [1e3, 1e6, self.gimbal_max, self.gimbal_max, 180.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

        # mass bookkeeping
        self.mass_state = {'fuel_mass': FUEL_MASS0, 'mass': DRY_MASS + FUEL_MASS0}
        set_mass_and_inertia(self.model, self.h['rocket_bid'], self.mass_state['mass'])
        mujoco.mj_forward(self.model, self.data)

        # goal config / success checks
        self.goal_xy = np.array(goal_xy if goal_xy is not None else (0.0, 0.0), dtype=float)
        self.goal_radius = float(goal_radius)
        self.success_vz = float(success_vz)
        self.success_vxy = float(success_vxy)
        self.success_tilt_deg = float(success_tilt_deg)
        self.success_hold_time = float(success_hold_time)
        self._success_hold_accum = 0.0

        # reset randomization
        self.init_z_range = tuple(init_z_range)
        self.init_down_vz_range = tuple(init_down_vz_range)
        self.init_xy_spread = float(init_xy_spread)
        self.init_vxy_spread = float(init_vxy_spread)

        self._rng = np.random.default_rng(seed)
        self._elapsed = 0.0
        self._last_thrust = 0.0
        self._last_pitch = 0.0
        self._last_yaw = 0.0

        # ensure goal site is visible
        self._move_goal_site(self.goal_xy)

    # ---------- public helpers ----------
    def set_goal(self, xy: Optional[Tuple[float,float]] = None):
        """Set a new goal (and move the site). If xy=None, sample around origin."""
        if xy is None:
            xy = (self._rng.uniform(-20, 20), self._rng.uniform(-20, 20))
        self.goal_xy = np.array(xy, dtype=float)
        self._move_goal_site(self.goal_xy)

    # ---------- RL API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # mass reset
        self.mass_state['fuel_mass'] = FUEL_MASS0
        self.mass_state['mass'] = DRY_MASS + FUEL_MASS0
        set_mass_and_inertia(self.model, self.h['rocket_bid'], self.mass_state['mass'])
        mujoco.mj_forward(self.model, self.data)

        # sample goal if requested via options
        if options and options.get("randomize_goal", False):
            self.set_goal(None)
        else:
            self._move_goal_site(self.goal_xy)

        # randomized initial state (downward & above ground)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        z0 = self._rng.uniform(*self.init_z_range)
        x0 = self.goal_xy[0] + self._rng.uniform(-self.init_xy_spread, self.init_xy_spread)
        y0 = self.goal_xy[1] + self._rng.uniform(-self.init_xy_spread, self.init_xy_spread)
        qpos[0:3] = np.array([x0, y0, z0])

        # small random tilt
        axis = self._unit(self._rng.normal(size=3))
        ang  = math.radians(self._rng.uniform(-4, 4))
        qw = math.cos(ang/2); qx,qy,qz = axis * math.sin(ang/2)
        qpos[3:7] = np.array([qw,qx,qy,qz])

        # downward velocity & some lateral
        vz = self._rng.uniform(*self.init_down_vz_range)
        qvel[0:3] = np.array([self._rng.uniform(-self.init_vxy_spread, self.init_vxy_spread),
                              self._rng.uniform(-self.init_vxy_spread, self.init_vxy_spread),
                              vz])
        qvel[3:6] = self._rng.uniform(-0.05, 0.05, size=3)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._elapsed = 0.0
        self._success_hold_accum = 0.0
        self._last_thrust = 0.0
        self._last_pitch = 0.0
        self._last_yaw = 0.0

        obs = self._get_obs(aoa_deg=angle_of_attack_deg(self.data, self.h['rocket_bid']))
        info = self._make_info(success=False, reason="")
        return obs, info

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        throttle, pitch, yaw = float(a[0]), float(a[1]), float(a[2])
        self._last_pitch, self._last_yaw = pitch, yaw

        # integrate physics for frame_skip
        T_cmd = 0.0
        for _ in range(self.frame_skip):
            # thrust cmd
            T_cmd = float(np.clip(throttle * T_MAX, T_MIN, T_MAX))
            # fuel burn
            mdot = T_cmd / (ISP * G0)
            fuel = max(0.0, self.mass_state['fuel_mass'] - mdot * self.model.opt.timestep)
            if fuel <= 0.0:
                T_cmd = 0.0
                throttle = 0.0
            new_mass = DRY_MASS + fuel
            if abs(new_mass - self.mass_state['mass']) > 1e-9:
                set_mass_and_inertia(self.model, self.h['rocket_bid'], new_mass, ref_mass=self.mass_state['mass'])
                mujoco.mj_forward(self.model, self.data)
                self.mass_state['mass'] = new_mass
                self.mass_state['fuel_mass'] = fuel

            # thrust vector
            dir_body  = thrust_direction_body(pitch, yaw)
            Rwb = self.data.xmat[self.h['rocket_bid']].reshape(3,3)
            dir_world = Rwb @ dir_body
            F = dir_world * T_cmd

            # wrench at engine site
            p = self.data.site_xpos[self.h['engine_sid']]
            com = self.data.xipos[self.h['rocket_bid']]
            tau = np.cross(p - com, F)

            self.data.xfrc_applied[self.h['rocket_bid'], :] = 0.0
            self.data.xfrc_applied[self.h['rocket_bid'], 0:3] = F
            self.data.xfrc_applied[self.h['rocket_bid'], 3:6] = tau

            update_flame_visual(self.model, self.data, self.h, dir_body, T_cmd)
            mujoco.mj_step(self.model, self.data)

        self._elapsed += self.ctrl_dt
        self._last_thrust = T_cmd

        # build obs & check terminations
        aoa_deg = angle_of_attack_deg(self.data, self.h['rocket_bid'])
        obs = self._get_obs(aoa_deg)

        success, stable = self._check_success()
        if stable:  # accumulate continuous stable time
            self._success_hold_accum += self.ctrl_dt
        else:
            self._success_hold_accum = 0.0

        terminated = False
        reason = ""
        if success and (self._success_hold_accum >= self.success_hold_time):
            terminated = True
            reason = "success"
        else:
            terminated, reason = self._check_failure()

        truncated = (self._elapsed >= self.time_limit_s)
        reward = 0.0  # you’ll fill this later

        info = self._make_info(success=(reason=="success"), reason=reason)
        return obs, reward, terminated, truncated, info

    # ---------- internals ----------
    def _get_obs(self, aoa_deg: float):
        bid = self.h['rocket_bid']
        d = self.data
        pos = d.xipos[bid].astype(np.float32)
        vel = d.xvelp[bid].astype(np.float32)
        quat = d.xquat[bid].astype(np.float32)
        ang  = d.xvelr[bid].astype(np.float32)

        rel_pos = pos[:2] - self.goal_xy  # x,y relative to goal
        rel_pos = np.array([rel_pos[0], rel_pos[1], pos[2]], dtype=np.float32)  # (dx,dy,z)
        rel_vel = vel.copy()  # world-frame velocities (vx,vy,vz)

        mass_frac = np.float32(self.mass_state['mass'] / (DRY_MASS + FUEL_MASS0))

        return np.concatenate([
            rel_pos, rel_vel, quat, ang,
            np.array([mass_frac, self._last_thrust, self._last_pitch, self._last_yaw, aoa_deg], dtype=np.float32)
        ]).astype(np.float32)

    def _check_success(self) -> Tuple[bool,bool]:
        """Return (criteria_met_now, stable_now)."""
        d = self.data; bid = self.h['rocket_bid']
        pos = d.xipos[bid].copy()
        vel = d.xvelp[bid].copy()
        # ground contact?
        in_contact = self._contact_with_ground()
        # within goal radius (horizontal)
        dist_xy = np.linalg.norm(pos[:2] - self.goal_xy)
        within = (dist_xy <= self.goal_radius)
        # attitude near upright
        Rwb = d.xmat[bid].reshape(3,3)
        tilt_deg = math.degrees(math.acos(np.clip(Rwb[2,2], -1.0, 1.0)))
        upright = (tilt_deg <= self.success_tilt_deg)
        # speeds small
        vxy = float(np.linalg.norm(vel[:2]))
        vz = float(abs(vel[2]))
        slow = (vz <= self.success_vz) and (vxy <= self.success_vxy)
        criteria = in_contact and within and upright and slow and (pos[2] <= 1.5)  # near ground
        # “stable now” = same as criteria (you could make this stricter)
        return criteria, criteria

    def _check_failure(self) -> Tuple[bool,str]:
        d = self.data; bid = self.h['rocket_bid']
        z = float(d.xipos[bid,2]); vz = float(d.xvelp[bid,2])
        Rwb = d.xmat[bid].reshape(3,3)
        tilt = math.degrees(math.acos(np.clip(Rwb[2,2], -1.0, 1.0)))
        # hard impact
        if self._contact_with_ground() and vz < -3.0:
            return True, "hard_contact"
        # tipped badly near ground
        if tilt > 85.0 and z < 5.0:
            return True, "tipped_over"
        # out of bounds
        pos = d.xipos[bid]
        if abs(pos[0]-self.goal_xy[0]) > 3000 or abs(pos[1]-self.goal_xy[1]) > 3000 or z > 5000:
            return True, "out_of_bounds"
        return False, ""

    def _contact_with_ground(self) -> bool:
        d = self.data
        gg = self.h['ground_gid']; rg = self.h['rocket_gid']
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == gg and c.geom2 == rg) or (c.geom2 == gg and c.geom1 == rg):
                return True
        return False

    def _make_info(self, success: bool, reason: str) -> Dict[str,Any]:
        d = self.data; bid = self.h['rocket_bid']
        pos = d.xipos[bid].copy(); vel = d.xvelp[bid].copy()
        dist_xy = float(np.linalg.norm(pos[:2] - self.goal_xy))
        vxy = float(np.linalg.norm(vel[:2])); vz = float(vel[2])
        return dict(
            success=success, reason=reason,
            time=self._elapsed,
            goal_xy=self.goal_xy.copy(),
            dist_to_goal_xy=dist_xy,
            z=float(pos[2]),
            vxy=vxy, vz=vz,
            mass=float(self.mass_state['mass']),
            fuel_mass=float(self.mass_state['fuel_mass']),
            thrust_N=float(self._last_thrust),
        )

    def _move_goal_site(self, goal_xy):
        if self.h['goal_sid'] is not None:
            # site is in world frame; ground plane is z=0
            self.model.site_pos[self.h['goal_sid'], :] = np.array([goal_xy[0], goal_xy[1], 0.0], dtype=float)
            mujoco.mj_forward(self.model, self.data)

    def _unit(self, v):
        n = np.linalg.norm(v)+1e-12
        return v/n

    def close(self): pass
