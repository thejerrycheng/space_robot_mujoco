# envs/rocket_env.py
import os, math, time, dataclasses
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# ----- constants (from your script) -----
G0 = 9.80665
DRY_MASS   = 10000.0
FUEL_MASS0 = 18000.0
ISP        = 320.0
T_MIN      = 0.0
T_MAX      = 500000.0
GIMBAL_MAX_DEG = 10.0

# ----- helpers you already had (trimmed/adapted) -----
def set_mass_and_inertia(m, bid, new_mass, ref_mass=None):
    if ref_mass is None:
        ref_mass = float(m.body_mass[bid])
    scale = float(new_mass / max(ref_mass, 1e-9))
    m.body_mass[bid] = float(new_mass)
    m.body_inertia[bid, :] *= scale

def thrust_direction_body(pitch, yaw):
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    # R_x(pitch) R_y(yaw) @ ez
    dx =  sp*cy
    dy =  sy
    dz =  cp*cy
    v = np.array([dx, dy, dz], dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)

def angle_of_attack_deg(d, bid):
    v = d.xvelp[bid].copy()
    speed = np.linalg.norm(v)
    if speed < 1e-6:
        return 0.0
    rw = -v / speed
    Rwb = d.xmat[bid].reshape(3,3)
    bz = Rwb[:,2]  # body +Z in world
    c = float(np.clip(np.dot(bz, rw), -1.0, 1.0))
    return math.degrees(math.acos(c))

def handles(m):
    return dict(
        rocket_bid = m.body('rocket_root').id,
        engine_sid = m.site('engine_site').id,
        flame_core_gid  = m.geom('flame_core').id,
        flame_plume_gid = m.geom('flame_plume').id,
    )

def update_flame_visual(m, d, h, dir_body, thrust_N):
    # lightweight version (visual only)
    r = float(thrust_N / max(T_MAX, 1e-9)); r = np.clip(r, 0.0, 1.0)
    flick = 0.92 + 0.16*np.random.random()
    core_len   = (0.25 + 2.0 * r) * flick
    core_rad   = 0.06 + 0.14 * r
    plume_len  = (0.40 + 3.5 * r) * flick
    plume_rad  = 0.10 + 0.25 * r
    core_size  = np.array([core_rad, core_rad, 0.5*core_len])
    plume_size = np.array([plume_rad, 0.5*plume_len, 0.0])
    # orient +Z with exhaust (-dir_body)
    z_body = -dir_body
    # quick-orient: align z to vector (no need for perfect orthonormal frame for visuals)
    def rotm_from_z(z):
        z = z / (np.linalg.norm(z) + 1e-12)
        ref = np.array([1.0,0,0]) if abs(z[0]) < 0.9 else np.array([0,1.0,0])
        x = np.cross(ref, z); x /= (np.linalg.norm(x)+1e-12)
        y = np.cross(z, x)
        return np.stack([x,y,z], axis=1)
    def quat_from_rotm(R):
        t = np.trace(R)
        if t > 0:
            r = math.sqrt(1.0 + t); w = 0.5 * r; s = 0.5 / r
            x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
        else:
            i = np.argmax([R[0,0],R[1,1],R[2,2]])
            if i == 0:
                r = math.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2]); x=0.5*r; s=0.5/r
                y=(R[0,1]+R[1,0])*s; z=(R[0,2]+R[2,0])*s; w=(R[2,1]-R[1,2])*s
            elif i == 1:
                r = math.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2]); y=0.5*r; s=0.5/r
                x=(R[0,1]+R[1,0])*s; z=(R[1,2]+R[2,1])*s; w=(R[0,2]-R[2,0])*s
            else:
                r = math.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1]); z=0.5*r; s=0.5/r
                x=(R[0,2]+R[2,0])*s; y=(R[1,2]+R[2,1])*s; w=(R[1,0]-R[0,1])*s
        return np.array([w,x,y,z], dtype=float)
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

# ----- env -----
class RocketEnv(gym.Env):
    metadata = {"render_modes": ["none"], "render_fps": 100}

    def __init__(self,
                 xml_path: str,
                 frame_skip: int = 5,
                 gimbal_max_deg: float = GIMBAL_MAX_DEG,
                 time_limit_s: float = 120.0,
                 seed: Optional[int] = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.h = handles(self.model)

        self.frame_skip = int(frame_skip)
        self.dt = float(self.model.opt.timestep)
        self.time_limit_s = float(time_limit_s)

        self.gimbal_max = math.radians(gimbal_max_deg)

        # action: [throttle, pitch, yaw]
        low  = np.array([0.0, -self.gimbal_max, -self.gimbal_max], dtype=np.float32)
        high = np.array([1.0,  self.gimbal_max,  self.gimbal_max], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # observation (flat vector)
        # [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz, mass_frac, thrust_N, pitch,yaw, aoa_deg]
        obs_high = np.ones(3)*np.inf   # x,y,z
        vel_high = np.ones(3)*np.inf
        quat_high= np.ones(4)          # unit quaternion (we won’t enforce here)
        ang_high = np.ones(3)*np.inf
        other    = np.array([np.inf, T_MAX, self.gimbal_max, self.gimbal_max, 180.0], dtype=np.float32)
        high_obs = np.concatenate([obs_high, vel_high, quat_high, ang_high, other]).astype(np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

        # mass bookkeeping
        self.mass_state = {'fuel_mass': FUEL_MASS0, 'mass': DRY_MASS + FUEL_MASS0}
        set_mass_and_inertia(self.model, self.h['rocket_bid'], self.mass_state['mass'])
        mujoco.mj_forward(self.model, self.data)

        self._elapsed = 0.0
        self._rng = np.random.default_rng(seed)

    # ----- RL API -----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.data = mujoco.MjData(self.model)   # fresh state
        mujoco.mj_forward(self.model, self.data)

        # Reset mass
        self.mass_state['fuel_mass'] = FUEL_MASS0
        self.mass_state['mass'] = DRY_MASS + FUEL_MASS0
        set_mass_and_inertia(self.model, self.h['rocket_bid'], self.mass_state['mass'])
        mujoco.mj_forward(self.model, self.data)

        # Randomize a bit
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        # free joint: 7 pos (xyz + quat), 6 vel
        z0 = 80.0 + self._rng.uniform(-2.0, 2.0)
        qpos[0:3] = np.array([self._rng.uniform(-2.0,2.0), self._rng.uniform(-2.0,2.0), z0])
        small_tilt = math.radians(self._rng.uniform(-3,3))
        # orientation: small random axis tilt around z
        axis = np.array(self._rng.normal(size=3)); axis /= (np.linalg.norm(axis)+1e-12)
        dq = self._axis_angle_to_quat(axis, small_tilt)
        qpos[3:7] = dq  # overwrite orientation
        qvel[0:3] = self._rng.uniform(-0.5,0.5, size=3)   # translational
        qvel[3:6] = self._rng.uniform(-0.05,0.05, size=3) # angular
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._elapsed = 0.0
        obs = self._get_obs(np.array([0.0, 0.0, 0.0], dtype=np.float32), thrust_N=0.0, aoa_deg=0.0)
        info = {"thrust_N": 0.0, "mass": self.mass_state['mass']}
        return obs, info

    def step(self, action: np.ndarray):
        # sanitize action
        a = np.clip(np.asarray(action, dtype=np.float32),
                    self.action_space.low, self.action_space.high)
        throttle, pitch, yaw = float(a[0]), float(a[1]), float(a[2])

        # integrate physics with frame_skip
        thrust_N = 0.0
        dir_body = np.array([0,0,1.0], dtype=float)
        dir_world = dir_body.copy()

        for _ in range(self.frame_skip):
            # compute thrust
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

            # thrust vector world
            dir_body  = thrust_direction_body(pitch, yaw)
            Rwb = self.data.xmat[self.h['rocket_bid']].reshape(3,3)
            dir_world = Rwb @ dir_body
            F = dir_world * T_cmd

            # apply wrench at engine site
            p = self.data.site_xpos[self.h['engine_sid']]
            com = self.data.xipos[self.h['rocket_bid']]
            tau = np.cross(p - com, F)

            self.data.xfrc_applied[self.h['rocket_bid'], :] = 0.0
            self.data.xfrc_applied[self.h['rocket_bid'], 0:3] = F
            self.data.xfrc_applied[self.h['rocket_bid'], 3:6] = tau

            # visuals (optional; cheap)
            update_flame_visual(self.model, self.data, self.h, dir_body, T_cmd)

            mujoco.mj_step(self.model, self.data)
            thrust_N = T_cmd

        self._elapsed += self.frame_skip * self.dt

        # compute diagnostics
        aoa_deg = angle_of_attack_deg(self.data, self.h['rocket_bid'])
        obs = self._get_obs(np.array([pitch, yaw, throttle], dtype=np.float32),
                            thrust_N=thrust_N, aoa_deg=aoa_deg)

        # term/trunc
        terminated, reason = self._check_termination()
        truncated = (self._elapsed >= self.time_limit_s)

        reward = 0.0  # placeholder

        info = {
            "thrust_N": thrust_N,
            "mass": self.mass_state['mass'],
            "fuel_mass": self.mass_state['fuel_mass'],
            "aoa_deg": aoa_deg,
            "terminated_reason": reason,
        }
        return obs, reward, terminated, truncated, info

    # ----- helpers -----
    def _get_obs(self, act_vec, thrust_N: float, aoa_deg: float):
        bid = self.h['rocket_bid']
        d = self.data
        pos = d.xipos[bid].astype(np.float32)
        vel = d.xvelp[bid].astype(np.float32)
        quat = d.xquat[bid].astype(np.float32)
        ang  = d.xvelr[bid].astype(np.float32)
        mass_frac = np.float32(self.mass_state['mass'] / (DRY_MASS + FUEL_MASS0))
        throttle = act_vec[2] if act_vec.size == 3 else 0.0
        pitch = act_vec[0]; yaw = act_vec[1]
        obs = np.concatenate([
            pos, vel, quat, ang,
            np.array([mass_frac, thrust_N, pitch, yaw, aoa_deg], dtype=np.float32)
        ]).astype(np.float32)
        return obs

    def _check_termination(self) -> Tuple[bool, str]:
        d = self.data
        bid = self.h['rocket_bid']
        z = float(d.xipos[bid,2])
        vz = float(d.xvelp[bid,2])
        Rwb = d.xmat[bid].reshape(3,3)
        tilt = math.degrees(math.acos(np.clip(Rwb[2,2], -1.0, 1.0)))  # cos tilt
        # naive terrain plane at z=0 in this model
        if z < 0.5 and vz < -2.0:
            return True, "hard_contact"
        if tilt > 80.0 and z < 5.0:
            return True, "tipped_over"
        if abs(d.xipos[bid,0]) > 2000 or abs(d.xipos[bid,1]) > 2000 or z > 3000:
            return True, "out_of_bounds"
        return False, ""

    def _axis_angle_to_quat(self, axis, angle):
        s = math.sin(angle/2.0)
        return np.array([math.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

    def close(self):
        # nothing special: viewer is managed by teleop script
        pass
