import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer
import importlib

# Define path relative to this file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "realistic_param.xml")

class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, reward_func=None):
        super().__init__()
        
        # ----------------------------------------------------------------
        # DYNAMIC REWARD LOADING
        # ----------------------------------------------------------------
        if reward_func is not None:
            self.reward_func = reward_func
        else:
            try:
                mod = importlib.import_module("rocket_env.rewards.flip_and_fuel")
                self.reward_func = mod.compute_reward
            except ImportError:
                print("⚠️  Warning: Could not import default reward 'flip_and_fuel'. Using placeholder.")
                self.reward_func = lambda env, m, t, term, succ: (0.0, {})

        # 1. LOAD MODEL & PHYSICS
        if not os.path.exists(MJCF_PATH):
            raise FileNotFoundError(f"Model file not found at: {MJCF_PATH}")

        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        
        # --- GRAVITY: MOON ---
        MOON_G = 1.62
        self.model.opt.gravity[:] = [0, 0, -MOON_G]

        # 2. IDENTIFIERS
        self.rocket_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.qpos_adr = self.model.jnt_qposadr[self.free_joint_id]
        self.qvel_adr = self.model.jnt_dofadr[self.free_joint_id]

        self.yaw_act   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
        self.pitch_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor")
        self.thrust_act= mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")

        # --- PHYSICS CONSTANTS ---
        self.DRY_MASS = self.model.body_mass[self.rocket_bid]
        self.START_FUEL = 0.5 * self.DRY_MASS 
        TOTAL_MASS = self.DRY_MASS + self.START_FUEL
        
        self.ISP = 250.0
        self.G0 = 9.81  
        self.DT = self.model.opt.timestep

        # --- CONTROL LIMITS ---
        self.MAX_THRUST = TOTAL_MASS * MOON_G * 5.0
        self.MAX_GIMBAL = np.deg2rad(20.0)

        # --- TASK CONSTANTS (FIXED) ---
        self.TARGET_POS_WORLD = np.array([0.0, 0.0, 0.0])
        self.START_POS_FIXED  = np.array([15.0, 0.0, 15.0])
        self.INITIAL_SPEED    = 3.0 
        self.PITCH_DOWN_DEG   = 0.0
        self.LANDING_Z = 0.5 
        
        self.MAX_STEPS = 2000
        self.MAX_LATERAL_DIST = 100.0 
        self.MAX_VELOCITY = 100.0     

        # Observation Space
        obs_high = np.ones(23) * 500
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # 4. INIT STATE
        self.fuel_mass = self.START_FUEL
        self.total_mass = TOTAL_MASS
        self.orig_inertia = self.model.body_inertia[self.rocket_bid].copy()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)

    # =========================================================================
    # CORE: STEP
    # =========================================================================
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1, 1)

        # Physics
        thrust_cmd = (action[0] + 1) * 0.5 * self.MAX_THRUST
        yaw_cmd    = action[1] * self.MAX_GIMBAL
        pitch_cmd  = action[2] * self.MAX_GIMBAL

        if self.fuel_mass > 0:
            mdot = -thrust_cmd / (self.ISP * self.G0)
            self.fuel_mass = max(self.fuel_mass + mdot * self.DT, 0)
        else:
            thrust_cmd = 0

        self.data.ctrl[self.thrust_act] = thrust_cmd
        self.data.ctrl[self.yaw_act]    = yaw_cmd
        self.data.ctrl[self.pitch_act]  = pitch_cmd

        mujoco.mj_step(self.model, self.data)

        # State & Reward
        obs = self._get_obs()
        state_metrics = self._get_state_metrics()
        terminated, truncated, success = self._check_termination(state_metrics)
        
        # --- CALCULATE SEMI-SUCCESS (Fix for KeyError) ---
        # Semi-Success: In target zone (< 5m horizontal) but not a full success
        dist_xy = state_metrics["dist_xy"]
        semi_success = (dist_xy < 5.0) and not success

        # --- DELEGATE TO EXTERNAL REWARD FUNCTION ---
        reward, reward_info = self.reward_func(self, state_metrics, thrust_cmd, terminated, success)

        # --- UPDATE INFO DICT (Must include 'semi_success' for Monitor) ---
        info = {
            "success": success,
            "semi_success": semi_success, # <--- FIXED: Added this key
            "fuel": self.fuel_mass,
            "dist": state_metrics["target_dist_3d"],
            **reward_info 
        }

        return obs, reward, terminated, truncated, info

    # =========================================================================
    # LOGIC: TERMINATION
    # =========================================================================
    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        if m["z"] < 0.4: terminated = True
        if m["dist_xy"] > self.MAX_LATERAL_DIST: terminated = True
        if m["vel_err"] > self.MAX_VELOCITY: terminated = True

        # Success: Low altitude, close to 0,0 XY, slow, upright
        if (0.0 < m["z"] < 1.0 and 
            m["dist_xy"] < 0.5 and
            m["vel_err"] < 0.5 and 
            m["tilt"] < 0.05):
            success = True
            terminated = True

        if self.step_count >= self.MAX_STEPS: truncated = True

        return terminated, truncated, success

    # =========================================================================
    # RESET & UTILS
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. SET POSITION
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = self.START_POS_FIXED

        # 2. CALCULATE HEADING (YAW)
        dx = self.TARGET_POS_WORLD[0] - self.START_POS_FIXED[0]
        dy = self.TARGET_POS_WORLD[1] - self.START_POS_FIXED[1]
        yaw_angle = np.arctan2(dy, dx)

        # 3. CALCULATE PITCH (90 + 10 deg down)
        pitch_angle_rad = np.deg2rad(90.0 + self.PITCH_DOWN_DEG)

        # 4. CONSTRUCT QUATERNION
        hp = pitch_angle_rad / 2
        hy = yaw_angle / 2
        q_pitch = np.array([np.cos(hp), 0, np.sin(hp), 0])
        q_yaw = np.array([np.cos(hy), 0, 0, np.sin(hy)])
        
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_pitch
        q_total = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = q_total

        # 5. SET VELOCITY
        nz = np.cos(pitch_angle_rad)
        nh = np.sin(pitch_angle_rad)
        nx = nh * np.cos(yaw_angle)
        ny = nh * np.sin(yaw_angle)
        
        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = [
            nx * self.INITIAL_SPEED, 
            ny * self.INITIAL_SPEED, 
            nz * self.INITIAL_SPEED
        ]
        self.data.qvel[self.qvel_adr+3 : self.qvel_adr+6] = [0, 0, 0]

        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.START_FUEL
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia.copy()
        
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None: self.viewer.sync()
        return self._get_obs(), {}

    def _get_state_metrics(self):
        pos = self._get_pos()
        vel = self._get_vel()
        quat = self._get_quat()
        ang_vel = self._get_ang_vel()
        
        dist_xy = np.linalg.norm(pos[:2])
        dist_3d = np.linalg.norm(pos - self.TARGET_POS_WORLD)

        return {
            "pos": pos, "vel": vel, "z": pos[2], "vz": vel[2], "quat_w": quat[0],
            "dist_xy": dist_xy,
            "target_dist_3d": dist_3d,
            "pos_err": dist_3d, "vel_err": np.linalg.norm(vel),
            "ang_err": np.linalg.norm(ang_vel), "tilt": 1.0 - quat[0]
        }

    def _get_obs(self):
        pos = self._get_pos()
        rel_pos = -1.0 * pos
        return np.array([*pos, *rel_pos, *self._get_vel(), *self._get_acc(), *self._get_quat(), *self._get_ang_vel(), *self._get_ang_acc(), self.fuel_mass], dtype=np.float32)

    def render(self):
        if self.render_mode != "human": return
        if self.viewer is None: self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        try: self.viewer.sync()
        except: pass
    def close(self):
        if self.viewer: self.viewer.close(); self.viewer = None
    # Accessors
    def _get_pos(self): return self.data.xpos[self.rocket_bid].copy()
    def _get_vel(self): return self.data.qvel[self.qvel_adr:self.qvel_adr+3].copy()
    def _get_acc(self): return self.data.cacc[self.rocket_bid][3:].copy()
    def _get_quat(self): return self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7].copy()
    def _get_ang_vel(self): return self.data.cvel[self.rocket_bid][:3].copy()
    def _get_ang_acc(self): return self.data.cacc[self.rocket_bid][:3].copy()