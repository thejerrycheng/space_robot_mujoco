# rocket_env/rocket_landing_env.py

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "tintin_thrust.xml")

class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # ----------------------------------------------------------------
        # 1. LOAD MODEL & PHYSICS
        # ----------------------------------------------------------------
        if not os.path.exists(MJCF_PATH):
            raise FileNotFoundError(f"Model file not found at: {MJCF_PATH}")

        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)

        # ----------------------------------------------------------------
        # 2. IDENTIFIERS & CONSTANTS
        # ----------------------------------------------------------------
        self.rocket_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.qpos_adr = self.model.jnt_qposadr[self.free_joint_id]
        self.qvel_adr = self.model.jnt_dofadr[self.free_joint_id]

        self.yaw_act   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
        self.pitch_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor")
        self.thrust_act= mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")

        # Physics Constants
        self.DRY_MASS = self.model.body_mass[self.rocket_bid]
        self.START_FUEL = 10.0
        self.ISP = 250.0
        self.G0 = 9.81
        self.DT = self.model.opt.timestep

        # Control Limits
        self.MAX_THRUST = 2200.0
        self.MAX_GIMBAL = np.deg2rad(20.0)

        # Task Constants
        self.TARGET_Z = 0.5
        self.MAX_STEPS = 2000
        self.MAX_LATERAL_DIST = 40.0  # Reset if rocket flies this far horizontally
        self.MAX_VELOCITY = 50.0      # Reset if rocket moves this fast

        # ----------------------------------------------------------------
        # 3. SPACES & CURRICULUM
        # ----------------------------------------------------------------
        self.curriculum_level = 0
        self.max_curriculum_level = 5
        
        # REVISED: Only Position, Velocity (Linear), and Tilt
        self.curriculum_params = {
            # Position
            "initial_altitude":     (10.0, 15.0),   # Start low, go high
            "lateral_offset":       (0.0, 2.0),   # Start center, go wide
            
            # Velocity (New: Random linear kicks in any direction)
            "initial_velocity_std": (0.0, 1.0),    # m/s variance
            
            # Orientation
            "initial_tilt_deg":     (0.0, 10.0),   # Up to 60 degrees tilt
        }
        
        self.success_history = []

        # Observation and Action spaces remain the same...
        obs_high = np.ones(20) * 200
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # ----------------------------------------------------------------
        # 4. STATE INITIALIZATION
        # ----------------------------------------------------------------
        self.fuel_mass = self.START_FUEL
        self.total_mass = self.DRY_MASS + self.START_FUEL
        self.orig_inertia = self.model.body_inertia[self.rocket_bid].copy()
        
        # Rendering
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

        # 1. Physics & Controls
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

        # 2. Extract State & Metrics
        obs = self._get_obs()
        state_metrics = self._get_state_metrics()

        # 3. Check Termination
        terminated, truncated, success = self._check_termination(state_metrics)

        # 4. Compute Reward
        reward, reward_info = self._compute_reward(
            state_metrics, thrust_cmd, terminated, success
        )

        # 5. Handle Curriculum
        if terminated:
            self.update_curriculum(success)

        # 6. Build Info
        info = {
            "success": success,
            "fuel_remaining": self.fuel_mass,
            **reward_info  # Merges reward components into info for debugging
        }

        return obs, reward, terminated, truncated, info

    # =========================================================================
    # LOGIC: REWARDS
    # =========================================================================
    def _compute_reward(self, m, thrust, terminated, success):
        """
        m: dict containing metrics (pos_err, vel_err, tilt, etc.)
        """
        rewards = {}
        
        # --- A. Shaping / Continuous Rewards ---
        
        # 1. Distance & Velocity Penalty (Standard)
        rewards["dist_pen"] = -1.0 * m["pos_err"]
        rewards["vel_pen"]  = -0.05 * m["vel_err"] # Low penalty to allow movement
        
        # 2. Upright Bonus (Dense)
        # Quat w=1 means upright. w^2 gives a nice curve 0 to 1.
        rewards["upright"] = 1.0 * (m["quat_w"] ** 2)

        # 3. Approach Bonus (Vector alignment)
        # Reward velocity pointing towards the target
        target_vec = np.array([0, 0, self.TARGET_Z]) - m["pos"]
        dist = np.linalg.norm(target_vec)
        if dist > 0.1:
            target_dir = target_vec / dist
            approach_vel = np.dot(m["vel"], target_dir)
            rewards["approach"] = 0.5 * approach_vel
        else:
            rewards["approach"] = 0.0

        # 4. Descent Profile (Glide Slope)
        # Desired vertical velocity decreases as we get closer to the ground
        desired_vz = -1.0 * max(m["z"] - self.TARGET_Z, 0.0)
        desired_vz = np.clip(desired_vz, -10.0, -0.5)
        # Reward for matching this velocity
        rewards["descent"] = 1.0 * np.exp(-0.5 * abs(m["vz"] - desired_vz))

        # 5. Action costs
        rewards["fuel"] = -0.0002 * thrust
        rewards["spin"] = -0.1 * m["ang_err"]

        # --- B. Terminal Rewards (Sparse) ---
        rewards["terminal"] = 0.0
        
        if terminated:
            if success:
                rewards["terminal"] = 300.0
                print("🌟 SUCCESS LANDING!")
            elif m["z"] < 0.1:  # Ground crash
                rewards["terminal"] = -100.0
            elif m["lateral_dist"] > self.MAX_LATERAL_DIST:  # Out of bounds
                rewards["terminal"] = -50.0
            elif m["vel_err"] > self.MAX_VELOCITY:  # Too fast
                rewards["terminal"] = -50.0

        # Sum total
        total_reward = sum(rewards.values()) + 0.1  # +0.1 survival bonus

        return total_reward, rewards

    # =========================================================================
    # LOGIC: TERMINATION
    # =========================================================================
    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        # 1. Crash (Ground Hit)
        if m["z"] < 0.5:
            terminated = True
        
        # 2. Out of Bounds
        if m["lateral_dist"] > self.MAX_LATERAL_DIST:
            terminated = True
            
        # 3. Unstable / Too Fast
        if m["vel_err"] > self.MAX_VELOCITY:
            terminated = True

        # 4. Success Conditions
        # Low height, close to center, slow speed, upright
        if (0.0 < m["z"] < 1.0 and 
            m["pos_err"] < 0.5 and 
            m["vel_err"] < 0.5 and 
            m["tilt"] < 0.05):
            success = True
            terminated = True

        # 5. Time Limit
        if self.step_count >= self.MAX_STEPS:
            truncated = True

        return terminated, truncated, success

    # =========================================================================
    # CORE: RESET & RENDER
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # 1. Reset Physics
        mujoco.mj_resetData(self.model, self.data)
        self._apply_curriculum_reset()
        
        # 2. Reset Parameters
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.START_FUEL
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia.copy()
        
        mujoco.mj_forward(self.model, self.data)

        # 3. Viewer Sync (Do not close!)
        if self.viewer is not None:
            self.viewer.sync()

        return self._get_obs(), {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            # Launch viewer in passive mode (runs on separate thread)
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        try:
            self.viewer.sync()
        except Exception:
            pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # =========================================================================
    # UTILITIES
    # =========================================================================
    def _apply_curriculum_reset(self):
        # 1. Retrieve current difficulty values
        alt      = self._curriculum_interp("initial_altitude")
        offset   = self._curriculum_interp("lateral_offset")
        tilt_max = self._curriculum_interp("initial_tilt_deg")
        vel_std  = self._curriculum_interp("initial_velocity_std")

        # -----------------------------------------------------------
        # A. POSITION (Altitude + Lateral Offset)
        # -----------------------------------------------------------
        # Randomize X/Y within the allowable lateral offset circle
        r = np.sqrt(np.random.uniform(0, offset**2))
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = alt
        
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = [x, y, z]

        # -----------------------------------------------------------
        # B. TILT (Random Direction)
        # -----------------------------------------------------------
        if tilt_max > 0:
            # Pick a random magnitude between 0 and max allowed tilt
            tilt_deg = np.random.uniform(0, tilt_max)
            tilt_rad = np.deg2rad(tilt_deg)
            
            # Pick a random axis in the XY plane to tilt around
            # (This ensures "different directions" of tilt)
            tilt_angle_direction = np.random.uniform(0, 2*np.pi)
            axis_x = np.cos(tilt_angle_direction)
            axis_y = np.sin(tilt_angle_direction)
            
            # Quaternion for axis-angle rotation
            # q = [cos(theta/2), ux*sin(theta/2), uy*sin(theta/2), uz*sin(theta/2)]
            half_angle = tilt_rad / 2
            q_tilt = np.array([
                np.cos(half_angle),
                axis_x * np.sin(half_angle),
                axis_y * np.sin(half_angle),
                0.0  # No Z component implies rotation vector is in XY plane
            ])
            self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = q_tilt
        else:
            # Perfectly upright
            self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = [1, 0, 0, 0]

        # -----------------------------------------------------------
        # C. VELOCITY (Linear Randomness)
        # -----------------------------------------------------------
        # Base descent velocity (always falling slightly)
        base_vz = -2.0 
        
        # Add random noise scaled by curriculum
        vx = np.random.uniform(-vel_std, vel_std)
        vy = np.random.uniform(-vel_std, vel_std)
        vz = np.random.uniform(-vel_std, vel_std) + base_vz

        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = [vx, vy, vz]
        
        # Zero out angular velocity (since we removed it from curriculum)
        self.data.qvel[self.qvel_adr+3 : self.qvel_adr+6] = [0, 0, 0]

    def _get_state_metrics(self):
        """Pre-calculates all common metrics needed for reward and logic."""
        pos = self._get_pos()
        vel = self._get_vel()
        quat = self._get_quat()
        ang_vel = self._get_ang_vel()

        return {
            "pos": pos,
            "vel": vel,
            "z": pos[2],
            "vz": vel[2],
            "quat_w": quat[0],
            "lateral_dist": np.linalg.norm(pos[:2]),
            "pos_err": np.linalg.norm([pos[0], pos[1], pos[2] - self.TARGET_Z]),
            "vel_err": np.linalg.norm(vel),
            "ang_err": np.linalg.norm(ang_vel),
            "tilt": 1.0 - quat[0]
        }

    # --- Data Accessors ---
    def _get_pos(self): return self.data.xpos[self.rocket_bid].copy()
    def _get_vel(self):
        # Convert world vel to body frame if needed, but here we usually want World frame for landing
        # Note: cvel is usually in body frame. 
        # For simplicity in this env, we might prefer World Frame linear velocity:
        return self.data.qvel[self.qvel_adr:self.qvel_adr+3].copy()
        
    def _get_acc(self): return self.data.cacc[self.rocket_bid][3:].copy()
    def _get_quat(self): return self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7].copy()
    def _get_ang_vel(self): return self.data.cvel[self.rocket_bid][:3].copy()
    def _get_ang_acc(self): return self.data.cacc[self.rocket_bid][:3].copy()

    def _get_obs(self):
        return np.array([
            *self._get_pos(),
            *self._get_vel(),
            *self._get_acc(),
            *self._get_quat(),
            *self._get_ang_vel(),
            *self._get_ang_acc(),
            self.fuel_mass
        ], dtype=np.float32)

    def _curriculum_interp(self, name):
        low, high = self.curriculum_params[name]
        alpha = self.curriculum_level / self.max_curriculum_level
        return low + alpha * (high - low)

    def update_curriculum(self, success):
        self.success_history.append(success)
        if len(self.success_history) > 100:
            self.success_history.pop(0)
        
        rate = np.mean(self.success_history)
        if rate > 0.7 and self.curriculum_level < self.max_curriculum_level:
            self.curriculum_level += 1
            print(f"🎯 Level Up: {self.curriculum_level}")
            self.success_history = [] # Reset history on level up
        elif rate < 0.2 and self.curriculum_level > 0:
            self.curriculum_level -= 1
            print(f"⚠️ Level Down: {self.curriculum_level}")