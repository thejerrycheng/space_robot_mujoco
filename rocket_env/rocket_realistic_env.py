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
        
        # --- GRAVITY: MOON ---
        MOON_G = 1.62
        self.model.opt.gravity[:] = [0, 0, -MOON_G]

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

        # --- PHYSICS CONSTANTS ---
        self.DRY_MASS = self.model.body_mass[self.rocket_bid]
        self.START_FUEL = 0.5 * self.DRY_MASS 
        TOTAL_MASS = self.DRY_MASS + self.START_FUEL
        
        self.ISP = 250.0
        self.G0 = 9.81  
        self.DT = self.model.opt.timestep

        # --- CONTROL LIMITS (TWR = 5.0) ---
        self.MAX_THRUST = TOTAL_MASS * MOON_G * 5.0
        self.MAX_GIMBAL = np.deg2rad(20.0)

        # Task Constants
        self.TARGET_Z = 0.5
        self.MAX_STEPS = 2000
        self.MAX_LATERAL_DIST = 100.0 # Increased: Rocket is flying fast horizontally
        self.MAX_VELOCITY = 100.0     # Increased: Entry speed is high

        # ----------------------------------------------------------------
        # 3. SPACES & CURRICULUM
        # ----------------------------------------------------------------
        self.curriculum_level = 0
        self.max_curriculum_level = 10
        
        self.curriculum_params = {
            # Altitude: High enough to allow recovery from the dive
            "initial_altitude":     (40.0, 40.0),
            
            # Target Distance: Moves further away
            "target_distance":      (10.0, 50.0),
            
            # Initial Speed (Magnitude along heading):
            # Level 0: 10 m/s (Manageable dive)
            # Level 10: 40 m/s (High speed re-entry simulation)
            "initial_speed":        (5.0, 20.0),
        }
        
        # Current target position
        self.current_target_pos = np.array([0.0, 0.0, self.TARGET_Z])
        
        self.success_history = []

        obs_high = np.ones(23) * 500
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # ----------------------------------------------------------------
        # 4. STATE INITIALIZATION
        # ----------------------------------------------------------------
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
            "target_dist": state_metrics["target_dist_3d"],
            **reward_info 
        }

        return obs, reward, terminated, truncated, info

    # =========================================================================
    # LOGIC: REWARDS
    # =========================================================================
    def _compute_reward(self, m, thrust, terminated, success):
        rewards = {}
        
        # --- A. Shaping / Continuous Rewards ---
        rewards["dist_pen"] = -1.0 * m["target_dist_3d"]
        rewards["vel_pen"]  = -0.05 * m["vel_err"]
        rewards["upright"]  = 1.0 * (m["quat_w"] ** 2)

        # Approach Bonus (Vector alignment towards Target)
        target_vec = self.current_target_pos - m["pos"]
        dist = np.linalg.norm(target_vec)
        if dist > 0.1:
            target_dir = target_vec / dist
            approach_vel = np.dot(m["vel"], target_dir)
            rewards["approach"] = 0.5 * approach_vel
        else:
            rewards["approach"] = 0.0

        # Descent Profile
        desired_vz = -1.0 * max(m["z"] - self.TARGET_Z, 0.0)
        desired_vz = np.clip(desired_vz, -10.0, -0.5)
        rewards["descent"] = 1.0 * np.exp(-0.5 * abs(m["vz"] - desired_vz))

        rewards["fuel"] = -0.0002 * thrust
        rewards["spin"] = -0.1 * m["ang_err"]

        # --- B. Terminal Rewards (Sparse) ---
        rewards["terminal"] = 0.0
        
        if terminated:
            if success:
                rewards["terminal"] = 500.0
                print(f"🌟 SUCCESS! Target: {self.current_target_pos[:2]}")
            elif m["z"] < 0.1:
                rewards["terminal"] = -100.0
            elif m["lateral_dist_from_target"] > self.MAX_LATERAL_DIST:
                rewards["terminal"] = -50.0
            elif m["vel_err"] > self.MAX_VELOCITY:
                rewards["terminal"] = -50.0

        total_reward = sum(rewards.values())
        return total_reward, rewards

    # =========================================================================
    # LOGIC: TERMINATION
    # =========================================================================
    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        if m["z"] < 0.5:
            terminated = True
        
        if m["lateral_dist_from_target"] > self.MAX_LATERAL_DIST:
            terminated = True
            
        if m["vel_err"] > self.MAX_VELOCITY:
            terminated = True

        # Success Conditions
        if (0.0 < m["z"] < 1.0 and 
            m["target_dist_2d"] < 0.5 and
            m["vel_err"] < 0.5 and 
            m["tilt"] < 0.05):
            success = True
            terminated = True

        if self.step_count >= self.MAX_STEPS:
            truncated = True

        return terminated, truncated, success

    # =========================================================================
    # CORE: RESET & RENDER
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        mujoco.mj_resetData(self.model, self.data)
        self._apply_curriculum_reset()
        
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.START_FUEL
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia.copy()
        
        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return self._get_obs(), {}

    def render(self):
        if self.render_mode != "human": return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        try: self.viewer.sync()
        except Exception: pass
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # =========================================================================
    # UTILITIES: CURRICULUM & RESET LOGIC
    # =========================================================================
    def _apply_curriculum_reset(self):
        # 1. Retrieve curriculum values
        alt = self._curriculum_interp("initial_altitude")
        speed_mag = self._curriculum_interp("initial_speed")
        target_dist = self._curriculum_interp("target_distance")

        # 2. TARGET POSITION
        # Place target somewhere in a circle around (0,0)
        theta_target = np.random.uniform(0, 2*np.pi)
        tx = target_dist * np.cos(theta_target)
        ty = target_dist * np.sin(theta_target)
        self.current_target_pos = np.array([tx, ty, self.TARGET_Z])

        # 3. ROCKET POSITION
        # Rocket always starts at (0,0, alt)
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = [0, 0, alt]

        # 4. ORIENTATION (90 deg + 10 deg down = 100 deg from Vertical)
        # We need to rotate around a random horizontal axis to point the nose
        # 10 degrees below the horizon.
        
        # Total pitch angle from Vertical Z+
        pitch_angle_deg = 90.0 + 10.0 
        pitch_angle_rad = np.deg2rad(pitch_angle_deg)
        
        # Pick a random Yaw direction (0 to 360) so the rocket isn't always flying North
        yaw_angle = np.random.uniform(0, 2*np.pi)
        
        # Combine rotations:
        # We want the vector (0,0,1) -> rotated by 100 deg -> rotated by Yaw.
        # This results in a quaternion.
        
        # Construct Quat: Pitch (around Y axis) * Yaw (around Z axis)
        # 1. Pitch Quaternion (Rotation around Y)
        # q = [cos(a/2), 0, sin(a/2), 0]
        hp = pitch_angle_rad / 2
        q_pitch = np.array([np.cos(hp), 0, np.sin(hp), 0])
        
        # 2. Yaw Quaternion (Rotation around Z)
        # q = [cos(a/2), 0, 0, sin(a/2)]
        hy = yaw_angle / 2
        q_yaw = np.array([np.cos(hy), 0, 0, np.sin(hy)])
        
        # 3. Multiply Quaternions: q_total = q_yaw * q_pitch
        # (Standard Hamilton product)
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_pitch
        
        q_total = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
        self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = q_total

        # 5. VELOCITY ALIGNMENT
        # The velocity must be aligned with the nose direction.
        # Nose direction (Z-local) in World Frame is calculated via spherical coords based on our angles:
        # Vertical angle = 100 deg. Azimuth = yaw_angle.
        
        # Z component = cos(100 deg)
        # Horizontal component = sin(100 deg)
        # X = H * cos(yaw)
        # Y = H * sin(yaw)
        
        nz = np.cos(pitch_angle_rad)
        nh = np.sin(pitch_angle_rad)
        nx = nh * np.cos(yaw_angle) # Note: yaw in math usually from X axis
        ny = nh * np.sin(yaw_angle)
        
        # Apply Speed Magnitude
        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = [
            nx * speed_mag, 
            ny * speed_mag, 
            nz * speed_mag
        ]
        
        # Zero angular velocity
        self.data.qvel[self.qvel_adr+3 : self.qvel_adr+6] = [0, 0, 0]

    def _get_state_metrics(self):
        pos = self._get_pos()
        vel = self._get_vel()
        quat = self._get_quat()
        ang_vel = self._get_ang_vel()

        target_vec = self.current_target_pos - pos
        target_dist_2d = np.linalg.norm(target_vec[:2])
        target_dist_3d = np.linalg.norm(target_vec)

        return {
            "pos": pos,
            "vel": vel,
            "z": pos[2],
            "vz": vel[2],
            "quat_w": quat[0],
            "lateral_dist_from_target": target_dist_2d,
            "target_dist_2d": target_dist_2d,
            "target_dist_3d": target_dist_3d,
            "pos_err": target_dist_3d,
            "vel_err": np.linalg.norm(vel),
            "ang_err": np.linalg.norm(ang_vel),
            "tilt": 1.0 - quat[0]
        }

    # --- Data Accessors ---
    def _get_pos(self): return self.data.xpos[self.rocket_bid].copy()
    def _get_vel(self): return self.data.qvel[self.qvel_adr:self.qvel_adr+3].copy()
    def _get_acc(self): return self.data.cacc[self.rocket_bid][3:].copy()
    def _get_quat(self): return self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7].copy()
    def _get_ang_vel(self): return self.data.cvel[self.rocket_bid][:3].copy()
    def _get_ang_acc(self): return self.data.cacc[self.rocket_bid][:3].copy()

    def _get_obs(self):
        pos = self._get_pos()
        rel_pos = self.current_target_pos - pos
        return np.array([
            *pos,
            *rel_pos,
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
            self.success_history = [] 
        elif rate < 0.2 and self.curriculum_level > 0:
            self.curriculum_level -= 1
            print(f"⚠️ Level Down: {self.curriculum_level}")