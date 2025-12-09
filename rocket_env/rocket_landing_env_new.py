import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer

# ----------------------------------------------------------------
# PATH SETUP
# ----------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ensure this path matches your actual file structure
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "final.xml")

class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # 1. Load Model
        if not os.path.exists(MJCF_PATH):
            raise FileNotFoundError(f"Model file not found at: {MJCF_PATH}")

        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)

        # 2. Identifiers
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
        self.MAX_THRUST = 2200.0
        self.MAX_GIMBAL = np.deg2rad(20.0)

        # Task Constants
        self.TARGET_Z = 0.5
        self.MAX_STEPS = 2000
        self.MAX_LATERAL_DIST = 40.0 
        self.MAX_VELOCITY = 50.0     

        # ----------------------------------------------------------------
        # 3. DISCRETE CURRICULUM DEFINITION
        # ----------------------------------------------------------------
        self.curriculum_level = 0
        self.success_history = []
        
        # Explicit definitions per your request
        self.LEVEL_CONFIGS = {
            0: {
                "desc": "Very Easy: 10m high, perfectly centered, upright, still.",
                "alt": 10.0, "lat_dev": 0.0, "tilt_max": 0.0, "vel_std": 0.0
            },
            1: {
                "desc": "Easy: 10m high, small horizontal deviation (+-0.5m).",
                "alt": 10.0, "lat_dev": 0.5, "tilt_max": 0.0, "vel_std": 0.0
            },
            2: {
                "desc": "Medium: Add small random tilt (15 deg).",
                "alt": 10.0, "lat_dev": 0.5, "tilt_max": 15.0, "vel_std": 0.0
            },
            3: {
                "desc": "Hard: Add random initial velocity (2 m/s).",
                "alt": 10.0, "lat_dev": 0.5, "tilt_max": 15.0, "vel_std": 2.0
            },
            4: {
                "desc": "Expert: Higher altitude (20m), more tilt, higher velocity.",
                "alt": 20.0, "lat_dev": 2.0, "tilt_max": 30.0, "vel_std": 5.0
            }
        }
        self.max_level = max(self.LEVEL_CONFIGS.keys())

        # Observation/Action Spaces
        obs_high = np.ones(20) * 200
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # State Init
        self.fuel_mass = self.START_FUEL
        self.total_mass = self.DRY_MASS + self.START_FUEL
        self.orig_inertia = self.model.body_inertia[self.rocket_bid].copy()
        
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)

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

        # Info & Rewards
        obs = self._get_obs()
        state_metrics = self._get_state_metrics()
        
        terminated, truncated, success = self._check_termination(state_metrics)
        reward, reward_info = self._compute_reward(state_metrics, thrust_cmd, terminated, success)

        # Curriculum Update
        if terminated:
            self.update_curriculum(success)

        info = {
            "success": success,
            "fuel_remaining": self.fuel_mass,
            "level": self.curriculum_level,
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    # =========================================================================
    # LOGIC: REWARDS (Updated with Fuel Efficiency)
    # =========================================================================
    def _compute_reward(self, m, thrust, terminated, success):
        rewards = {}

        # 1. SHAPED DISTANCE (Exponential)
        dist_to_target = m["pos_err"]
        rewards["pos_shaped"] = 2.0 * np.exp(-2.0 * dist_to_target)

        # 2. UPRIGHT BONUS
        rewards["upright"] = 0.5 * (m["quat_w"] ** 2)

        # 3. VELOCITY DAMPING (Gate)
        # Scales with altitude: penalty is harsh near ground, lenient high up
        alt_factor = np.clip(m["z"] / 10.0, 0, 1) 
        vel_penalty_scale = 1.0 - (0.8 * alt_factor) 
        rewards["vel_pen"] = -0.1 * vel_penalty_scale * m["vel_err"]

        # 4. ACTION COSTS (Continuous Fuel Penalty)
        # Increased slightly to discourage hovering unnecessarily
        rewards["fuel_burn"] = -0.0005 * thrust 
        rewards["spin"] = -0.05 * m["ang_err"]

        # 5. TERMINAL REWARDS
        rewards["terminal"] = 0.0
        
        if terminated:
            if success:
                # Base Success Reward
                rewards["terminal"] = 100.0
                
                # --- NEW: FUEL EFFICIENCY BONUS ---
                # Reward = Weight * Amount of fuel left
                # Max Fuel is 10.0. If fully efficient, this adds up to +50.0 extra.
                efficiency_bonus = 5.0 * self.fuel_mass
                rewards["terminal"] += efficiency_bonus
                
                print(f"🌟 SUCCESS! Fuel Left: {self.fuel_mass:.2f} (Bonus: +{efficiency_bonus:.2f})")
                
            elif m["z"] < 0.1: # Crash
                rewards["terminal"] = -10.0 - (0.5 * m["vel_err"])
            elif m["lateral_dist"] > self.MAX_LATERAL_DIST:
                rewards["terminal"] = -10.0
            
        total_reward = sum(rewards.values())
        return total_reward, rewards
    

    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        if m["z"] < 0.5: terminated = True # Crash
        if m["lateral_dist"] > self.MAX_LATERAL_DIST: terminated = True
        if m["vel_err"] > self.MAX_VELOCITY: terminated = True

        # STRICTER SUCCESS CONDITION
        # Height < 1m, Dist < 0.5m, Vel < 2.0m/s, Upright
        if (0.0 < m["z"] < 1.0 and 
            m["pos_err"] < 0.5 and 
            m["vel_err"] < 2.0 and 
            m["tilt"] < 0.1):
            success = True
            terminated = True

        if self.step_count >= self.MAX_STEPS: truncated = True

        return terminated, truncated, success

    # =========================================================================
    # CURRICULUM LOGIC
    # =========================================================================
    def update_curriculum(self, success):
        self.success_history.append(int(success))
        
        # Keep window size smallish (e.g., 20 episodes) for faster reaction
        if len(self.success_history) > 20:
            self.success_history.pop(0)

        # Only check if we have enough data
        if len(self.success_history) >= 20:
            success_rate = np.mean(self.success_history)
            
            # Level Up Condition: > 80% Success
            if success_rate > 0.8 and self.curriculum_level < self.max_level:
                old_lvl = self.curriculum_level
                self.curriculum_level += 1
                
                # Clear history so we have to prove ourselves at the new level
                self.success_history = [] 
                
                # PRINT CHANGES
                print(f"\n🚀 LEVEL UP! {old_lvl} -> {self.curriculum_level}")
                print(f"   Config: {self.LEVEL_CONFIGS[self.curriculum_level]['desc']}")
                
                # Diff print
                old_c = self.LEVEL_CONFIGS[old_lvl]
                new_c = self.LEVEL_CONFIGS[self.curriculum_level]
                for k in ["alt", "lat_dev", "tilt_max", "vel_std"]:
                    if old_c[k] != new_c[k]:
                        print(f"   Change [{k}]: {old_c[k]} -> {new_c[k]}")
                print("-" * 30)

            # NO LEVEL DOWN LOGIC (As requested)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        mujoco.mj_resetData(self.model, self.data)
        self._apply_curriculum_reset()
        
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.START_FUEL
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia.copy()
        
        mujoco.mj_forward(self.model, self.data)
        if self.viewer: self.viewer.sync()

        return self._get_obs(), {}

    def _apply_curriculum_reset(self):
        # Get config for current level
        config = self.LEVEL_CONFIGS[self.curriculum_level]
        
        alt = config["alt"]
        lat_dev = config["lat_dev"]
        tilt_max = config["tilt_max"]
        vel_std = config["vel_std"]

        # 1. Position (Cylinder random)
        r = np.sqrt(np.random.uniform(0, lat_dev**2)) if lat_dev > 0 else 0
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = alt
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = [x, y, z]

        # 2. Tilt (Random Axis)
        if tilt_max > 0:
            tilt_deg = np.random.uniform(0, tilt_max)
            tilt_rad = np.deg2rad(tilt_deg)
            tilt_axis_angle = np.random.uniform(0, 2*np.pi)
            axis_x = np.cos(tilt_axis_angle)
            axis_y = np.sin(tilt_axis_angle)
            
            half_angle = tilt_rad / 2
            q_tilt = np.array([
                np.cos(half_angle),
                axis_x * np.sin(half_angle),
                axis_y * np.sin(half_angle),
                0.0
            ])
            self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = q_tilt
        else:
            self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = [1, 0, 0, 0]

        # 3. Velocity (Random Linear)
        if vel_std > 0:
            vx = np.random.uniform(-vel_std, vel_std)
            vy = np.random.uniform(-vel_std, vel_std)
            vz = np.random.uniform(-vel_std, vel_std)
        else:
            vx, vy, vz = 0, 0, 0
            
        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = [vx, vy, vz]
        self.data.qvel[self.qvel_adr+3 : self.qvel_adr+6] = [0, 0, 0] # No angular vel

    # --- Metrics & Obs Helper (Same as before) ---
    def _get_state_metrics(self):
        pos = self.data.xpos[self.rocket_bid].copy()
        vel = self.data.qvel[self.qvel_adr:self.qvel_adr+3].copy()
        quat = self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7].copy()
        ang_vel = self.data.cvel[self.rocket_bid][:3].copy()
        
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

    def _get_obs(self):
        # ... (Same as original code)
        pos = self.data.xpos[self.rocket_bid].copy()
        vel = self.data.qvel[self.qvel_adr:self.qvel_adr+3].copy()
        acc = self.data.cacc[self.rocket_bid][3:].copy()
        quat = self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7].copy()
        avel = self.data.cvel[self.rocket_bid][:3].copy()
        aacc = self.data.cacc[self.rocket_bid][:3].copy()
        return np.array([*pos, *vel, *acc, *quat, *avel, *aacc, self.fuel_mass], dtype=np.float32)

    def render(self):
        if self.render_mode != "human": return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close()