import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "final.xml")

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
        self.START_FUEL = 100.0
        self.ISP = 250.0
        self.G0 = 1.62 # Moon Gravity for Isp calculation
        self.DT = self.model.opt.timestep

        # Control Limits
        self.MAX_THRUST = 4000.0
        self.MAX_GIMBAL = np.deg2rad(20.0)

        # Task Constants
        self.TARGET_Z = 0.5
        self.MAX_STEPS = 2000
        self.MAX_LATERAL_DIST = 50.0  
        self.MAX_VELOCITY = 60.0      
        self.LANDING_TOLERANCE = 0.5 

        # ----------------------------------------------------------------
        # 3. SPACES & CURRICULUM
        # ----------------------------------------------------------------
        self.curriculum_level = 0
        self.max_curriculum_level = 45 
        
        # REVISED CURRICULUM: (Start Value, Step Increment, Max Limit)
        self.curriculum_params = {
            "initial_altitude":     (5.0, 1.0, 50.0),
            "lateral_offset":       (0.0, 0.5, 22.5),
            "initial_velocity_std": (0.0, 0.5, 22.5),
            "initial_tilt_deg":     (0.0, 2.0, 90.0),
        }
        
        self.success_history = []

        # Explicitly cast to float32 to avoid Gym warnings
        obs_high = (np.ones(20) * 200).astype(np.float32)
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
        self.touchdown_steps = 0 

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
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    # =========================================================================
    # LOGIC: REWARDS
    # =========================================================================
    def _compute_reward(self, m, thrust, terminated, success):
        rewards = {}
        
        # 1. Distance & Lateral
        rewards["dist_pen"] = -2.0 * m["pos_err"]
        rewards["lat_pen"]  = -1.0 * m["lateral_dist"]

        # 2. Velocity
        rewards["vel_pen"]  = -0.05 * m["vel_err"]
        
        # 3. Upright Bonus
        rewards["upright"] = 2.0 * (m["quat_w"] ** 2)

        # 4. Approach Bonus
        target_vec = np.array([0, 0, self.TARGET_Z]) - m["pos"]
        dist = np.linalg.norm(target_vec)
        if dist > 0.1:
            target_dir = target_vec / dist
            approach_vel = np.dot(m["vel"], target_dir)
            rewards["approach"] = 1.0 * approach_vel if approach_vel > 0 else 0.0
        else:
            rewards["approach"] = 0.0

        # 5. Descent Profile
        desired_vz = -1.0 * max(m["z"] - self.TARGET_Z, 0.0)
        desired_vz = np.clip(desired_vz, -10.0, -0.5)
        rewards["descent"] = 1.0 * np.exp(-0.5 * abs(m["vz"] - desired_vz))

        # 6. Costs
        rewards["fuel"] = -0.0002 * thrust
        rewards["spin"] = -0.1 * m["ang_err"]

        # 7. Terminal
        rewards["terminal"] = 0.0
        
        if terminated:
            if success:
                rewards["terminal"] = 500.0
                print("🌟 PERFECT LANDING!")
            elif m["z"] < 0.1:
                if m["lateral_dist"] < self.LANDING_TOLERANCE:
                    rewards["terminal"] = -20.0 
                else:
                    rewards["terminal"] = -100.0 
            elif m["lateral_dist"] > self.MAX_LATERAL_DIST:
                rewards["terminal"] = -100.0
            elif m["vel_err"] > self.MAX_VELOCITY:
                rewards["terminal"] = -100.0

        total_reward = sum(rewards.values()) + 0.1
        return total_reward, rewards

    # =========================================================================
    # LOGIC: TERMINATION
    # =========================================================================
    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        # 1. Safety / Bounds Limits
        if m["lateral_dist"] > self.MAX_LATERAL_DIST:
            terminated = True
        if m["vel_err"] > self.MAX_VELOCITY:
            terminated = True

        # 2. Ground Interaction Logic
        if m["z"] < 0.3: 
            # Case A: Landed far from target -> Immediate Failure
            if m["lateral_dist"] > self.LANDING_TOLERANCE:
                terminated = True
            
            # Case B: Landed on target -> Check Persistence
            else:
                self.touchdown_steps += 1
                time_on_ground = self.touchdown_steps * self.DT
                
                # Check stability during the wait
                is_stable = (m["vel_err"] < 1.0 and m["tilt"] < 0.2)
                
                if not is_stable:
                    terminated = True
                elif time_on_ground >= 1.0:
                    # Final strict check
                    if m["vel_err"] < 0.5 and m["tilt"] < 0.05:
                        success = True
                    terminated = True
        else:
            self.touchdown_steps = 0

        # 3. Time Limit
        if self.step_count >= self.MAX_STEPS:
            truncated = True

        return terminated, truncated, success

    # =========================================================================
    # CORE: RESET & RENDER
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.touchdown_steps = 0 
        
        mujoco.mj_resetData(self.model, self.data)
        self.model.opt.gravity[:] = [0, 0, -1.62] 
        
        self._apply_curriculum_reset()
        
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.START_FUEL
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia.copy()
        
        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return self._get_obs(), {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
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
        # Revised interpolation: Start + Level * Step (capped at Limit)
        alt    = self._curriculum_interp("initial_altitude")
        offset = self._curriculum_interp("lateral_offset")
        tilt_d = self._curriculum_interp("initial_tilt_deg")
        vel_s  = self._curriculum_interp("initial_velocity_std")

        # Position (Randomized around center)
        r = np.sqrt(np.random.uniform(0, offset**2))
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = alt
        
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = [x, y, z]

        # Orientation
        if tilt_d > 0:
            tilt_rnd = np.random.uniform(0, tilt_d)
            tilt_rad = np.deg2rad(tilt_rnd)
            tilt_angle_direction = np.random.uniform(0, 2*np.pi)
            axis_x = np.cos(tilt_angle_direction)
            axis_y = np.sin(tilt_angle_direction)
            
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

        # Velocity
        base_vz = -2.0 
        vx = np.random.uniform(-vel_s, vel_s)
        vy = np.random.uniform(-vel_s, vel_s)
        vz = np.random.uniform(-vel_s, vel_s) + base_vz

        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = [vx, vy, vz]
        self.data.qvel[self.qvel_adr+3 : self.qvel_adr+6] = [0, 0, 0]

    def _get_state_metrics(self):
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

    def _get_pos(self): return self.data.xpos[self.rocket_bid].copy()
    def _get_vel(self): return self.data.qvel[self.qvel_adr:self.qvel_adr+3].copy()
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
        """ Calculates value based on start + level * step """
        start, step, limit = self.curriculum_params[name]
        val = start + self.curriculum_level * step
        return min(val, limit)

    def update_curriculum(self, success):
        self.success_history.append(success)
        if len(self.success_history) > 100:
            self.success_history.pop(0)
        
        rate = np.mean(self.success_history)
        
        # CHANGED: Threshold increased to 0.95 (95% success rate required)
        if rate > 0.95 and self.curriculum_level < self.max_curriculum_level:
            self.curriculum_level += 1
            print(f"🎯 Level Up: {self.curriculum_level}")
            self.success_history = []
            
        # REMOVED: Level Down logic
        # elif rate < 0.2 and self.curriculum_level > 0:
        #     self.curriculum_level -= 1
        #     print(f"⚠️ Level Down: {self.curriculum_level}")