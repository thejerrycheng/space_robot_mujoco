import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer

# Helper to normalize angles
def normalize_angle(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, model_path="assets/mjcf/realistic_param.xml"):
        super().__init__()

        # ----------------------------------------------------------------
        # 1. LOAD MODEL & PHYSICS
        # ----------------------------------------------------------------
        # Ensure the path points to your new 'realistic_param.xml'
        if not os.path.exists(model_path):
             # Fallback or raise error
             pass 

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # ----------------------------------------------------------------
        # 2. IDENTIFIERS & CONSTANTS
        # ----------------------------------------------------------------
        self.rocket_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.qpos_adr = self.model.jnt_qposadr[self.free_joint_id]
        self.qvel_adr = self.model.jnt_dofadr[self.free_joint_id]

        # UPDATED: Actuator names matching the new XML
        self.yaw_act   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor")
        self.pitch_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor")
        self.thrust_act= mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")

        # Physics Constants
        self.DRY_MASS = 5_000_000.0 # 5000 tons
        # UPDATED: Fuel needs to be massive. 
        # Mdot ~ 25MN / (250 * 1.62) ~ 61,000 kg/s. 
        # 1,000,000 kg gives approx 16 seconds of full burn.
        self.START_FUEL = 1_000_000.0 
        self.ISP = 250.0
        self.G0 = 1.62 # Moon Gravity for Isp calculation
        self.DT = self.model.opt.timestep

        # Control Limits
        # UPDATED: 25 MN Thrust
        self.MAX_THRUST = 25_000_000.0 
        # UPDATED: +/- 30 degrees gimbal range
        self.MAX_GIMBAL = np.deg2rad(30.0)

        # Task Constants
        # UPDATED: Calculated based on geometry:
        # Ground @ -56m. Thruster bottom @ -40m (relative to CoM).
        # Touchdown happens when CoM Z = -16.0m.
        self.TARGET_Z = -16.0 
        self.MAX_STEPS = 2000 # 20 seconds
        self.MAX_LATERAL_DIST = 100.0 # Increased for larger scale
        self.MAX_VELOCITY = 100.0      
        self.LANDING_TOLERANCE = 1.0   

        # ----------------------------------------------------------------
        # 3. SPACES & CURRICULUM
        # ----------------------------------------------------------------
        self.curriculum_level = 0
        self.max_curriculum_level = 45 
        
        # REVISED CURRICULUM for Heavy Rocket (Start higher)
        self.curriculum_params = {
            # Altitude (Relative to Target Z): Start 20m above target, go up to 200m
            "initial_altitude_offset": (20.0, 4.0, 200.0),
            
            # Position: Start 0m offset, go to 40m
            "lateral_offset":       (0.0, 1.0, 40.0),
            
            # Velocity: Start 0 variance, +1.0 m/s per level
            "initial_velocity_std": (0.0, 1.0, 30.0),
            
            # Tilt: Start 0 deg, +1 deg per level (Heavy rockets are harder to recover)
            "initial_tilt_deg":     (0.0, 1.0, 45.0),
        }
        
        self.success_history = []

        # Observation space scaled for larger values
        obs_high = np.inf
        self.observation_space = spaces.Box(-obs_high, obs_high, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # ----------------------------------------------------------------
        # 4. STATE INITIALIZATION
        # ----------------------------------------------------------------
        self.fuel_mass = self.START_FUEL
        self.total_mass = self.DRY_MASS + self.START_FUEL
        # Important: Scale inertia based on mass? 
        # For now, we keep the XML inertia as the "Wet" inertia and just reduce mass.
        self.orig_inertia = self.model.body_inertia[self.rocket_bid].copy()
        
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.touchdown_steps = 0 

    # =========================================================================
    # CORE: STEP
    # =========================================================================
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1, 1)

        # 1. Controls
        # Map Action [-1, 1] -> [0, MAX_THRUST]
        thrust_cmd = (action[0] + 1) * 0.5 * self.MAX_THRUST
        
        # Map Action [-1, 1] -> Radians [-30deg, 30deg]
        # We pass the radian value to the servo. 
        # (Assuming the XML actuator is configured to take radians or user desires direct angle mapping)
        yaw_cmd    = action[1] * self.MAX_GIMBAL
        pitch_cmd  = action[2] * self.MAX_GIMBAL

        # 2. Physics Update (Fuel Consumption)
        if self.fuel_mass > 0:
            # m_dot = F / (Isp * g0)
            mdot = thrust_cmd / (self.ISP * self.G0)
            consumed = mdot * self.DT
            
            if consumed > self.fuel_mass:
                # Partial thrust if running out
                ratio = self.fuel_mass / consumed
                thrust_cmd *= ratio
                self.fuel_mass = 0
            else:
                self.fuel_mass -= consumed
                
            # Update MuJoCo mass (Simple approximation)
            # Note: Changing mass without changing inertia is physically slightly wrong 
            # but standard for simple Gym envs.
            self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.fuel_mass
        else:
            thrust_cmd = 0

        # Apply Controls
        self.data.ctrl[self.thrust_act] = thrust_cmd
        self.data.ctrl[self.yaw_act]    = yaw_cmd
        self.data.ctrl[self.pitch_act]  = pitch_cmd

        mujoco.mj_step(self.model, self.data)

        # 3. Extract State
        obs = self._get_obs()
        state_metrics = self._get_state_metrics()

        # 4. Logic
        terminated, truncated, success = self._check_termination(state_metrics)
        reward, reward_info = self._compute_reward(state_metrics, thrust_cmd, terminated, success)

        if terminated:
            self.update_curriculum(success)

        info = {
            "success": success,
            "fuel_remaining": self.fuel_mass,
            "altitude": state_metrics["z"] - self.TARGET_Z,
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    # =========================================================================
    # LOGIC: REWARDS (Scaled for new masses/distances)
    # =========================================================================
    def _compute_reward(self, m, thrust, terminated, success):
        rewards = {}
        
        # Normalize distance rewards by scale
        dist_to_target = m["pos_err"]
        
        # 1. Distance Penalty (Scaled down because distances are larger)
        rewards["dist_pen"] = -0.5 * dist_to_target
        
        # 2. Velocity Penalty
        rewards["vel_pen"]  = -0.1 * m["vel_err"]
        
        # 3. Upright Bonus (Heavily weighted)
        rewards["upright"] = 5.0 * (m["quat_w"] ** 2)

        # 4. Descent Profile
        # We want vz to be proportional to distance from ground
        # target_vz = -0.5 * (Altitude)
        altitude = m["z"] - self.TARGET_Z
        if altitude > 0:
            target_vz = -1.0 * np.clip(altitude / 10.0, 0.5, 20.0) # Cap descent speed
            vel_diff = abs(m["vz"] - target_vz)
            rewards["descent"] = 2.0 * np.exp(-0.5 * vel_diff)
        else:
            rewards["descent"] = 0.0

        # 5. Fuel Cost (Scaled: Thrust is in Millions, need small multiplier)
        # 25,000,000 * 1e-7 = 2.5 per step max
        rewards["fuel"] = -1e-7 * thrust 
        
        # 6. Terminal
        rewards["terminal"] = 0.0
        
        if terminated:
            if success:
                rewards["terminal"] = 2000.0
                print("🌟 5000-TON LANDING SUCCESS!")
            elif m["z"] < (self.TARGET_Z + 1.0):
                # Crash logic
                if m["lateral_dist"] < self.LANDING_TOLERANCE * 2:
                    rewards["terminal"] = -50.0 # Hard landing on pad
                else:
                    rewards["terminal"] = -200.0 # Missed pad
            else:
                rewards["terminal"] = -200.0 # Out of bounds/Fly away

        total_reward = sum(rewards.values())
        return total_reward, rewards

    # =========================================================================
    # LOGIC: TERMINATION
    # =========================================================================
    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        # 1. Bounds
        if m["lateral_dist"] > self.MAX_LATERAL_DIST:
            terminated = True
        
        # 2. Ground Interaction
        # Check against TARGET_Z (approx -16.0)
        # We give a buffer of 0.5m
        if m["z"] < (self.TARGET_Z + 0.5): 
            
            # Case A: Too fast or tilted -> Crash
            if m["vel_err"] > 5.0 or m["tilt"] > 0.2:
                terminated = True
            
            # Case B: Soft touchdown check
            else:
                self.touchdown_steps += 1
                # Must hold for 1 second (100 steps)
                if self.touchdown_steps > 100:
                    if m["vel_err"] < 1.0 and m["tilt"] < 0.1 and m["lateral_dist"] < self.LANDING_TOLERANCE:
                        success = True
                    terminated = True
        else:
            self.touchdown_steps = 0

        if self.step_count >= self.MAX_STEPS:
            truncated = True

        return terminated, truncated, success

    # =========================================================================
    # CORE: RESET
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.touchdown_steps = 0 
        
        mujoco.mj_resetData(self.model, self.data)
        
        self._apply_curriculum_reset()
        
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.START_FUEL
        
        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return self._get_obs(), {}

    def _apply_curriculum_reset(self):
        alt_offset = self._curriculum_interp("initial_altitude_offset")
        lat_offset = self._curriculum_interp("lateral_offset")
        tilt_d     = self._curriculum_interp("initial_tilt_deg")
        vel_s      = self._curriculum_interp("initial_velocity_std")

        # Position
        r = np.sqrt(np.random.uniform(0, lat_offset**2))
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Z is relative to Target (Ground)
        z = self.TARGET_Z + alt_offset
        
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = [x, y, z]

        # Orientation (Random tilt)
        if tilt_d > 0:
            tilt_rad = np.deg2rad(np.random.uniform(0, tilt_d))
            axis_angle = np.random.uniform(0, 2*np.pi)
            
            # Create quaternion from axis-angle
            # Axis is in XY plane
            ax = np.cos(axis_angle) * np.sin(tilt_rad/2)
            ay = np.sin(axis_angle) * np.sin(tilt_rad/2)
            az = 0
            w  = np.cos(tilt_rad/2)
            
            self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = [w, ax, ay, az]
        else:
            self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = [1, 0, 0, 0]

        # Velocity
        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = np.random.normal(0, vel_s, 3)
        self.data.qvel[self.qvel_adr+2] -= 2.0 # Initial downward velocity bias

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _get_obs(self):
        # Normalize obs slightly to help network
        # Pos divided by 100, Vel divided by 100, etc.
        p = self._get_pos()
        v = self._get_vel()
        
        return np.concatenate([
            (p - np.array([0,0,self.TARGET_Z])) / 100.0,
            v / 100.0,
            self._get_acc() / 10.0,
            self._get_quat(),
            self._get_ang_vel(),
            self._get_ang_acc(),
            [self.fuel_mass / self.START_FUEL]
        ]).astype(np.float32)

    def _get_state_metrics(self):
        pos = self._get_pos()
        vel = self._get_vel()
        quat = self._get_quat()
        
        # Distance from target center
        lat_dist = np.linalg.norm(pos[:2])
        
        # Total distance error (taking Target Z into account)
        dist_3d = np.linalg.norm([pos[0], pos[1], pos[2] - self.TARGET_Z])

        return {
            "pos": pos,
            "vel": vel,
            "z": pos[2],
            "vz": vel[2],
            "quat_w": quat[0],
            "lateral_dist": lat_dist,
            "pos_err": dist_3d,
            "vel_err": np.linalg.norm(vel),
            "tilt": 1.0 - abs(quat[0]), # Simple tilt metric
        }

    def _curriculum_interp(self, name):
        start, step, limit = self.curriculum_params[name]
        val = start + self.curriculum_level * step
        return min(val, limit)

    def update_curriculum(self, success):
        self.success_history.append(int(success))
        if len(self.success_history) > 50:
            self.success_history.pop(0)
        
        win_rate = np.mean(self.success_history)
        
        if len(self.success_history) >= 20:
            if win_rate > 0.75 and self.curriculum_level < self.max_curriculum_level:
                self.curriculum_level += 1
                self.success_history = []
                print(f"🎯 CURRICULUM LEVEL UP: {self.curriculum_level}")
            elif win_rate < 0.2 and self.curriculum_level > 0:
                self.curriculum_level -= 1
                self.success_history = []
                print(f"⚠️ CURRICULUM LEVEL DOWN: {self.curriculum_level}")

    # Data Access Wrappers
    def _get_pos(self): return self.data.xpos[self.rocket_bid].copy()
    def _get_vel(self): return self.data.qvel[self.qvel_adr:self.qvel_adr+3].copy()
    def _get_acc(self): return self.data.cacc[self.rocket_bid][3:].copy()
    def _get_quat(self): return self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7].copy()
    def _get_ang_vel(self): return self.data.cvel[self.rocket_bid][:3].copy()
    def _get_ang_acc(self): return self.data.cacc[self.rocket_bid][:3].copy()
    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()