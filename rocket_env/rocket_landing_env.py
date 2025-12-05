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
        # Moon gravity is approx 1.62 m/s^2
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

        # --- PHYSICS CONSTANTS (Sanity Checked) ---
        self.DRY_MASS = self.model.body_mass[self.rocket_bid]
        
        # Req: Fuel is 100% of body weight
        self.START_FUEL = 1 * self.DRY_MASS 
        TOTAL_MASS = self.DRY_MASS + self.START_FUEL
        
        self.ISP = 250.0
        self.G0 = 9.81  # Standard gravity for ISP calc
        self.DT = self.model.opt.timestep

        # --- CONTROL LIMITS (Dynamic TWR) ---
        # To make sense, a lander usually needs a TWR (Thrust to Weight Ratio) > 1.0.
        # A TWR of 5.0 is comfortable for learning (can hover at 20% throttle).
        # Max Thrust = Mass * Gravity * TWR
        self.MAX_THRUST = TOTAL_MASS * MOON_G * 5.0
        
        # Gimbal range
        self.MAX_GIMBAL = np.deg2rad(20.0)

        # Task Constants
        self.TARGET_Z = 0.5
        self.MAX_STEPS = 2000
        self.MAX_LATERAL_DIST = 40.0
        self.MAX_VELOCITY = 50.0

        # ----------------------------------------------------------------
        # 3. SPACES & CURRICULUM (Smoother Gradient)
        # ----------------------------------------------------------------
        self.curriculum_level = 0
        # Increased to 10 so each step up is a smaller increment
        self.max_curriculum_level = 10
        
        self.curriculum_params = {
            # Altitude: Start reasonably low, go higher
            "initial_altitude":     (10.0, 25.0),
            
            # Lateral: Start at 0, but expand to 8m radius (Definitively 3D)
            # 8m offset is significant enough to require tilting to translate.
            "lateral_offset":       (0.0, 5.0),
            
            # Velocity: Start stable, add "wind/kick" noise
            "initial_velocity_std": (0.0, 1.0),
            
            # Orientation: Start upright, end with 15 deg tilt
            # 15 deg is recoverable but requires immediate action.
            "initial_tilt_deg":     (0.0, 15.0),
        }
        
        self.success_history = []

        # Observation and Action spaces
        obs_high = np.ones(20) * 200
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
            **reward_info  # Merges reward components into info for debugging
        }

        return obs, reward, terminated, truncated, info

    # =========================================================================
    # LOGIC: REWARDS
    # =========================================================================
    def _check_termination(self, m):
            terminated = False
            truncated = False
            success = False

            # 1. Crash
            if m["z"] < 0.4: terminated = True
            
            # 2. Out of Bounds
            if m["lateral_dist"] > self.MAX_LATERAL_DIST: terminated = True
                
            # 3. Unstable / Too Fast
            if m["vel_err"] > self.MAX_VELOCITY: terminated = True

            # 4. Success Conditions
            if (m["z"] < 1 and 
                m["pos_err"] < 1.0 and 
                m["vel_err"] < 1.0 and 
                m["tilt"] < 5.0): # CHANGED: Now checks for < 5 degrees
                success = True
                terminated = True

            if self.step_count >= self.MAX_STEPS: truncated = True

            return terminated, truncated, success

    def _compute_reward(self, m, thrust, terminated, success):
        rewards = {}
        
        # 1. UPRIGHT
        # Note: We still use quat_w for the smooth reward curve as it's cleaner for gradients
        rewards["upright"] = 2.0 * (m["quat_w"] ** 4)

        # 2. DISTANCE
        dist_reward = 5.0 / (1.0 + m["pos_err"])
        rewards["distance"] = dist_reward * (m["quat_w"] ** 4)

        # 3. PENALTIES
        rewards["speed"] = -0.05 * (m["vel_err"] ** 2)
        rewards["fuel"] = -0.001 * thrust

        # 4. STABILITY
        # CHANGED: tilt < 10 degrees (was 0.2)
        if m["vel_err"] < 2.0 and m["tilt"] < 10.0:
            rewards["stability"] = 0.5
        else:
            rewards["stability"] = 0.0

        # 5. TERMINAL
        rewards["terminal"] = 0.0
        
        if terminated:
            if success:
                rewards["terminal"] = 1000.0
                print("🌟 SUCCESS LANDING!")
            else:
                # Semi-Success Logic
                # CHANGED: tilt < 20 degrees
                is_upright = m["tilt"] < 20.0 
                is_close   = m["pos_err"] < 5.0
                is_slow    = m["vel_err"] < 5.0

                if is_upright and is_close and is_slow:
                    # Normalize quality based on degrees (0 to 20)
                    quality = (1.0 - m["tilt"]/20.0) + (1.0 - m["vel_err"]/5.0)
                    rewards["terminal"] = 100.0 * quality
                    print(f"⚠️ Semi-Success: {rewards['terminal']:.1f}")
                elif m["z"] < 0.2: 
                    rewards["terminal"] = -200.0
                elif m["lateral_dist"] > self.MAX_LATERAL_DIST: 
                    rewards["terminal"] = -100.0
                elif m["vel_err"] > self.MAX_VELOCITY: 
                    rewards["terminal"] = -100.0

        return sum(rewards.values()), rewards

    # =========================================================================
    # LOGIC: TERMINATION
    # =========================================================================
    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        # 1. Crash (Ground Hit) - slightly relaxed floor to prevent instant-death on touch
        if m["z"] < 0.4:
            terminated = True
        
        # 2. Out of Bounds
        if m["lateral_dist"] > self.MAX_LATERAL_DIST:
            terminated = True
            
        # 3. Unstable / Too Fast
        if m["vel_err"] > self.MAX_VELOCITY:
            terminated = True

        # 4. Success Conditions
        # Strict requirements for the final "+1000" reward
        if (m["z"] < 1 and          # Close to ground
            m["pos_err"] < 1.0 and    # Close to target X/Y
            m["vel_err"] < 1.0 and    # Very slow (Soft landing)
            m["tilt"] < 0.1):         # Upright
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

            # --- 🎥 CAMERA CONFIGURATION ---
            self.viewer.cam.lookat[:] = [0, 0, 0] 
            self.viewer.cam.distance = 20.0       
            self.viewer.cam.azimuth = 35
            self.viewer.cam.elevation = -20

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
        quat = self._get_quat() # [w, x, y, z]
        ang_vel = self._get_ang_vel()

        # --- TILT CALCULATION (DEGREES) ---
        # 1. Extract quaternion components
        w, x, y, z = quat
        
        # 2. Calculate the Z-component of the local Z-vector rotated to world space.
        #    This represents how "upright" the rocket is relative to global Z.
        #    Range: 1.0 (Up) to -1.0 (Upside Down)
        #    Formula derived from rotation matrix of quaternion.
        z_projection = 1.0 - 2.0 * (x**2 + y**2)
        
        # 3. Safe Arccos (clip to avoid numerical errors slightly outside -1,1)
        z_projection = np.clip(z_projection, -1.0, 1.0)
        
        # 4. Calculate angle in degrees
        tilt_deg = np.rad2deg(np.arccos(z_projection))

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
            "tilt": tilt_deg # Now in DEGREES (0 to 180)
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