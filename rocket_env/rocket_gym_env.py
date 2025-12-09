import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

# Import your reward script
from rewards.landing_reward import RocketReward

class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

# ===== Initialization =====

    def __init__(self, xml_path="assets/mjcf/realistic_param.xml", render_mode=None):
        super().__init__()
        
        # --- CONFIGURATION ---
        self.xml_path = xml_path
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Could not find XML at {self.xml_path}")
            
        self.render_mode = render_mode
        
        # Physics Constants
        self.START_FUEL = 200_000.0  
        self.DRY_MASS   = 200_000.0    
        self.ISP        = 400.0        
        self.G0         = 1.62         
        self.MAX_THRUST = 25_000_000.0 
        self.INITIAL_INERTIA = np.array([1.2e9, 1.2e9, 3.0e7])
        self.INITIAL_MASS = self.DRY_MASS + self.START_FUEL
        
        # Constraints
        self.MAX_STEPS = 2000 

        # INITIAL CONDITIONS (FIXED)
        self.INIT_ALTITUDE = 500.0
        self.INIT_LATERAL_DIST = 500.0
        self.INIT_DOWNWARD_SPEED = 50.0
        self.INIT_FORWARD_SPEED = 0.0
        self.INIT_OFFSET_DIST = 0

        # Success Criteria
        self.TARGET_VEL = 0  # m/s
        self.TARGET_POS = 0   # m
        self.TARGET_TILT_ANGLE = 0   # degrees
        self.TARGET_ALTITUDE = 55.0   # m

        self.LANDING_TOLERANCE_ALT = 10.0
        self.LANDING_TOLERANCE_POS = 100.0 
        self.LANDING_TOLERANCE_VEL = 20.0 
        self.LANDING_TOLERANCE_TILT = 30.0 

        # Truncation Boundaries
        self.MAX_FLIGHT_SPEED = 500.0   # Example: Max speed allowed (m/s)
        self.MAX_LATERAL_DIST = 700.0  # Example: Max distance from target (m)

 
        # --- REWARD SETUP ---
        self.rewarder = RocketReward({
            'start_fuel': self.START_FUEL
        })

        # --- MUJOCO SETUP ---
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.rocket_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.act_thrust  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
        self.act_yaw     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
        self.act_pitch   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
        
        # --- VIEWER INITIALIZATION (Fix is here) ---
        self.viewer = None

        # --- SPACES ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        high = np.inf
        self.observation_space = spaces.Box(low=-high, high=high, shape=(14,), dtype=np.float32)

# ===== Reset Function =====

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Initialize episode with FIXED conditions
        self._set_fixed_initial_state()
        
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_body] = self.DRY_MASS + self.fuel_mass
        
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        
        # REMOVED: self.viewer = None 
        # (Do not reset the viewer here, or the window will disconnect)

        return self._get_obs(), {}
    

# ===== Step Function =====

    def step(self, action):
        self.step_count += 1
        
        # --- 1. CONTROL & PHYSICS ---
        applied_thrust = self._flight_control(action)
        self._rocket_dynamics(applied_thrust)

        # --- 2. OBSERVATION ---
        obs = self._get_obs()
        state_dict = self._get_state_dict()

        # --- 3. LOGIC (Termination/Truncation) ---
        terminated, success = self._check_termination(state_dict)
        
        truncated = False
        if state_dict['lateral_dist'] > 3000.0 or state_dict['alt'] > 4000.0:
            truncated = True
        if self.step_count >= self.MAX_STEPS:
            truncated = True

        # --- 4. REWARD ---
        reward, r_info = self.rewarder.compute(state_dict, action, terminated, truncated, success)

        info = {
            "is_success": success,
            "fuel_remaining": self.fuel_mass,
            **r_info
        }

        return obs, reward, terminated, truncated, info
    

    def _flight_control(self, action):
        """
        Maps RL actions to MuJoCo control inputs.
        Returns the applied thrust value for physics calculations.
        """
        # Map [-1, 1] to [0, MAX_THRUST]
        ctrl_thrust = (action[0] + 1) / 2.0 * self.MAX_THRUST
        
        # Hard constraint: No thrust if empty
        if self.fuel_mass <= 0: 
            ctrl_thrust = 0.0
            
        # Apply to MuJoCo data
        self.data.ctrl[self.act_thrust] = ctrl_thrust
        self.data.ctrl[self.act_yaw]    = action[1]
        self.data.ctrl[self.act_pitch]  = action[2]
        
        return ctrl_thrust

    def _rocket_dynamics(self, ctrl_thrust):
        """
        Updates mass, inertia, and steps the physics simulator.
        """
        dt = self.model.opt.timestep
        
        # Only update mass properties if engine is firing
        if ctrl_thrust > 0 and self.fuel_mass > 0:
            mdot = ctrl_thrust / (self.ISP * self.G0)
            burn = mdot * dt
            self.fuel_mass -= burn
            
            # 1. Update Mass
            self.model.body_mass[self.rocket_body] = self.DRY_MASS + self.fuel_mass
            
            # 2. Update Inertia (Linear Scaling)
            mass_ratio = self.model.body_mass[self.rocket_body] / self.INITIAL_MASS
            self.model.body_inertia[self.rocket_body] = self.INITIAL_INERTIA * mass_ratio
            
        # 3. Step Simulator
        mujoco.mj_step(self.model, self.data)

    def _check_termination(self, state):
        """
        Determines termination (ground contact) and success (smooth landing),
        and checks for truncation boundaries.
        """
        terminated = False
        success = False
        
        # --- NEW: Check for Excessive Tilt ---
        # state['tilt'] is assumed to be the deviation from the upright Z-axis in degrees.
        # We check if the deviation exceeds 100 degrees.
        if abs(state['tilt']) > 100.0:
            terminated = True
            print("🚀💥❌: Tilt too large (>{:.1f} deg from Z-axis)".format(100.0))
            # Return early since the episode is over due to tilt
            return terminated, success

        # Check for flight boundaries
        if np.linalg.norm(state['vel']) > self.MAX_FLIGHT_SPEED:
            terminated = True
            print("🚀💥❌: Max Speed Exceeded") # Optional logging
        
        if state['lateral_dist'] > self.MAX_LATERAL_DIST:
            terminated = True
            print("🚀💥❌: Max Distance Exceeded") # Optional logging

        # --- Termination (Ground Contact) ---
        
        # Termination condition is triggered when altitude drops below landing height
        if state['alt'] < self.TARGET_ALTITUDE + self.LANDING_TOLERANCE_ALT:
            terminated = True
            
            # Success Criteria (Checked only upon termination)
            is_upright = abs(state['tilt']) < self.LANDING_TOLERANCE_TILT
            is_slow    = np.linalg.norm(state['vel']) < self.LANDING_TOLERANCE_VEL
            is_close   = state['lateral_dist'] < self.LANDING_TOLERANCE_POS
            
            if is_upright and is_slow and is_close:
                success = True
                print("🚀✅ Successful Landing!") # Optional logging
            else:
                print("🚀💥❌ Crash Landing!") # Optional logging
                
        # We return both flags: success and terminated.
        # Note: If multiple conditions are met, `terminated` remains True.
        return terminated, success

    # ==========================================================
    # INITIALIZATION LOGIC (FIXED)
    # ==========================================================
    def _set_fixed_initial_state(self):
        """
        Sets a strictly deterministic initial state.
        Pos: (500, 0, 2000)
        Vel: (0, 0, -350)
        Orient: 90 deg (Horizontal), pointing towards (0,0)
        """
        
        # --- 1. POSITION ---
        # Fixed 50m on X axis, 500m up
        x = self.INIT_LATERAL_DIST + self.rng.uniform(-20.0, 20.0)  # Add ±20m noise to x position
        y = self.INIT_OFFSET_DIST + self.rng.uniform(-20.0, 20.0)  # Add ±20m noise to y position
        alt = self.INIT_ALTITUDE + self.rng.uniform(-50.0, 50.0)  # Add ±50m noise to altitude
        
        # --- 2. VELOCITY ---
        # Fixed 50 m/s downwards
        vel = np.array([self.INIT_FORWARD_SPEED, 0.0, -self.INIT_DOWNWARD_SPEED])  # Add small forward speed
        
        # --- 3. ORIENTATION ---
        # Point the nose roughly toward the target (0,0)
        # Since we are at x=500, y=0, the target is -X direction
        yaw = np.arctan2(-y, -x) # Should be pi (180 deg) or -pi
        
        # Fixed pitch: 90 degrees (Horizontal)
        # 0 deg is Vertical Up, 90 is Horizontal
        target_pitch_from_up = 90.0
        
        rad_pitch = np.radians(target_pitch_from_up)
        local_z = np.cos(rad_pitch)
        local_fwd = np.sin(rad_pitch)
        
        # Convert local forward/up to global vector based on Yaw
        vec_x = local_fwd * np.cos(yaw)
        vec_y = local_fwd * np.sin(yaw)
        vec_z = local_z
        
        target_nose_vec = np.array([vec_x, vec_y, vec_z])
        target_nose_vec /= np.linalg.norm(target_nose_vec)
        
        # Calculate Quaternion to align Z-axis with target_nose_vec
        base_vec = np.array([0,0,1])
        cross_axis = np.cross(base_vec, target_nose_vec)
        dot_val = np.dot(base_vec, target_nose_vec)
        
        if np.linalg.norm(cross_axis) < 1e-6:
            # If pointing straight down (dot=-1), rotate 180 deg around X
            if dot_val < 0:
                quat = [0, 1, 0, 0] # 180 deg around X
            else:
                quat = [1, 0, 0, 0] # Identity
        else:
            cross_axis /= np.linalg.norm(cross_axis)
            half_angle = np.arccos(dot_val) / 2.0
            
            w = np.cos(half_angle)
            xyz = cross_axis * np.sin(half_angle)
            quat = [w, xyz[0], xyz[1], xyz[2]]

        # Apply to Data
        self.data.qpos[0:3] = [x, y, alt]
        self.data.qpos[3:7] = quat # [w, x, y, z]
        self.data.qvel[0:3] = vel
        self.data.qvel[3:6] = [0,0,0] 

    def _get_obs(self):
        pos = self.data.xpos[self.rocket_body]
        vel = self.data.qvel[0:3]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        fuel_ratio = self.fuel_mass / self.START_FUEL
        return np.concatenate([pos, vel, quat, ang_vel, [fuel_ratio]], dtype=np.float32)

    def _get_state_dict(self):
        pos = self.data.xpos[self.rocket_body]
        vel = self.data.qvel[0:3]
        quat = self.data.qpos[3:7]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        z_axis = r.apply([0, 0, 1])
        # Tilt angle relative to world Z (0 is upright)
        tilt = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
        return {
            'pos': pos, 
            'vel': vel, 
            'alt': pos[2],
            'lateral_dist': np.linalg.norm(pos[0:2]),
            'tilt': tilt, 
            'fuel_mass': self.fuel_mass
        }

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()