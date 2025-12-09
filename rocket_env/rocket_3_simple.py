import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

# Import your reward script
from rewards.landing_reward import RocketReward # Assuming this path is correct

class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_path="assets/mjcf/realistic_param.xml", render_mode=None):
        super().__init__()
        
        # --- CONFIGURATION ---
        self.xml_path = xml_path
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Could not find XML at {self.xml_path}")
            
        self.render_mode = render_mode
        
        # Physics Constants
        self.START_FUEL = 4_500_000.0  
        self.DRY_MASS   = 500_000.0    
        self.ISP        = 400.0        
        self.G0         = 1.62         
        self.MAX_THRUST = 25_000_000.0 
        
        # Constraints
        self.MAX_STEPS = 1000 
        self.LANDING_TOLERANCE_POS = 10.0 
        self.LANDING_TOLERANCE_VEL = 10.0 
        self.LANDING_TOLERANCE_TILT = 8.0 

        # --- INITIAL CONDITIONS (FIXED) ---
        self.INIT_ALTITUDE = 500.0
        self.INIT_LATERAL_DIST = 10.0
        self.INIT_DOWNWARD_SPEED = 50.0

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

        # --- SPACES ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        high = np.inf
        self.observation_space = spaces.Box(low=-high, high=high, shape=(14,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Apply specific fixed reset conditions
        self._apply_fixed_reset()
        
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_body] = self.DRY_MASS + self.fuel_mass
        
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        self.viewer = None

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        
        # 1. APPLY CONTROL 
        # Original: Thrust from [-1, 1] to [0, MAX_THRUST]
        ctrl_thrust = (action[0] + 1) / 2.0 * self.MAX_THRUST
        if self.fuel_mass <= 0: ctrl_thrust = 0.0
            
        ctrl_yaw = action[1]
        ctrl_pitch = action[2]
        
        self.data.ctrl[self.act_thrust] = ctrl_thrust
        self.data.ctrl[self.act_yaw] = ctrl_yaw
        self.data.ctrl[self.act_pitch] = ctrl_pitch
        
        # 2. PHYSICS STEP
        dt = self.model.opt.timestep
        if ctrl_thrust > 0 and self.fuel_mass > 0:
            mdot = ctrl_thrust / (self.ISP * self.G0)
            burn = mdot * dt
            self.fuel_mass -= burn
            # Update rocket mass due to fuel burn
            self.model.body_mass[self.rocket_body] = self.DRY_MASS + self.fuel_mass
            
        mujoco.mj_step(self.model, self.data)

        # 3. OBSERVATION
        obs = self._get_obs()
        state_dict = self._get_state_dict()

        # 4. TERMINATION LOGIC
        terminated = False
        truncated = False
        success = False
        
        # Ground Contact
        if state_dict['alt'] < 50.0: 
            terminated = True
            is_upright = abs(state_dict['tilt']) < self.LANDING_TOLERANCE_TILT
            is_slow    = np.linalg.norm(state_dict['vel']) < self.LANDING_TOLERANCE_VEL
            is_close   = state_dict['lateral_dist'] < self.LANDING_TOLERANCE_POS
            
            if is_upright and is_slow and is_close:
                success = True

        # Bounds / Timeout
        if state_dict['lateral_dist'] > 3000.0 or state_dict['alt'] > 4000.0:
            truncated = True
        
        if self.step_count >= self.MAX_STEPS:
            truncated = True

        # 5. REWARD
        # Removed curriculum update, so no need to pass 'terminated'/'truncated' to rewarder
        # If the rewarder uses them for dense vs sparse rewards, they should be kept.
        # Assuming the original 'landing_reward' script might still use them:
        reward, r_info = self.rewarder.compute(state_dict, action, terminated, truncated, success)
        
        # 6. INFO
        info = {
            "is_success": success,
            "fuel_remaining": self.fuel_mass,
            **r_info
        }

        return obs, reward, terminated, truncated, info

    # ==========================================================
    # INITIALIZATION LOGIC (FIXED START)
    # ==========================================================
    def _apply_fixed_reset(self):
        """
        Sets the rocket to:
        - Alt: 500m
        - Dist: 10m away
        - Vel: 50 m/s (Down)
        - Orient: Horizontal (90 deg from vertical), pointing directly at the target.
        """
        
        # --- 1. POSITION ---
        alt = self.INIT_ALTITUDE
        dist_radius = self.INIT_LATERAL_DIST 
        
        # Choose a random angle for lateral position
        angle_pos = self.rng.uniform(0, 2*np.pi)
        
        x = dist_radius * np.cos(angle_pos)
        y = dist_radius * np.sin(angle_pos)
        
        # --- 2. VELOCITY ---
        speed = self.INIT_DOWNWARD_SPEED
        vel = np.array([0.0, 0.0, -speed])
        
        # --- 3. ORIENTATION (Horizontal, pointing at target) ---
        # Target (0, 0, 0)
        # Rocket is at (x, y, alt)
        
        # Yaw angle to point at (0,0) from (x,y)
        yaw = np.arctan2(-y, -x)
        
        # Horizontal orientation means pitch is 90 degrees from vertical (z-axis)
        target_pitch_from_up = 90.0
        rad_pitch = np.radians(target_pitch_from_up)
        
        # Rocket's nose vector (local_fwd) should be horizontal (cos(90)=0 vertical component)
        local_z = np.cos(rad_pitch) # Should be 0.0
        local_fwd = np.sin(rad_pitch) # Should be 1.0

        # Global vector components
        vec_x = local_fwd * np.cos(yaw)
        vec_y = local_fwd * np.sin(yaw)
        vec_z = local_z
        
        target_nose_vec = np.array([vec_x, vec_y, vec_z])
        # Ensure it's a unit vector (should be for sin/cos)
        target_nose_vec /= np.linalg.norm(target_nose_vec)
        
        # Convert target_nose_vec (desired [0,0,1] local frame) to Quat from base_vec [0,0,1]
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

        # Apply to MuJoCo Data
        self.data.qpos[0:3] = [x, y, alt]
        self.data.qpos[3:7] = quat # [w, x, y, z]
        self.data.qvel[0:3] = vel
        self.data.qvel[3:6] = [0,0,0] # Initial angular velocity is zero
        
        print(f"🚀 Environment Reset to Fixed Start: Alt={alt}m | Lateral_Dist={dist_radius}m | Vel_Down={speed} m/s | Orientation=Horizontal")


    # Removed _update_curriculum method
    
    # --- UTILITY METHODS ---
    def _get_obs(self):
        pos = self.data.xpos[self.rocket_body]
        vel = self.data.qvel[0:3]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        fuel_ratio = self.fuel_mass / self.START_FUEL
        # Observation space: [pos(3), vel(3), quat(4), ang_vel(3), fuel_ratio(1)] -> 14 elements
        return np.concatenate([pos, vel, quat, ang_vel, [fuel_ratio]], dtype=np.float32)

    def _get_state_dict(self):
        pos = self.data.xpos[self.rocket_body]
        vel = self.data.qvel[0:3]
        quat = self.data.qpos[3:7]
        # Convert quaternion [w, x, y, z] to scipy's [x, y, z, w] for rotation object
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        # Get the global Z-axis vector of the rocket (the nose vector)
        z_axis = r.apply([0, 0, 1])
        # Calculate tilt: angle between rocket's Z-axis and the global Z-axis (straight up)
        tilt = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
        return {
            'pos': pos, 'vel': vel, 'alt': pos[2],
            'lateral_dist': np.linalg.norm(pos[0:2]),
            'tilt': tilt, 'fuel_mass': self.fuel_mass
        }

    def render(self):
        if self.render_mode == "human":
            # Lazy initialization of the viewer
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()