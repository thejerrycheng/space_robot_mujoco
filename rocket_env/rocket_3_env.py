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

        # --- CURRICULUM SETUP ---
        self.curriculum_level = 0
        self.max_level = 10
        self.success_history = [] 
        
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
        
        # Apply specific Level 0 -> Max Curriculum
        self._apply_curriculum_reset()
        
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_body] = self.DRY_MASS + self.fuel_mass
        
        mujoco.mj_forward(self.model, self.data)
        
        self.step_count = 0
        self.viewer = None

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        
        # 1. APPLY CONTROL 
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
        reward, r_info = self.rewarder.compute(state_dict, action, terminated, truncated, success)
        
        # 6. CURRICULUM UPDATE
        if terminated or truncated:
            self._update_curriculum(success)

        info = {
            "is_success": success,
            "fuel_remaining": self.fuel_mass,
            "curriculum_level": self.curriculum_level,
            **r_info
        }

        return obs, reward, terminated, truncated, info

    # ==========================================================
    # CURRICULUM & INITIALIZATION LOGIC
    # ==========================================================
    def _apply_curriculum_reset(self):
        """
        Level 0: 
          - Vel: 350 m/s (Down)
          - Orient: Horizontal (90 deg), Pointing at target
        Max Level:
          - Vel: 500 m/s (Down)
          - Orient: Steep Dive (30 deg off vertical), Pointing at target
        """
        # 0.0 to 1.0 progress
        ratio = self.curriculum_level / self.max_level
        
        # --- 1. POSITION ---
        alt = 2000.0 
        dist_radius = 500.0 + (500.0 * ratio) # 500m to 1000m away
        angle_pos = self.rng.uniform(0, 2*np.pi)
        
        x = dist_radius * np.cos(angle_pos)
        y = dist_radius * np.sin(angle_pos)
        
        # --- 2. VELOCITY ---
        speed = 350.0 + (150.0 * ratio)
        vel = np.array([0.0, 0.0, -speed])
        
        # --- 3. ORIENTATION ---
        yaw = np.arctan2(-y, -x)
        start_angle = 90.0 # Horizontal
        end_angle = 150.0  # Steep dive (30 deg from pure down)
        target_pitch_from_up = start_angle + (end_angle - start_angle) * ratio
        rad_pitch = np.radians(target_pitch_from_up)
        local_z = np.cos(rad_pitch)
        local_fwd = np.sin(rad_pitch)
        vec_x = local_fwd * np.cos(yaw)
        vec_y = local_fwd * np.sin(yaw)
        vec_z = local_z
        target_nose_vec = np.array([vec_x, vec_y, vec_z])
        target_nose_vec /= np.linalg.norm(target_nose_vec)
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
        
        print(f"init_lvl_{self.curriculum_level} | Pitch_Off_Vert: {target_pitch_from_up:.1f}° | Vel: {speed:.1f} m/s")

    def _update_curriculum(self, success):
        self.success_history.append(1 if success else 0)
        if len(self.success_history) > 20:
            self.success_history.pop(0)
        
        win_rate = sum(self.success_history) / len(self.success_history)
        
        if len(self.success_history) >= 20:
            if win_rate > 0.7 and self.curriculum_level < self.max_level:
                self.curriculum_level += 1
                self.success_history = [] 
                print(f"🎓 PROMOTION! Now at Curriculum Level {self.curriculum_level}")
            elif win_rate < 0.2 and self.curriculum_level > 0:
                self.curriculum_level -= 1
                self.success_history = []
                print(f"📉 DEMOTION. Back to Level {self.curriculum_level}")

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
        tilt = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
        return {
            'pos': pos, 'vel': vel, 'alt': pos[2],
            'lateral_dist': np.linalg.norm(pos[0:2]),
            'tilt': tilt, 'fuel_mass': self.fuel_mass
        }

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()