import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer
import importlib

# Define path relative to this file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "tintin_thrust.xml")

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
                # Fallback if custom reward not found
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
        # UPDATED: Set Start Fuel to 100% of Dry Mass (was 0.5)
        self.START_FUEL = 1.0 * self.DRY_MASS 
        TOTAL_MASS = self.DRY_MASS + self.START_FUEL
        
        self.ISP = 250.0
        self.G0 = 9.81  
        self.DT = self.model.opt.timestep

        # --- CONTROL LIMITS ---
        self.MAX_THRUST = TOTAL_MASS * MOON_G * 5.0
        self.MAX_GIMBAL = np.deg2rad(20.0)

        # --- TASK CONSTANTS ---
        self.TARGET_POS_WORLD = np.array([0.0, 0.0, 0.0])
        
        # Initial Polar Config (r, h)
        # UPDATED: Initial Radius changed to 15.0 (was 25.0)
        self.INIT_RADIUS = 15.0
        self.INIT_HEIGHT = 10.0
        self.INITIAL_SPEED = 5.0
        self.INITIAL_ROLL_DEG = 45.0
        
        self.LANDING_Z = 0.5 
        self.MAX_STEPS = 2000
        self.MAX_LATERAL_DIST = 25.0 
        self.MAX_VELOCITY = 100.0     

        # Observation Space (User Specific Polar)
        # Position: [r_2d, h, theta_elev] (3)
        # Velocity: [v_r_2d, v_h, v_theta] (3)
        # Rest: acc(3), quat(4), w(3), ang_acc(3), fuel(1)
        # Total Size = 3 + 3 + 3 + 4 + 3 + 3 + 1 = 20
        obs_high = np.inf * np.ones(20)
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

        # Physics Action Mapping
        thrust_cmd = (action[0] + 1) * 0.5 * self.MAX_THRUST
        yaw_cmd    = action[1] * self.MAX_GIMBAL
        pitch_cmd  = action[2] * self.MAX_GIMBAL

        # Fuel Consumption
        if self.fuel_mass > 0:
            mdot = -thrust_cmd / (self.ISP * self.G0)
            self.fuel_mass = max(self.fuel_mass + mdot * self.DT, 0)
        else:
            thrust_cmd = 0

        # Apply Control
        self.data.ctrl[self.thrust_act] = thrust_cmd
        self.data.ctrl[self.yaw_act]    = yaw_cmd
        self.data.ctrl[self.pitch_act]  = pitch_cmd

        mujoco.mj_step(self.model, self.data)

        # Get State & Reward
        obs = self._get_obs()
        state_metrics = self._get_state_metrics()
        
        terminated, truncated, success = self._check_termination(state_metrics)
        
        # Semi-Success
        dist_xy = state_metrics["dist_xy"]
        semi_success = (dist_xy < 5.0) and not success

        # Reward
        reward, reward_info = self.reward_func(self, state_metrics, thrust_cmd, terminated, success)

        info = {
            "success": success,
            "semi_success": semi_success,
            "fuel": self.fuel_mass,
            "dist": state_metrics["target_dist_3d"],
            **reward_info 
        }

        return obs, reward, terminated, truncated, info

    # =========================================================================
    # TERMINATION
    # =========================================================================
    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        # Failures
        if m["z"] < 0.4: terminated = True # Ground collision
        if m["dist_xy"] > self.MAX_LATERAL_DIST: terminated = True
        if m["vel_err"] > self.MAX_VELOCITY: terminated = True

        # Success Criteria
        if (0.0 < m["z"] < 1.0 and 
            m["dist_xy"] < 1 and
            m["vel_err"] < 1.0 and 
            m["tilt"] < 0.1): 
            success = True
            terminated = True

        if self.step_count >= self.MAX_STEPS: truncated = True

        return terminated, truncated, success

    # =========================================================================
    # RESET
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. SET POSITION (Polar -> Cartesian)
        # Start at INIT_RADIUS, on X-axis (theta=0)
        r = self.INIT_RADIUS
        theta_pos = 0.0
        
        start_x = r * np.cos(theta_pos)
        start_y = r * np.sin(theta_pos)
        start_z = self.INIT_HEIGHT
        
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = [start_x, start_y, start_z]

        # 2. CALCULATE ORIENTATION
        dx = self.TARGET_POS_WORLD[0] - start_x
        dy = self.TARGET_POS_WORLD[1] - start_y
        yaw_to_target = np.arctan2(dy, dx)

        # Pitch 90 degrees = Horizontal
        # The rocket's Z-axis (length axis) will lie in the XY plane
        pitch_rad = np.deg2rad(90.0) 
        roll_rad = np.deg2rad(self.INITIAL_ROLL_DEG)

        cy = np.cos(yaw_to_target * 0.5)
        sy = np.sin(yaw_to_target * 0.5)
        cp = np.cos(pitch_rad * 0.5)
        sp = np.sin(pitch_rad * 0.5)
        cr = np.cos(roll_rad * 0.5)
        sr = np.sin(roll_rad * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = [w, x, y, z]

        # 3. SET VELOCITY (AoA = 0)
        # Velocity matches nose vector
        nh = np.sin(pitch_rad) 
        nz = np.cos(pitch_rad) 
        
        vx = nh * np.cos(yaw_to_target) * self.INITIAL_SPEED
        vy = nh * np.sin(yaw_to_target) * self.INITIAL_SPEED
        vz = nz * self.INITIAL_SPEED

        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = [vx, vy, vz]
        self.data.qvel[self.qvel_adr+3 : self.qvel_adr+6] = [0, 0, 0]

        # 4. Reset Physics Props
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.START_FUEL
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia.copy()
        
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None: self.viewer.sync()
        return self._get_obs(), {}

    # =========================================================================
    # STATE METRICS (Internal)
    # =========================================================================
    def _get_state_metrics(self):
        pos = self._get_pos()
        vel = self._get_vel()
        quat = self._get_quat()
        
        dist_xy = np.linalg.norm(pos[:2])
        dist_3d = np.linalg.norm(pos - self.TARGET_POS_WORLD)

        return {
            "pos": pos, "vel": vel, "z": pos[2], "vz": vel[2], "quat_w": quat[0],
            "dist_xy": dist_xy,
            "target_dist_3d": dist_3d,
            "pos_err": dist_3d, "vel_err": np.linalg.norm(vel),
            "tilt": 1.0 - quat[0]
        }

    # =========================================================================
    # CUSTOM POLAR FUNCTIONS
    # =========================================================================
    def _get_polar_pos(self):
        """
        Returns [r_xy, h, theta_elev]
        r_xy: 2D horizontal distance to target (0,0,0)
        h: Height (z)
        theta_elev: Angle between position vector and horizontal plane
        """
        pos = self._get_pos()
        x, y, z = pos
        
        # 1. r_xy (2D Horizontal Distance)
        r_xy = np.linalg.norm(pos[:2])
        
        # 2. h (Height)
        h = z
        
        # 3. theta (Elevation Angle)
        # tan(theta) = h / r_xy
        if r_xy < 1e-6:
            if h > 0: theta = np.pi/2
            elif h < 0: theta = -np.pi/2
            else: theta = 0.0
        else:
            theta = np.arctan2(h, r_xy)
            
        return np.array([r_xy, h, theta], dtype=np.float32)

    def _get_polar_vel(self):
        """
        Returns rates of change for [r_xy, h, theta]
        """
        pos = self._get_pos()
        vel = self._get_vel()
        
        x, y, z = pos
        vx, vy, vz = vel
        
        r_xy = np.linalg.norm(pos[:2])
        h = z
        
        # 1. v_r (Radial velocity in 2D plane)
        if r_xy < 1e-6:
            v_r = 0.0
        else:
            v_r = (x*vx + y*vy) / r_xy
            
        # 2. v_h (Vertical velocity)
        v_h = vz
        
        # 3. v_theta (Angular velocity of elevation)
        # theta = arctan(h/r_xy) -> d(theta)/dt = (h'r - hr') / (r^2 + h^2)
        denom = r_xy**2 + h**2
        if denom < 1e-6:
            v_theta = 0.0
        else:
            v_theta = (vz*r_xy - h*v_r) / denom
            
        return np.array([v_r, v_h, v_theta], dtype=np.float32)

    # =========================================================================
    # OBSERVATION
    # =========================================================================
    def _get_obs(self):
        # 1. Position State (User Defined: r, h, theta)
        polar_pos = self._get_polar_pos()
        
        # 2. Velocity State (User Defined: v_r, v_h, v_theta)
        polar_vel = self._get_polar_vel()
        
        # 3. Other Body States
        acc = self._get_acc()
        quat = self._get_quat()
        ang_vel = self._get_ang_vel()
        ang_acc = self._get_ang_acc()
        
        return np.concatenate([
            polar_pos,  # 3
            polar_vel,  # 3
            acc,        # 3
            quat,       # 4
            ang_vel,    # 3
            ang_acc,    # 3
            [self.fuel_mass] # 1
        ])

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