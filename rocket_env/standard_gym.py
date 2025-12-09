import os
import copy
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

# Import your reward script
from rewards.landing_reward import RocketReward

# Path configuration
PATH = os.getcwd()
MODEL_XML_PATH = os.path.join(PATH, "assets", "mjcf", "realistic_param.xml")
DEFAULT_SIZE = 500


class RobotEnv(gym.Env):
    """Base robot environment class using MuJoCo."""
    
    def __init__(self, model_path, initial_qpos, n_substeps):
        """
        Initialize the base robot environment.
        
        Args:
            model_path: Path to the MuJoCo XML model file
            initial_qpos: Dictionary of initial joint positions
            n_substeps: Number of substeps per environment step
        """
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = n_substeps
        
        # Render settings
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        
        # Seed
        self.seed()
        
        # Allow subclass to setup specific IDs and parameters
        self._setup_references()
        
        # Initialization
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self._get_mjstate())
        self.goal = self._sample_goal()
        
        # Set spaces
        obs = self._get_obs()
        self._set_action_space()
        self.observation_space = self._set_observation_space(obs)
    
    @property
    def dt(self):
        """Return the timestep of each environment step."""
        return self.model.opt.timestep * self.n_substeps
    
    def seed(self, seed=None):
        """Seed the random number generator."""
        self.rng = np.random.default_rng(seed)
        return [seed]
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Action to apply
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply action and step simulation
        self._set_action(action)
        self._step_callback()
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward and check termination
        info = self._compute_info(obs, action)
        terminated, truncated = self._check_termination(obs, info)
        reward = self.compute_reward(obs, info, terminated, truncated)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            observation, info
        """
        if seed is not None:
            self.seed(seed)
            
        # Reset simulation
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        
        # Sample new goal
        self.goal = self._sample_goal()
        obs = self._get_obs()
        
        return obs, {}
    
    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        """Render the environment."""
        if mode == "rgb_array":
            if self.viewer is None or mode not in self._viewers:
                self._get_viewer(mode)
            
            mujoco.mj_forward(self.model, self.data)
            self.viewer.update_scene(self.data, camera="track")
            return self.viewer.render()
            
        elif mode == "human":
            if self.viewer is None:
                self._get_viewer(mode)
            self.viewer.sync()
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            if hasattr(self.viewer, 'close'):
                self.viewer.close()
            self.viewer = None
            self._viewers = {}
    
    def _get_viewer(self, mode):
        """Get or create viewer for rendering."""
        self.viewer = self._viewers.get(mode)
        
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer_setup()
            elif mode == "rgb_array":
                self.viewer = mujoco.Renderer(self.model, height=DEFAULT_SIZE, width=DEFAULT_SIZE)
                self._viewer_setup()
            
            self._viewers[mode] = self.viewer
        
        return self.viewer
    
    def _get_mjstate(self):
        """Get current MuJoCo state for copying."""
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'ctrl': self.data.ctrl.copy(),
            'time': self.data.time
        }
    
    def _set_mjstate(self, state):
        """Set MuJoCo state from saved state."""
        self.data.qpos[:] = state['qpos']
        self.data.qvel[:] = state['qvel']
        self.data.ctrl[:] = state['ctrl']
        self.data.time = state['time']
        mujoco.mj_forward(self.model, self.data)
    
    def _reset_sim(self):
        """Reset simulation to initial state."""
        self._set_mjstate(self.initial_state)
        return True
    
    # Abstract methods to be implemented by subclasses
    def _set_action_space(self):
        """Define action space."""
        raise NotImplementedError()
    
    def _set_observation_space(self, obs):
        """Define observation space."""
        raise NotImplementedError()
    
    def _get_obs(self):
        """Get current observation."""
        raise NotImplementedError()
    
    def _set_action(self, action):
        """Apply action to simulation."""
        raise NotImplementedError()
    
    def compute_reward(self, obs, info, terminated, truncated):
        """Compute reward for current step."""
        raise NotImplementedError()
    
    def _compute_info(self, obs, action):
        """Compute info dictionary."""
        raise NotImplementedError()
    
    def _check_termination(self, obs, info):
        """Check if episode should terminate."""
        raise NotImplementedError()
    
    def _sample_goal(self):
        """Sample a new goal."""
        raise NotImplementedError()
    
    def _env_setup(self, initial_qpos):
        """Setup initial environment configuration."""
        pass
    
    def _viewer_setup(self):
        """Setup viewer camera and settings."""
        pass
    
    def _step_callback(self):
        """Custom callback after each simulation step."""
        pass
    
    def _setup_references(self):
        """Setup references to bodies, actuators, etc. Override in subclass."""
        pass


class RocketEnv(RobotEnv):
    """Base class for rocket landing environments."""
    
    def __init__(
        self,
        model_path,
        n_substeps,
        initial_qpos,
        reward_params,
        distance_threshold=10.0,
        velocity_threshold=5.0,
        tilt_threshold=8.0,
        max_steps=1000,
    ):
        """
        Initialize rocket environment.
        
        Args:
            model_path: Path to MuJoCo XML file
            n_substeps: Simulation substeps per step
            initial_qpos: Initial configuration
            reward_params: Parameters for reward function
            distance_threshold: Max lateral distance for success (m)
            velocity_threshold: Max velocity for success (m/s)
            tilt_threshold: Max tilt angle for success (degrees)
            max_steps: Maximum episode length
        """
        # Physics constants
        self.START_FUEL = 200_000.0
        self.DRY_MASS = 200_000.0
        self.ISP = 400.0
        self.G0 = 1.62
        self.MAX_THRUST = 25_000_000.0
        self.INITIAL_INERTIA = np.array([1.2e9, 1.2e9, 3.0e7])
        self.INITIAL_MASS = self.DRY_MASS + self.START_FUEL
        
        # Thresholds
        self.distance_threshold = distance_threshold
        self.velocity_threshold = velocity_threshold
        self.tilt_threshold = tilt_threshold
        self.max_steps = max_steps
        
        # Initialize parent (this will call _setup_references)
        super(RocketEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos,
        )
        
        # Reward system
        self.rewarder = RocketReward(reward_params)
    
    def _setup_references(self):
        """Setup references to bodies and actuators."""
        # Get body and actuator IDs
        self.rocket_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.act_thrust = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
        self.act_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
        self.act_pitch = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
    
    def _set_action_space(self):
        """Define action space: [thrust, yaw, pitch]."""
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )
        return self.action_space
    
    def _set_observation_space(self, obs):
        """Define observation space."""
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32
        )
    
    def _set_action(self, action):
        """
        Apply control action to rocket.
        
        Args:
            action: [thrust_cmd, yaw_cmd, pitch_cmd] in range [-1, 1]
        """
        assert action.shape == (3,), f"Expected action shape (3,), got {action.shape}"
        
        # Map thrust from [-1, 1] to [0, MAX_THRUST]
        ctrl_thrust = (action[0] + 1) / 2.0 * self.MAX_THRUST
        
        # No thrust if fuel depleted
        if self.fuel_mass <= 0:
            ctrl_thrust = 0.0
        
        # Apply controls
        self.data.ctrl[self.act_thrust] = ctrl_thrust
        self.data.ctrl[self.act_yaw] = action[1]
        self.data.ctrl[self.act_pitch] = action[2]
        
        # Update rocket dynamics
        self._update_rocket_dynamics(ctrl_thrust)
        
        # Step simulation
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
    
    def _update_rocket_dynamics(self, ctrl_thrust):
        """Update mass and inertia based on fuel consumption."""
        dt = self.model.opt.timestep
        
        if ctrl_thrust > 0 and self.fuel_mass > 0:
            # Compute fuel burn
            mdot = ctrl_thrust / (self.ISP * self.G0)
            burn = mdot * dt
            self.fuel_mass = max(0, self.fuel_mass - burn)
            
            # Update mass
            current_mass = self.DRY_MASS + self.fuel_mass
            self.model.body_mass[self.rocket_body] = current_mass
            
            # Update inertia (linear scaling)
            mass_ratio = current_mass / self.INITIAL_MASS
            self.model.body_inertia[self.rocket_body] = self.INITIAL_INERTIA * mass_ratio
    
    def _get_obs(self):
        """Get observation: [pos(3), vel(3), quat(4), ang_vel(3), fuel_ratio(1)]."""
        pos = self.data.xpos[self.rocket_body].copy()
        vel = self.data.qvel[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()
        fuel_ratio = self.fuel_mass / self.START_FUEL
        
        return np.concatenate([pos, vel, quat, ang_vel, [fuel_ratio]], dtype=np.float32)
    
    def _get_state_dict(self):
        """Extract state dictionary for reward computation."""
        pos = self.data.xpos[self.rocket_body].copy()
        vel = self.data.qvel[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        
        # Compute tilt angle
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        z_axis = r.apply([0, 0, 1])
        tilt = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
        
        return {
            'pos': pos,
            'vel': vel,
            'alt': pos[2],
            'lateral_dist': np.linalg.norm(pos[0:2]),
            'tilt': tilt,
            'fuel_mass': self.fuel_mass
        }
    
    def compute_reward(self, obs, info, terminated, truncated):
        """Compute reward using external reward function."""
        state_dict = self._get_state_dict()
        reward, r_info = self.rewarder.compute(
            state_dict, 
            info['action'], 
            terminated, 
            truncated, 
            info['is_success']
        )
        info.update(r_info)
        return reward
    
    def _compute_info(self, obs, action):
        """Compute info dictionary."""
        state_dict = self._get_state_dict()
        
        # Check success conditions
        is_upright = abs(state_dict['tilt']) < self.tilt_threshold
        is_slow = np.linalg.norm(state_dict['vel']) < self.velocity_threshold
        is_close = state_dict['lateral_dist'] < self.distance_threshold
        
        return {
            'is_success': is_upright and is_slow and is_close,
            'fuel_remaining': self.fuel_mass,
            'action': action,
            'altitude': state_dict['alt'],
            'lateral_distance': state_dict['lateral_dist'],
            'tilt_angle': state_dict['tilt'],
        }
    
    def _check_termination(self, obs, info):
        """Check termination and truncation conditions."""
        state_dict = self._get_state_dict()
        
        # Terminated if landed (low altitude)
        terminated = state_dict['alt'] < 50.0
        
        # Truncated if out of bounds or max steps
        truncated = False
        if state_dict['lateral_dist'] > 3000.0 or state_dict['alt'] > 4000.0:
            truncated = True
        if self.step_count >= self.max_steps:
            truncated = True
        
        return terminated, truncated
    
    def _sample_goal(self):
        """Sample target landing location (origin)."""
        return np.array([0.0, 0.0, 0.0])
    
    def _env_setup(self, initial_qpos):
        """Setup initial environment state."""
        # Initialize fuel
        self.fuel_mass = self.START_FUEL
        self.step_count = 0
        
        # Set initial state with randomization
        self._set_initial_state()
    
    def _set_initial_state(self):
        """Set randomized initial state."""
        # Position with noise
        x = 500.0 + self.rng.uniform(-20.0, 20.0)
        y = 0.0
        alt = 500.0 + self.rng.uniform(-50.0, 50.0)
        
        # Velocity
        vel = np.array([10.0, 0.0, -50.0])
        
        # Orientation (point toward target)
        yaw = np.arctan2(-y, -x)
        target_pitch_deg = 90.0  # Horizontal
        
        quat = self._compute_orientation_quat(yaw, target_pitch_deg)
        
        # Apply to data
        self.data.qpos[0:3] = [x, y, alt]
        self.data.qpos[3:7] = quat
        self.data.qvel[0:3] = vel
        self.data.qvel[3:6] = [0, 0, 0]
        
        mujoco.mj_forward(self.model, self.data)
    
    def _compute_orientation_quat(self, yaw, pitch_deg):
        """Compute quaternion from yaw and pitch angles."""
        rad_pitch = np.radians(pitch_deg)
        local_z = np.cos(rad_pitch)
        local_fwd = np.sin(rad_pitch)
        
        # Convert to global frame
        vec_x = local_fwd * np.cos(yaw)
        vec_y = local_fwd * np.sin(yaw)
        vec_z = local_z
        
        target_nose_vec = np.array([vec_x, vec_y, vec_z])
        target_nose_vec /= np.linalg.norm(target_nose_vec)
        
        # Compute quaternion to align Z-axis with target
        base_vec = np.array([0, 0, 1])
        cross_axis = np.cross(base_vec, target_nose_vec)
        dot_val = np.dot(base_vec, target_nose_vec)
        
        if np.linalg.norm(cross_axis) < 1e-6:
            if dot_val < 0:
                quat = [0, 1, 0, 0]  # 180 deg around X
            else:
                quat = [1, 0, 0, 0]  # Identity
        else:
            cross_axis /= np.linalg.norm(cross_axis)
            half_angle = np.arccos(np.clip(dot_val, -1, 1)) / 2.0
            
            w = np.cos(half_angle)
            xyz = cross_axis * np.sin(half_angle)
            quat = [w, xyz[0], xyz[1], xyz[2]]
        
        return quat
    
    def _viewer_setup(self):
        """Setup viewer camera."""
        if hasattr(self.viewer, 'cam'):
            self.viewer.cam.distance = 1000.0
            self.viewer.cam.azimuth = 45.0
            self.viewer.cam.elevation = -30.0
    
    def _reset_sim(self):
        """Reset simulation state."""
        self._set_mjstate(self.initial_state)
        self.fuel_mass = self.START_FUEL
        self.step_count = 0
        return True


class RocketLandingEnv(RocketEnv, gym.utils.EzPickle):
    """Rocket landing environment with default parameters."""
    
    def __init__(self, reward_type="dense", render_mode=None):
        """
        Initialize rocket landing environment.
        
        Args:
            reward_type: Type of reward ('sparse' or 'dense')
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        initial_qpos = {}  # No joint positions needed for floating body
        
        reward_params = {
            'start_fuel': 200_000.0,
            'reward_type': reward_type
        }
        
        RocketEnv.__init__(
            self,
            MODEL_XML_PATH,
            n_substeps=20,
            initial_qpos=initial_qpos,
            reward_params=reward_params,
            distance_threshold=10.0,
            velocity_threshold=5.0,
            tilt_threshold=8.0,
            max_steps=1000,
        )
        gym.utils.EzPickle.__init__(self, reward_type, render_mode)
        
        self.render_mode = render_mode