import argparse
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
import os
import mujoco

# --- IMPORTANT ---
# For the testing script to run correctly, we must define the environment class
# here or import it. Since you provided the full class, we'll place it here
# to create a single runnable file.

# --- Assume 'rewards/landing_reward.py' exists and RocketReward works ---
# Since I don't have the actual content of RocketReward, I'll use a mock 
# for a fully runnable script. You should ensure your real RocketReward 
# class is importable or defined.

class MockRocketReward:
    def __init__(self, config):
        pass
    def compute(self, state_dict, action, terminated, truncated, success):
        # Simple placeholder reward logic
        reward = 1.0
        if success: reward += 1000.0
        if terminated: reward -= 500.0
        if action[0] > 0: reward -= 0.1 # Penalty for continuous thrust burn
        
        r_info = {'reward_components': 'mocked'}
        return reward, r_info

# Replace this line with your actual import:
# from rewards.landing_reward import RocketReward as RewarderClass
RewarderClass = MockRocketReward 
# ------------------------------------------------------------------------


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # ==========================================================
    # 🌟 CONSTANTS DEFINITION 🌟
    # ==========================================================
    # Physics Constants
    START_FUEL = 200_000.0  
    DRY_MASS   = 200_000.0    
    ISP        = 400.0        
    G0         = 1.62         
    MAX_THRUST = 25_000_000.0 
    INITIAL_INERTIA = np.array([1.2e9, 1.2e9, 3.0e7])
    INITIAL_MASS = DRY_MASS + START_FUEL
    
    # Environment Constraints
    MAX_STEPS = 1000 
    
    # Landing Success Criteria
    LANDING_TOLERANCE_POS = 10.0 
    LANDING_TOLERANCE_VEL = 5.0 
    LANDING_TOLERANCE_TILT = 8.0 

    # Initial Conditions (Fixed)
    INIT_ALTITUDE = 1000.0
    INIT_LATERAL_DIST = 10.0
    INIT_DOWNWARD_SPEED = 50.0
    INIT_OFFSET_DIST = 0

    INIT_NOISE_POS_XY = 20.0
    INIT_NOISE_POS_Z = 50.0
    INIT_NOISE_VEL_XY = 0.0
    
    # Discrete Action Space Configuration
    THRUST_LEVELS = [0.0, MAX_THRUST] 
    NUM_ANGLE_STATES = 7
    SERVO_CONTROL_VALUES = np.linspace(-1.0, 1.0, NUM_ANGLE_STATES)
    
    NUM_THRUST = len(THRUST_LEVELS)
    NUM_ACTIONS = NUM_THRUST * NUM_ANGLE_STATES * NUM_ANGLE_STATES

    # ==========================================================
    # ===== Initialization =====
    # ==========================================================

    def __init__(self, xml_path="assets/mjcf/realistic_param.xml", render_mode=None):
        super().__init__()
        
        # --- CONFIGURATION ---
        self.xml_path = xml_path
        if not os.path.exists(self.xml_path):
            # Try a common fallback path if running from a different directory
            fallback_path = os.path.join(os.path.dirname(__file__), self.xml_path)
            if os.path.exists(fallback_path):
                self.xml_path = fallback_path
            else:
                raise FileNotFoundError(f"Could not find XML at {self.xml_path}")
            
        self.render_mode = render_mode
        
        # --- REWARD SETUP ---
        # Use the actual reward class if imported, or the mock
        self.rewarder = RewarderClass({
            'start_fuel': self.START_FUEL
        })

        # --- MUJOCO SETUP ---
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.rocket_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.act_thrust  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
        self.act_yaw     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
        self.act_pitch   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
        
        # --- VIEWER INITIALIZATION ---
        self.viewer = None

        # --- SPACES ---
        self._setup_action_map()
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        
        high = np.inf
        self.observation_space = spaces.Box(low=-high, high=high, shape=(14,), dtype=np.float32)

    def _setup_action_map(self):
        """
        Creates a map from the single integer action index to the control values.
        """
        self.action_map = {}
        action_idx = 0
        N = self.NUM_ANGLE_STATES
        
        for thrust_idx in range(self.NUM_THRUST):
            for yaw_idx in range(N):
                for pitch_idx in range(N):
                    self.action_map[action_idx] = {
                        'thrust': self.THRUST_LEVELS[thrust_idx],
                        'yaw_ctrl': self.SERVO_CONTROL_VALUES[yaw_idx],
                        'pitch_ctrl': self.SERVO_CONTROL_VALUES[pitch_idx]
                    }
                    action_idx += 1
        
    # ===== Reset Function =====

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self._set_fixed_initial_state()
        
        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_body] = self.DRY_MASS + self.fuel_mass
        
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        
        return self._get_obs(), {}
    

    # ===== Step Function =====

    def step(self, action_index):
        self.step_count += 1
        
        action_values = self.action_map[action_index]
        applied_thrust, proxy_action = self._flight_control(action_values)
        self._rocket_dynamics(applied_thrust)

        obs = self._get_obs()
        state_dict = self._get_state_dict()

        terminated, success = self._check_termination(state_dict)
        
        truncated = False
        if state_dict['lateral_dist'] > 3000.0 or state_dict['alt'] > 4000.0:
            truncated = True
        if self.step_count >= self.MAX_STEPS:
            truncated = True

        reward, r_info = self.rewarder.compute(state_dict, proxy_action, terminated, truncated, success)

        info = {
            "is_success": success,
            "fuel_remaining": self.fuel_mass,
            **r_info
        }

        return obs, reward, terminated, truncated, info
    

    def _flight_control(self, action_values):
        ctrl_thrust = action_values['thrust']
        ctrl_yaw = action_values['yaw_ctrl']
        ctrl_pitch = action_values['pitch_ctrl']
        
        if self.fuel_mass <= 0: 
            ctrl_thrust = 0.0
            
        self.data.ctrl[self.act_thrust] = ctrl_thrust
        self.data.ctrl[self.act_yaw]    = ctrl_yaw
        self.data.ctrl[self.act_pitch]  = ctrl_pitch
        
        proxy_thrust = 2 * (ctrl_thrust / self.MAX_THRUST) - 1.0 
        proxy_action = np.array([proxy_thrust, ctrl_yaw, ctrl_pitch], dtype=np.float32)
        
        return ctrl_thrust, proxy_action

    def _rocket_dynamics(self, ctrl_thrust):
        dt = self.model.opt.timestep
        
        if ctrl_thrust > 0 and self.fuel_mass > 0:
            mdot = ctrl_thrust / (self.ISP * self.G0)
            burn = mdot * dt
            self.fuel_mass -= burn
            
            current_mass = self.DRY_MASS + self.fuel_mass
            self.model.body_mass[self.rocket_body] = current_mass
            
            mass_ratio = current_mass / self.INITIAL_MASS
            self.model.body_inertia[self.rocket_body] = self.INITIAL_INERTIA * mass_ratio
            
        mujoco.mj_step(self.model, self.data)

    def _check_termination(self, state):
        terminated = False
        success = False
        
        if state['alt'] < 50.0:
            terminated = True
            is_upright = abs(state['tilt']) < self.LANDING_TOLERANCE_TILT
            is_slow    = np.linalg.norm(state['vel']) < self.LANDING_TOLERANCE_VEL
            is_close   = state['lateral_dist'] < self.LANDING_TOLERANCE_POS
            
            if is_upright and is_slow and is_close:
                success = True
                
        return terminated, success

    def _set_fixed_initial_state(self):
        x = self.INIT_LATERAL_DIST + self.rng.uniform(-self.INIT_NOISE_POS_XY, self.INIT_NOISE_POS_XY)
        y = self.INIT_OFFSET_DIST + self.rng.uniform(-self.INIT_NOISE_POS_XY, self.INIT_NOISE_POS_XY)
        alt = self.INIT_ALTITUDE + self.rng.uniform(-self.INIT_NOISE_POS_Z, self.INIT_NOISE_POS_Z)
        
        vel_z = -self.INIT_DOWNWARD_SPEED 
        vel = np.array([0, 0, vel_z])
        
        yaw = np.arctan2(-y, -x) 
        target_pitch_from_up = 90.0
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

        self.data.qpos[0:3] = [x, y, alt]
        self.data.qpos[3:7] = quat 
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
        tilt = np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
        return {
            'pos': pos, 'vel': vel, 'alt': pos[2],
            'lateral_dist': np.linalg.norm(pos[0:2]),
            'tilt': tilt, 'fuel_mass': self.fuel_mass
        }

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()


# ==============================================================================
# 🚀 TESTING SCRIPT LOGIC (main function)
# ==============================================================================

def run_discrete_test(num_episodes, render_mode):
    """
    Tests the RocketLandingEnv with a random discrete agent.
    """
    
    # --- 1. Environment Setup ---
    # NOTE: Adjust the xml_path if "assets/mjcf/realistic_param.xml" is not correct
    try:
        env = RocketLandingEnv(
            xml_path="assets/mjcf/realistic_param.xml", 
            render_mode=render_mode
        )
    except FileNotFoundError as e:
        print(f"❌ Error setting up environment: {e}")
        print("Please check the 'xml_path' in the RocketLandingEnv initialization.")
        return
    
    # We cannot use env.spec.id because we instantiated the class directly.
    print("--- Testing Discrete RocketLandingEnv ---")
    print(f"Action Space: {env.action_space} (Total Actions: {env.action_space.n})")
    print(f"Observation Space: {env.observation_space.shape}")
    print("-" * 40)
    
    success_count = 0
    
    # --- 2. Episode Loop ---
    for episode in range(1, num_episodes + 1):
        # Reset the environment
        obs, info = env.reset(seed=int(time.time()))
        
        terminated = False
        truncated = False
        episode_reward = 0.0
        step_count = 0
        
        start_time = time.time()
        
        print(f"▶️  Episode {episode}/{num_episodes} started...")

        while not terminated and not truncated:
            # ⭐️ Correct way to sample a discrete action (single integer)
            action_index = env.action_space.sample() 
            
            # Step the environment with the single integer action
            obs, reward, terminated, truncated, info = env.step(action_index)
            
            episode_reward += reward
            step_count += 1
            
            if render_mode == "human":
                env.render()
                time.sleep(0.01)

        # --- 3. Episode End Logging ---
        duration = time.time() - start_time
        is_success = info.get("is_success", False)
        
        if is_success:
            success_count += 1
            status_emoji = "✅ SUCCESS"
        elif terminated:
            status_emoji = "💥 FAILED (Grounded)"
        else: # Truncated (due to timeout or bounds violation)
            status_emoji = "⏱️ TRUNCATED"
            
        print(f"   {status_emoji} | Steps: {step_count:4d} | Total Reward: {episode_reward:8.2f} | Fuel Left: {info['fuel_remaining']:.2f} kg | Time: {duration:.2f}s")
        
    # --- 4. Final Summary ---
    print("-" * 40)
    print(f"🏁 Test Finished. Total Episodes: {num_episodes}")
    print(f"   Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes:.1%})")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for the discrete RocketLandingEnv.")
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=5, 
        help="Number of episodes to run."
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Enable human rendering (GUI)."
    )
    
    args = parser.parse_args()
    
    render_mode = "human" if args.render else None
    
    # You can now save this entire file as test_gym_discrete.py and run it.
    run_discrete_test(args.episodes, render_mode)