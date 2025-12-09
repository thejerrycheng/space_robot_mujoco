import os
import time
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer

# =========================================================================
# 1. ENVIRONMENT DEFINITION (Copied from your code)
# =========================================================================

# Helper to normalize angles
def normalize_angle(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, model_path="assets/mjcf/realistic_param.xml"):
        super().__init__()

        # Check path
        if not os.path.exists(model_path):
             print(f"❌ ERROR: Model not found at {model_path}")
             # You might want to try an absolute path or raise an error here
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.rocket_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.qpos_adr = self.model.jnt_qposadr[self.free_joint_id]
        self.qvel_adr = self.model.jnt_dofadr[self.free_joint_id]

        self.yaw_act   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
        self.pitch_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
        self.thrust_act= mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")

        self.DRY_MASS = 500_000.0 
        self.START_FUEL = 4_500_000.0 
        self.ISP = 400.0
        self.G0 = 1.62 
        self.DT = self.model.opt.timestep

        self.MAX_THRUST = 3.0 * (self.DRY_MASS + self.START_FUEL) * self.G0
        self.MAX_GIMBAL = np.deg2rad(30.0)

        self.TARGET_Z = 60.0 
        self.MAX_STEPS = 200 
        self.MAX_LATERAL_DIST = 3000.0 
        self.MAX_VELOCITY = 1000.0      
        self.LANDING_TOLERANCE = 10   

        self.curriculum_level = 0
        self.max_curriculum_level = 10 
        
        self.curriculum_params = {
            "initial_altitude_offset": (1000, 10.0, 2000.0),
            "lateral_offset":       (500.0, 1.0, 1000.0),
            "initial_velocity_std": (300.0, 10.0, 500.0), 
            "initial_tilt_deg":     (0.0, 1.0, 5.0),
        }
        
        self.success_history = []
        obs_high = np.inf
        self.observation_space = spaces.Box(-obs_high, obs_high, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.fuel_mass = self.START_FUEL
        self.total_mass = self.DRY_MASS + self.START_FUEL
        self.orig_inertia = self.model.body_inertia[self.rocket_bid].copy()
        
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0
        self.touchdown_steps = 0 

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1, 1)

        thrust_cmd = (action[0] + 1) * 0.5 * self.MAX_THRUST
        yaw_cmd    = action[1] * self.MAX_GIMBAL
        pitch_cmd  = action[2] * self.MAX_GIMBAL

        if self.fuel_mass > 0:
            mdot = thrust_cmd / (self.ISP * self.G0)
            consumed = mdot * self.DT
            if consumed > self.fuel_mass:
                ratio = self.fuel_mass / consumed
                thrust_cmd *= ratio
                self.fuel_mass = 0
            else:
                self.fuel_mass -= consumed
            self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.fuel_mass
        else:
            thrust_cmd = 0

        self.data.ctrl[self.thrust_act] = thrust_cmd
        self.data.ctrl[self.yaw_act]    = yaw_cmd
        self.data.ctrl[self.pitch_act]  = pitch_cmd

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        state_metrics = self._get_state_metrics()
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

    def _compute_reward(self, m, thrust, terminated, success):
        # ... (Simplified for brevity, strictly used for calculation)
        return 0.0, {}

    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        if m["lateral_dist"] > self.MAX_LATERAL_DIST:
            terminated = True
        
        if m["z"] < (self.TARGET_Z + 0.5): 
            terminated = True 
            # Note: We are just testing free fall crashes, 
            # so we assume ground contact terminates immediately.

        if self.step_count >= self.MAX_STEPS:
            truncated = True

        return terminated, truncated, success

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

        r = np.sqrt(np.random.uniform(0, lat_offset**2))
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = self.TARGET_Z + alt_offset
        
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = [x, y, z]

        if tilt_d > 0:
            tilt_rad = np.deg2rad(np.random.uniform(0, tilt_d))
            axis_angle = np.random.uniform(0, 2*np.pi)
            ax = np.cos(axis_angle) * np.sin(tilt_rad/2)
            ay = np.sin(axis_angle) * np.sin(tilt_rad/2)
            az = 0
            w  = np.cos(tilt_rad/2)
            self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = [w, ax, ay, az]
        else:
            self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = [1, 0, 0, 0]

        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = np.random.normal(0, vel_s, 3)
        self.data.qvel[self.qvel_adr+2] -= 2.0 

    def _get_obs(self): return np.zeros(20) # Dummy for viz
    def _get_state_metrics(self):
        pos = self._get_pos()
        vel = self._get_vel()
        quat = self._get_quat()
        return {
            "pos": pos, "vel": vel, "z": pos[2], "vz": vel[2],
            "quat_w": quat[0], "lateral_dist": np.linalg.norm(pos[:2]),
            "pos_err": 0, "vel_err": 0, "tilt": 0
        }
    def _curriculum_interp(self, name):
        start, step, limit = self.curriculum_params[name]
        val = start + self.curriculum_level * step
        return min(val, limit)
    def update_curriculum(self, success): pass
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
                # Set cool camera view
                self.viewer.cam.distance = 2500
                self.viewer.cam.azimuth = 90
                self.viewer.cam.elevation = -10
                self.viewer.cam.lookat[:] = [0, 0, 500]
            self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()

# =========================================================================
# 2. TESTING LOGIC (Free Fall)
# =========================================================================

def run_free_fall_tests():
    print("\n🚀 STARTING FREE FALL TESTS (10 SAMPLES)")
    
    # 1. Initialize Env
    # Ensure this script is run from the folder containing 'assets'
    env = RocketLandingEnv(render_mode="human", model_path="assets/mjcf/realistic_param.xml")
    
    # 2. Run 10 Episodes
    for i in range(1, 11):
        print(f"\n--- Sample {i} ---")
        obs, _ = env.reset()
        
        # Log Initial Conditions
        start_pos = env.data.xpos[env.rocket_bid]
        start_vel = env.data.qvel[env.qvel_adr:env.qvel_adr+3]
        print(f"Start Pos:  [{start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f}]")
        print(f"Start Vel:  [{start_vel[0]:.1f}, {start_vel[1]:.1f}, {start_vel[2]:.1f}]")
        
        step = 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # 3. ACTION: FREE FALL
            # In your step function: thrust_cmd = (action[0] + 1) * 0.5 * MAX
            # To get 0 thrust, action[0] must be -1.0
            action = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step += 1
            
            # Simple sleep to make it watchable (since there's no heavy computation)
            time.sleep(1/60.0)

        # Log Final Status
        end_pos = env.data.xpos[env.rocket_bid]
        print(f"Crashed at: [{end_pos[0]:.1f}, {end_pos[1]:.1f}, {end_pos[2]:.1f}] after {step} steps")
        
        # Brief pause between samples
        time.sleep(1.0)

    env.close()

if __name__ == "__main__":
    run_free_fall_tests()