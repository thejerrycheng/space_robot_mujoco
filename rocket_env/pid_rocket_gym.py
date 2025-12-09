import numpy as np
import mujoco
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

# Assume the user's class is in a file named `rocket_env.py`
# or simply paste the user's class definition above this code block.
# from rocket_env import RocketLandingEnv 

class RocketPID:
    def __init__(self, target_alt=100.0):
        # --- TUNING PARAMETERS ---
        
        # Attitude Gains (PD Controller)
        # High inertia (1.2e9) requires substantial P gains to react.
        self.kp_att = 8.0   # Proportional term for angle
        self.kd_att = 15.0  # Derivative term for angular velocity

        # Altitude/Throttle Gains (PID Controller)
        # We need to counteract massive gravity (Mass * 1.62)
        self.kp_alt = 1.0
        self.ki_alt = 0.05
        self.kd_alt = 2.0
        
        self.target_alt = target_alt
        self.integral_error_alt = 0.0
        
    def compute_action(self, obs, info):
        """
        Inputs:
            obs: [pos(3), vel(3), quat(4), ang_vel(3), fuel(1)]
            info: dictionary containing easier-to-access state data
        Returns:
            action: [thrust, yaw_servo, pitch_servo] (all -1 to 1)
        """
        
        # --- 1. EXTRACT STATE ---
        # While obs is available, it's easier to use the info dict 
        # provided by the env or recalculate from obs for clarity.
        
        # Quaternion comes as [w, x, y, z] in MuJoCo obs, but Scipy needs [x, y, z, w]
        # Your env code does: np.concatenate([pos, vel, quat, ...])
        # So obs[6:10] is [w, x, y, z]
        mj_quat = obs[6:10] 
        scipy_quat = [mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]]
        
        r = R.from_quat(scipy_quat)
        euler_angles = r.as_euler('xyz', degrees=False) # [Roll, Pitch, Yaw]
        
        # Current Angles (deviations from vertical)
        # Note: Depending on your model alignment, X and Y are usually the tilt axes
        current_angle_x = euler_angles[0]
        current_angle_y = euler_angles[1]
        
        # Angular Velocities
        ang_vel = obs[10:13]
        av_x = ang_vel[0]
        av_y = ang_vel[1]
        
        # Altitude and Vertical Velocity
        z_pos = obs[2]
        z_vel = obs[5]

        # --- 2. ATTITUDE CONTROL (GIMBAL) ---
        # Goal: Angle = 0, AngVel = 0
        # Formula: u = - (Kp * angle + Kd * ang_vel)
        # We invert the signal because if we tilt Right (+X), we want torque Left (-X).
        # Based on your XML:
        # 'yaw_servo' acts on 'thruster_yaw' (X-axis hinge)
        # 'pitch_servo' acts on 'thruster_pitch' (Y-axis hinge)
        
        # Calculate PD output
        ctrl_x = - (self.kp_att * current_angle_x + self.kd_att * av_x)
        ctrl_y = - (self.kp_att * current_angle_y + self.kd_att * av_y)
        
        # Clip to actuator range [-1, 1]
        act_yaw = np.clip(ctrl_x, -1.0, 1.0)
        act_pitch = np.clip(ctrl_y, -1.0, 1.0)

        # --- 3. ALTITUDE CONTROL (THROTTLE) ---
        # Goal: Maintain slow descent or hover
        # A simple hover throttle is roughly: Mass * Gravity / Max_Thrust
        # But Mass changes as fuel burns, so PID is robust here.
        
        alt_error = self.target_alt - z_pos
        self.integral_error_alt += alt_error * 0.01 # approximate dt
        
        # Basic Hover Feedforward (approximate) to help PID start near correct value
        # 500t dry + fuel roughly 50% = 2,500,000kg approx
        # G = 1.62. Weight ~ 4,000,000 N. Max Thrust = 25,000,000.
        # Feedforward ~ 0.16 (mapping -1 to 1 is tricky, gym expects -1 to 1)
        # Let's rely on PID.
        
        throttle_signal = (self.kp_alt * alt_error) + \
                          (self.ki_alt * self.integral_error_alt) + \
                          (self.kd_alt * (0 - z_vel))
        
        # We want to hover, so we bias the throttle up.
        # Since action space is [-1, 1], where -1 is 0 thrust and 1 is Max:
        # We shift the signal.
        act_thrust = np.clip(throttle_signal, -1.0, 1.0)
        
        # Safety: If we are way too high, cut engine, if too low, max burn
        if z_pos > 3000: act_thrust = -1.0
        
        return np.array([act_thrust, act_yaw, act_pitch], dtype=np.float32)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Initialize Environment
    # Ensure "assets/mjcf/realistic_param.xml" exists or update path
    env = RocketLandingEnv(render_mode="human")
    
    # Initialize Controller
    controller = RocketPID(target_alt=150.0)
    
    obs, _ = env.reset(seed=42)
    
    # --- OVERRIDE INITIAL STATE ---
    # The default reset puts the rocket at 90 degrees (Horizontal).
    # We must set it upright to test the PID, or it will crash immediately.
    print("Overriding initial state to UPRIGHT...")
    env.data.qpos[3:7] = [1, 0, 0, 0] # Quaternion identity (Upright)
    env.data.qvel[0:3] = [0, 0, -10]  # Small downward velocity
    mujoco.mj_forward(env.model, env.data)
    
    for i in range(1000):
        # 1. Get Action from Controller
        # We need to reconstruct the state_dict logic or pass obs
        # For simplicity, we pass obs.
        action = controller.compute_action(obs, {})
        
        # 2. Step Environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. Render
        env.render()
        
        # Print telemetry every 10 frames
        if i % 10 == 0:
            z = obs[2]
            tilt = info['tilt']
            print(f"Step {i} | Alt: {z:.2f}m | Tilt: {tilt:.2f} deg | ThrustCmd: {action[0]:.2f}")

        if terminated or truncated:
            print("Episode Finished.")
            break

    env.close()