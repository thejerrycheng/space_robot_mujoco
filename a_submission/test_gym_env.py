import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# 1. Import your Environment
# (Ensure this matches your actual file name)
from rocket_env.rocket_3_env import RocketLandingEnv

# 2. Import your Plotting Utility
from utils.plotting import generate_interactive_plot, Col

def get_euler_from_quat(quat):
    """ 
    Helper to convert [w, x, y, z] to Euler angles (degrees).
    Returns [Roll, Pitch, Yaw] (approximate for analysis)
    """
    # Scipy expects [x, y, z, w]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    angles = r.as_euler('xyz', degrees=True)
    return angles

def run_test(num_episodes):
    # Setup Environment (No internal rendering to avoid conflicts)
    env = RocketLandingEnv(render_mode=None)
    
    # Container for all episodes
    all_histories = []

    print(f"{Col.BOLD}🚀 STARTING TEST: {num_episodes} Episodes{Col.RESET}")
    print(f"   (Data will be saved to 'logs/' for interactive plotting)")
    
    # Launch Viewer (Single Window Pattern)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        
        # --- Camera Setup ---
        viewer.cam.lookat[:] = [0, 0, 500] 
        viewer.cam.distance = 3000.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0
        
        for ep in range(num_episodes):
            if not viewer.is_running(): break
            
            print(f"▶️  Episode {ep+1}/{num_episodes} ...", end=" ", flush=True)
            
            # Reset
            obs, _ = env.reset()
            
            # Initialize Episode History
            history = {
                'time': [],
                'pos': [],
                'vel': [],
                'quat': [],      # CRITICAL for Orientation Cones
                'attitude': [],  # Used for 2D tilt graphs
                'thrust': [],
                'gimbal': [],
                'mass': []
            }
            
            terminated = False
            truncated = False
            step = 0
            
            while viewer.is_running() and not (terminated or truncated):
                step_start = time.time()
                
                # --- ACTION (Simple Test Policy) ---
                # Full Throttle with slight random gimbal noise
                thrust_cmd = 1.0
                yaw_cmd = np.random.uniform(-0.1, 0.1)
                pitch_cmd = np.random.uniform(-0.1, 0.1)
                action = np.array([thrust_cmd, yaw_cmd, pitch_cmd])
                
                # --- STEP ---
                obs, reward, terminated, truncated, info = env.step(action)
                
                # --- COLLECT DATA ---
                # 1. Physics State
                # Use .copy() to ensure we store the value at this specific timestep
                pos = env.data.xpos[env.rocket_body].copy()
                vel = env.data.qvel[0:3].copy()
                quat = env.data.qpos[3:7].copy() # [w, x, y, z]
                mass = env.model.body_mass[env.rocket_body]
                
                # 2. Derived State
                euler = get_euler_from_quat(quat)
                
                # 3. Store
                history['time'].append(env.data.time)
                history['pos'].append(pos)
                history['vel'].append(vel)
                history['quat'].append(quat) # Stores Orientation
                history['attitude'].append(euler)
                history['thrust'].append(thrust_cmd * env.MAX_THRUST)
                history['gimbal'].append([np.degrees(yaw_cmd), np.degrees(pitch_cmd)])
                history['mass'].append(mass)

                # --- SYNC ---
                viewer.sync()
                step += 1
                
                # FPS Limiter (Optional, remove for max speed)
                time_until_next = env.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

            # Store Episode
            all_histories.append(history)
            
            # Print Outcome
            outcome = info.get('outcome', 'Unknown')
            color = Col.GREEN if info.get('is_success') else Col.RED
            print(f"{color}{outcome}{Col.RESET} ({step} steps)")
            
            # Brief pause between episodes
            time.sleep(0.5)

    # --- GENERATE PLOT ---
    # This function looks for the 'pos' and 'quat' keys to draw the path and cones
    generate_interactive_plot(all_histories, save_dir="logs", env_name="Rocket_V3")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
    args = parser.parse_args()
    
    run_test(args.episodes)