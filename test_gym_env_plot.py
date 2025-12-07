import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import importlib
import os
import sys

# 2. Import your Plotting Utility
from utils.plotting import generate_interactive_plot, Col

# ================================================================
#   UTILITIES (Remains the same)
# ================================================================

def get_euler_from_quat(quat):
    """ 
    Helper to convert [w, x, y, z] to Euler angles (degrees).
    """
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    angles = r.as_euler('xyz', degrees=True)
    return angles

def load_environment_class(env_name):
    """ Dynamically loads the environment class from rocket_env/{env_name}.py """
    try:
        module_path = f"rocket_env.{env_name}"
        mod = importlib.import_module(module_path)
        
        for attr_name in dir(mod):
            if attr_name.endswith('Env') and attr_name != 'gym':
                return getattr(mod, attr_name)
        
        raise AttributeError(f"Could not find a class ending in 'Env' in module {module_path}")
        
    except ImportError:
        print(f"\n{Col.RED}❌ Error: Could not import module 'rocket_env.{env_name}'.")
        print("   Ensure the file exists and is on the PYTHONPATH.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Col.RED}❌ Error loading environment: {e}{Col.RESET}")
        sys.exit(1)

# ... (UTILITIES section remains the same)

# ================================================================
#   MAIN RUN LOGIC
# ================================================================

def run_test(env_class, num_episodes, camera_mode):
    # Determine the name for logging/plotting
    env_name = env_class.__name__
    
    # Setup Environment 
    env = env_class(render_mode=None)
    
    all_histories = []

    print(f"{Col.BOLD}🚀 STARTING TEST: {env_name} ({num_episodes} Episodes){Col.RESET}")
    print(f"   (Camera mode: {camera_mode})")
    
    # Launch Viewer (Single Window Pattern)
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            
            # --- Camera Setup (Executed ONCE before the episode loop) ---
            rocket_body_id = env.rocket_body # Get the body ID once

            if camera_mode == 'rocket':
                # Set tracking-specific view parameters ONCE.
                viewer.cam.distance = 150.0 
                viewer.cam.azimuth = 135.0   
                viewer.cam.elevation = -30.0
                viewer.cam.trackbodyid = -1 # Initial setup: disabled until after reset
                print(f"{Col.YELLOW}   Camera set to TRACKING mode (Rocket).{Col.RESET}")
                
            elif camera_mode == 'default':
                # FIXED World View
                viewer.cam.trackbodyid = -1 # Ensure tracking is disabled
                viewer.cam.lookat[:] = [0, 0, 500] 
                viewer.cam.distance = 3000.0
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -20.0
                print(f"{Col.YELLOW}   Camera set to FIXED mode (World View).{Col.RESET}")
            
            else:
                print(f"{Col.RED}   Invalid camera mode '{camera_mode}'. Using default view.{Col.RESET}")
                viewer.cam.trackbodyid = -1


            for ep in range(num_episodes):
                if not viewer.is_running(): break
                
                print(f"▶️  Episode {ep+1}/{num_episodes} ...", end=" ", flush=True)
                
                # Reset
                obs, _ = env.reset()
                
                # --- Apply/Re-apply tracking immediately after reset ---
                if camera_mode == 'rocket':
                    # Lock tracking onto the rocket's new initial position
                    viewer.cam.trackbodyid = rocket_body_id 
                
                viewer.sync() # Sync viewer to the new position/tracking target
                
                # Initialize Episode History
                history = {
                    'time': [], 'pos': [], 'vel': [], 'quat': [], 
                    'attitude': [], 'thrust': [], 'gimbal': [], 'mass': []
                }
                
                terminated = False
                truncated = False
                step = 0
                
                while viewer.is_running() and not (terminated or truncated):
                    step_start = time.time()
                    
                    # --- ACTION (Simple Test Policy: Full Throttle) ---
                    thrust_cmd = 0.065  # 10% Throttle
                    yaw_cmd = 0.0
                    pitch_cmd = 0.0   
                    action = np.array([thrust_cmd, yaw_cmd, pitch_cmd])
                    
                    # --- STEP ---
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # --- COLLECT & LOG DATA ---
                    
                    pos = env.data.xpos[env.rocket_body].copy()
                    vel = env.data.qvel[0:3].copy()
                    quat = env.data.qpos[3:7].copy() 
                    mass = env.model.body_mass[env.rocket_body]
                    euler = get_euler_from_quat(quat)
                    roll, pitch, yaw = euler
                    dist_xy = np.sqrt(pos[0]**2 + pos[1]**2)
                    max_tilt = max(abs(pitch), abs(roll))
                    state_str = f"Alt:{pos[2]:5.1f}m Dis:{dist_xy:5.1f}m Vz:{vel[2]:5.1f} Tlt:{max_tilt:4.1f}° Mass:{mass:5.1f}kg"
                    ctrl_str = f"Thr:{thrust_cmd * env.MAX_THRUST:6.0f}N GimY:{np.degrees(yaw_cmd):4.1f}°"
                    log_line = (
                        f"\r{step:04} | {Col.CYAN}{state_str}{Col.RESET} | "
                        f"{Col.YELLOW}{ctrl_str}{Col.RESET} | {Col.GREEN}Rew:{reward:6.2f}{Col.RESET} \033[K"
                    )
                    sys.stdout.write(log_line)
                    sys.stdout.flush()
                    history['time'].append(env.data.time)
                    history['pos'].append(pos)
                    history['vel'].append(vel)
                    history['quat'].append(quat) 
                    history['attitude'].append(euler)
                    history['thrust'].append(thrust_cmd * env.MAX_THRUST)
                    history['gimbal'].append([np.degrees(yaw_cmd), np.degrees(pitch_cmd)])
                    history['mass'].append(mass)

                    # --- SYNC (Guaranteed Tracking) ---
                    if camera_mode == 'rocket':
                        # CRITICAL FIX 3: Re-set the ID every frame to maintain the lock.
                        # This combats the viewer potentially resetting the property after a physics step.
                        viewer.cam.trackbodyid = rocket_body_id 
                    
                    viewer.sync() 
                    step += 1
                    
                    # FPS Limiter (Optional)
                    time_until_next = env.model.opt.timestep - (time.time() - step_start)
                    if time_until_next > 0:
                        time.sleep(time_until_next)

                    if terminated or truncated:
                        print("\n" + log_line.strip() + f" | Episode terminated.")
                        time.sleep(1)

                # Store Episode History
                all_histories.append(history)
                
                # Print Outcome
                outcome = info.get('outcome', 'Unknown')
                color = Col.GREEN if info.get('is_success') else Col.RED
                
                print(f"{color}🏁 {outcome}{Col.RESET} ({step} steps)")
            
    except Exception as e:
        print(f"\n{Col.RED}❌ MuJoCo/Runtime Error: {e}{Col.RESET}")
        
    finally:
        env.close()

    # --- GENERATE PLOT ---
    if all_histories:
        generate_interactive_plot(all_histories, save_dir="logs", env_name=env_name)
    else:
        print(f"{Col.YELLOW}⚠️ No trajectories to plot.{Col.RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', 
        type=str, 
        default='rocket_gym_env', 
        help='Name of the environment file in rocket_env/'
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=5, 
        help='Number of episodes'
    )
    # --- NEW ARGUMENT FOR CAMERA SELECTION ---
    parser.add_argument(
        '--camera',
        type=str,
        default='rocket',
        choices=['default', 'rocket'],
        help='Camera view mode: "default" (fixed world view) or "rocket" (tracking view).'
    )
    # ----------------------------------------
    args = parser.parse_args()
    
    # 1. Dynamically Load the Class
    RocketEnvClass = load_environment_class(args.env)
    
    # 2. Run the Test with the Loaded Class and Camera Mode
    run_test(RocketEnvClass, args.episodes, args.camera)