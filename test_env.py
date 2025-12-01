import numpy as np
import time
import sys
import os
import argparse
import importlib
import mujoco
import matplotlib

# CRITICAL FIX: Use 'Agg' backend to prevent macOS main-thread crashes
# This must be set BEFORE importing pyplot
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================================================
#   UTILITIES: COLORS & MATH
# ================================================================
class Col:
    RESET = '\033[0m'
    CYAN = '\033[96m'   # For Physics State
    YELLOW = '\033[93m' # For Controls
    GREEN = '\033[92m'  # For Success/Reward
    RED = '\033[91m'    # For Crash
    BOLD = '\033[1m'

def quat_to_euler(quat):
    """ Convert [w, x, y, z] to [roll, pitch, yaw] in degrees. """
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(np.array([roll, pitch, yaw]))

def randomize_initial_state(env):
    """ Applies randomization to the environment. """
    # Position
    env.data.qpos[env.qpos_adr : env.qpos_adr+3] = [
        np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(10, 15)
    ]
    # Orientation (Max 30 deg tilt)
    tilt = np.deg2rad(np.random.uniform(0, 30))
    axis = np.random.randn(3); axis[2]=0; axis/=np.linalg.norm(axis)
    env.data.qpos[env.qpos_adr+3 : env.qpos_adr+7] = [
        np.cos(tilt/2), axis[0]*np.sin(tilt/2), axis[1]*np.sin(tilt/2), 0
    ]
    # Velocity
    env.data.qvel[env.qvel_adr : env.qvel_adr+3] = [
        np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-5, -1)
    ]

    mujoco.mj_forward(env.model, env.data)

def plot_all_trajectories(trajectories, env_name):
    """ Plots all collected trajectories in a 3D space and saves to file. """
    print(f"\n{Col.BOLD}📊 Generating 3D Trajectory Plot...{Col.RESET}")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        if len(trajectory) > 0:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=colors[i], label=f'Ep {i+1}')
            # Mark start and end
            ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2], color='green', marker='o', s=20)
            ax.scatter(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2], color='red', marker='x', s=30)
        
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position (Altitude)')
    ax.set_title(f'Rocket Trajectories ({env_name})')
    # ax.legend() # Legend can be crowded with many episodes
    
    filename = f"test_trajectories_{env_name}.png"
    plt.savefig(filename)
    plt.close(fig)
    print(f"💾 Plot saved to: {Col.CYAN}{filename}{Col.RESET}")
    
    # Try to open the file automatically
    try:
        if sys.platform == "darwin": subprocess.call(["open", filename])
        elif sys.platform == "win32": os.startfile(filename)
        else: subprocess.call(["xdg-open", filename])
    except: pass

import subprocess

# ================================================================
#   DYNAMIC ENV LOADER
# ================================================================
def get_env_class(env_name):
    """
    Dynamically imports the RocketLandingEnv class from the specified file.
    """
    # Map friendly names to module paths
    env_map = {
        "default": "rocket_env.rocket_landing_env",
        "env2":    "rocket_env.rocket_landing_env_2",
        "env3":    "rocket_env.rocket_landing_env_3",
        "simple":  "rocket_env.rocket_landing_env_simple",
    }

    if env_name not in env_map:
        print(f"{Col.RED}Error: Unknown environment '{env_name}'. Available: {list(env_map.keys())}{Col.RESET}")
        sys.exit(1)

    module_path = env_map[env_name]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, "RocketLandingEnv")
    except ImportError as e:
        print(f"{Col.RED}Error importing {module_path}: {e}{Col.RESET}")
        sys.exit(1)
    except AttributeError:
        print(f"{Col.RED}Error: 'RocketLandingEnv' class not found in {module_path}{Col.RESET}")
        sys.exit(1)

# ================================================================
#   MAIN LOOP
# ================================================================
def test_env(env_name, episodes=5):
    # 1. Load the correct environment class
    EnvClass = get_env_class(env_name)
    env = EnvClass(render_mode="human")
    
    print(f"\n{Col.BOLD}🚀 Testing Environment: {env_name} ({EnvClass.__module__}){Col.RESET}")

    all_trajectories = []

    for ep in range(episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{episodes}{Col.RESET}")
        print("-" * 140)
        print(f"{'STEP':<5} | {Col.CYAN}{'STATE (Alt/Vel/Tilt/Mass)':<40}{Col.RESET} | "
              f"{Col.YELLOW}{'CONTROLS (Thrust/Gimbal)':<30}{Col.RESET} | {Col.GREEN}{'REWARD':<10}{Col.RESET}")

        env.reset()
        env.render()
        
        # Apply extra randomization if desired (optional)
        randomize_initial_state(env)
        env.render()

        done = False
        truncated = False
        step = 0
        episode_trajectory = []

        while not (done or truncated):
            step += 1
            
            # --- 1. NO ACTION (PASSIVE DROP TEST) ---
            action = np.array([-1.0, 0.0, 0.0]) # Free fall (0 thrust)

            # --- 2. STEP ---
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            
            # --- 3. LOGGING ---
            pos = env.data.xpos[env.rocket_bid]
            vel = env.data.cvel[env.rocket_bid][3:]
            quat = env.data.qpos[env.qpos_adr+3 : env.qpos_adr+7]
            roll, pitch, yaw = quat_to_euler(quat)
            current_mass = env.DRY_MASS + env.fuel_mass
            thrust_N = env.data.ctrl[env.thrust_act]
            g_yaw    = np.degrees(env.data.ctrl[env.yaw_act])
            g_pit    = np.degrees(env.data.ctrl[env.pitch_act])

            # Collect trajectory data
            episode_trajectory.append(pos.copy())

            # Format String
            state_str = (
                f"Alt:{pos[2]:5.1f}m "
                f"Vz:{vel[2]:5.1f} "
                f"Tlt:{max(abs(pitch), abs(roll)):4.1f}° "
                f"Kg:{current_mass:5.1f}"
            )
            ctrl_str = f"Thr:{thrust_N:6.0f}N Gmb:{g_yaw:3.0f}/{g_pit:3.0f}"

            log_line = (
                f"\r{step:04}  | "
                f"{Col.CYAN}{state_str}{Col.RESET} | "
                f"{Col.YELLOW}{ctrl_str}{Col.RESET}     | "
                f"{Col.GREEN}{reward:6.1f}{Col.RESET} \033[K"
            )

            sys.stdout.write(log_line)
            sys.stdout.flush()
            time.sleep(0.01) 

        all_trajectories.append(episode_trajectory)

        # Episode Result
        result_color = Col.GREEN if info.get('success') else Col.RED
        result_msg = "✅ SUCCESS" if info.get('success') else "❌ FAILURE"
        print(f"\n{result_color}>>> RESULT: {result_msg}{Col.RESET}")

    env.close()
    
    # Plotting
    plot_all_trajectories(all_trajectories, env_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rocket Landing Environment")
    parser.add_argument("--env", type=str, default="default", 
                        choices=["default", "env2", "env3", "simple"],
                        help="Which environment file to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    test_env(args.env, args.episodes)