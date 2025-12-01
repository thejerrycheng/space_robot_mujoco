import os
import sys
import time
import argparse
import numpy as np
import subprocess
import matplotlib

# CRITICAL FIX: Use 'Agg' backend to prevent macOS main-thread crashes
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rocket_env.rocket_landing_env import RocketLandingEnv

# ================================================================
#   UTILITIES: COLORS & DASHBOARD
# ================================================================
class Col:
    RESET = '\033[0m'
    CYAN = '\033[96m'   # State
    YELLOW = '\033[93m' # Control
    GREEN = '\033[92m'  # Success
    RED = '\033[91m'    # Failure
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

# ================================================================
#   PLOTTING FUNCTIONS
# ================================================================
def open_file(path):
    """ Cross-platform file opener """
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.call(["open", path])
        elif sys.platform == "win32": # Windows
            os.startfile(path)
        else: # Linux
            subprocess.call(["xdg-open", path])
    except Exception:
        pass

def plot_episode_data(history, episode_num, save_dir):
    """ Generates detailed analysis for a single episode. """
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])   # [Roll, Pitch, Yaw]
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])  # [Yaw, Pitch]
    mass = np.array(history['mass'])
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"PPO Analysis - Episode {episode_num}", fontsize=16)

    # 1. 3D Trajectory
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b', linewidth=2)
    ax3d.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='g', marker='o', label='Start')
    ax3d.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color='r', marker='x', label='End')
    ax3d.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.legend()

    # 2. Position
    ax_pos = fig.add_subplot(2, 3, 2)
    ax_pos.set_title("Position")
    ax_pos.plot(times, pos[:, 0], label='X', linestyle='--', alpha=0.7)
    ax_pos.plot(times, pos[:, 1], label='Y', linestyle='--', alpha=0.7)
    ax_pos.plot(times, pos[:, 2], label='Z (Alt)', linewidth=2, color='green')
    ax_pos.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax_pos.set_ylabel("Position (m)"); ax_pos.grid(True); ax_pos.legend()

    # 3. Orientation
    ax_att = fig.add_subplot(2, 3, 3)
    ax_att.set_title("Orientation")
    ax_att.plot(times, att[:, 1], label='Pitch', color='orange')
    ax_att.plot(times, att[:, 2], label='Yaw', color='purple')
    ax_att.set_ylabel("Deg"); ax_att.grid(True); ax_att.legend()

    # 4. Thrust
    ax_thr = fig.add_subplot(2, 3, 4)
    ax_thr.set_title("Thrust")
    ax_thr.plot(times, thrust, color='r')
    ax_thr.fill_between(times, thrust, color='r', alpha=0.1)
    ax_thr.set_ylabel("Newtons"); ax_thr.grid(True)

    # 5. Gimbal
    ax_gim = fig.add_subplot(2, 3, 5)
    ax_gim.set_title("Gimbal Angles")
    ax_gim.plot(times, gimbal[:, 0], label='Yaw')
    ax_gim.plot(times, gimbal[:, 1], label='Pitch')
    ax_gim.set_ylabel("Deg"); ax_gim.grid(True); ax_gim.legend()

    # 6. Mass
    ax_mass = fig.add_subplot(2, 3, 6)
    ax_mass.set_title("Fuel Mass")
    ax_mass.plot(times, mass, color='black')
    ax_mass.set_ylabel("kg"); ax_mass.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"episode_{episode_num}_ppo_analysis.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"{Col.BOLD}📊 Episode plot saved to: {save_path}{Col.RESET}")
    # open_file(save_path) # Optional: open every single one?

def plot_aggregate_trajectories(all_histories, save_dir):
    """ Plots all trajectories in one 3D plot at the end. """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Aggregate PPO Trajectories ({len(all_histories)} Episodes)")

    # Use a colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(all_histories)))

    for i, history in enumerate(all_histories):
        pos = np.array(history['pos'])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i], label=f'Ep {i+1}', alpha=0.8)
        # Mark touchdown
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color=colors[i], marker='x')

    # Target
    ax.scatter(0, 0, 0, color='k', marker='*', s=200, label='Target')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    # ax.legend() # Legend might be too big if many episodes
    
    save_path = os.path.join(save_dir, "ALL_PPO_TRAJECTORIES.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\n{Col.BOLD}🌎 Aggregate 3D Plot saved to: {save_path}{Col.RESET}")
    open_file(save_path)

# ================================================================
#   MAIN TESTING LOGIC
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO Rocket Agent")
    parser.add_argument("run_path", type=str, help="Path to the run folder (e.g., runs/ppo_rocket_...)")
    parser.add_argument("--model", type=str, default="best", choices=["best", "final", "latest"], help="Which model to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    # 1. Resolve Paths
    run_dir = args.run_path
    if not os.path.exists(run_dir):
        print(f"{Col.RED}Error: Run directory not found: {run_dir}{Col.RESET}")
        return

    norm_path = os.path.join(run_dir, "vec_normalize.pkl")
    if not os.path.exists(norm_path):
        print(f"{Col.RED}Error: vec_normalize.pkl not found in {run_dir}{Col.RESET}")
        return

    # Locate Model
    if args.model == "best":
        model_path = os.path.join(run_dir, "best_model", "best_model.zip")
    elif args.model == "final":
        model_path = os.path.join(run_dir, "final_model.zip")
    elif args.model == "latest":
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
            if ckpts:
                latest = sorted(ckpts)[-1] 
                model_path = os.path.join(ckpt_dir, latest)
            else:
                print(f"{Col.RED}No checkpoints found.{Col.RESET}")
                return
        else:
            print(f"{Col.RED}No checkpoints folder found.{Col.RESET}")
            return
    
    if not os.path.exists(model_path):
        print(f"{Col.RED}Error: Model file not found: {model_path}{Col.RESET}")
        return

    print(f"\n{Col.BOLD}🚀 LOADING PPO AGENT{Col.RESET}")
    print(f"   Run Dir:   {run_dir}")
    print(f"   Model:     {model_path}")

    # 2. Setup Environment
    env = DummyVecEnv([lambda: RocketLandingEnv(render_mode="human" if not args.no_render else None)])
    env = VecNormalize.load(norm_path, env)
    env.training = False 
    env.norm_reward = False

    # 3. Load Agent
    model = PPO.load(model_path)

    # 4. Evaluation Loop
    all_histories = []

    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        print("-" * 80)
        
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0

        # Data Recording
        history = {'time': [], 'pos': [], 'attitude': [], 'thrust': [], 'gimbal': [], 'mass': []}

        while not done:
            step += 1
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            if done: break
            
            real_env = env.venv.envs[0] 
            
            # Fetch Data
            pos = real_env.data.xpos[real_env.rocket_bid].copy()
            vel = real_env.data.cvel[real_env.rocket_bid][3:].copy()
            quat = real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            mass = real_env.DRY_MASS + real_env.fuel_mass
            
            thrust_N = real_env.data.ctrl[real_env.thrust_act]
            g_yaw = np.degrees(real_env.data.ctrl[real_env.yaw_act])
            g_pit = np.degrees(real_env.data.ctrl[real_env.pitch_act])
            
            # Store
            history['time'].append(step * real_env.DT)
            history['pos'].append(pos)
            history['attitude'].append([roll, pitch, yaw])
            history['thrust'].append(thrust_N)
            history['gimbal'].append([g_yaw, g_pit])
            history['mass'].append(mass)

            # Dashboard
            if not args.no_render:
                real_env.render()
                state_str = f"Alt:{pos[2]:5.1f}m Vz:{vel[2]:5.1f} Tlt:{max(abs(pitch),abs(roll)):4.1f}°"
                ctrl_str  = f"Thr:{thrust_N:6.0f}N Gmb:{g_yaw:3.0f}/{g_pit:3.0f}"
                log_line = (
                    f"\r{step:04} | {Col.CYAN}{state_str}{Col.RESET} | "
                    f"{Col.YELLOW}{ctrl_str}{Col.RESET} | {Col.GREEN}Rew:{reward[0]:6.2f}{Col.RESET} \033[K"
                )
                sys.stdout.write(log_line)
                sys.stdout.flush()
                time.sleep(0.01)

        # End of Episode
        all_histories.append(history)
        
        info_dict = info[0]
        result_msg = "✅ SUCCESS" if info_dict.get('success') else "❌ FAILURE"
        color = Col.GREEN if info_dict.get('success') else Col.RED
        print(f"\n{color}>>> {result_msg} | Total Reward: {total_reward:.2f}{Col.RESET}")

        # Plot Individual Episode
        plot_episode_data(history, ep+1, run_dir)

    env.close()
    
    # 5. Plot All Trajectories Together
    print(f"\n{Col.BOLD}📊 Generating Aggregate 3D Plot...{Col.RESET}")
    plot_aggregate_trajectories(all_histories, run_dir)
    
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    main()