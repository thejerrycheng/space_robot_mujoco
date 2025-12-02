import os
import sys
import time
import argparse
import numpy as np
import subprocess
import csv
import matplotlib

# CRITICAL FIX: Use 'Agg' backend to prevent macOS main-thread crashes
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rocket_env.rocket_landing_env_simple import RocketLandingEnv

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
#   FILE & PLOTTING FUNCTIONS
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

def save_to_csv(history, episode_num, save_dir):
    """ Saves episode trajectory data to a CSV file. """
    filename = os.path.join(save_dir, f"episode_{episode_num}.csv")
    
    # Extract data
    times = history['time']
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    rewards = np.array(history['reward'])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow([
            "Step", "Time", "X", "Y", "Z", 
            "Roll", "Pitch", "Yaw", 
            "Thrust", "GimbalYaw", "GimbalPitch", 
            "Mass", "Reward"
        ])
        
        # Rows
        for i in range(len(times)):
            writer.writerow([
                i, times[i], pos[i,0], pos[i,1], pos[i,2],
                att[i,0], att[i,1], att[i,2],
                thrust[i], gimbal[i,0], gimbal[i,1],
                mass[i], rewards[i]
            ])
            
    print(f"💾 Data saved to: {filename}")

def generate_interactive_plot(all_histories, save_dir):
    """ 
    Generates an interactive 3D plot using Plotly (HTML).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"\n{Col.YELLOW}⚠️  'plotly' not found. Skipping interactive 3D plot.")
        print(f"   Run 'pip install plotly' to enable this feature.{Col.RESET}")
        return

    fig = go.Figure()

    for i, history in enumerate(all_histories):
        pos = np.array(history['pos'])
        
        # Trajectory Line
        fig.add_trace(go.Scatter3d(
            x=pos[:,0], y=pos[:,1], z=pos[:,2],
            mode='lines',
            name=f'Episode {i+1}',
            line=dict(width=4)
        ))
        
        # Start Point
        fig.add_trace(go.Scatter3d(
            x=[pos[0,0]], y=[pos[0,1]], z=[pos[0,2]],
            mode='markers',
            marker=dict(size=5, color='green'),
            name=f'Start Ep{i+1}',
            showlegend=False
        ))
        
        # End Point
        fig.add_trace(go.Scatter3d(
            x=[pos[-1,0]], y=[pos[-1,1]], z=[pos[-1,2]],
            mode='markers',
            marker=dict(size=5, color='red', symbol='x'),
            name=f'End Ep{i+1}',
            showlegend=False
        ))

    # Target Marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name='Target'
    ))

    fig.update_layout(
        title="Interactive Rocket Trajectories (3D)",
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Altitude (m)',
            aspectmode='data' # Keeps 1:1 scale ratios
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    filename = os.path.join(save_dir, "interactive_trajectories.html")
    fig.write_html(filename)
    print(f"\n{Col.BOLD}🌎 Interactive 3D Plot saved to: {filename}{Col.RESET}")
    open_file(filename)

def plot_static_analysis(history, episode_num, save_dir):
    """ Generates static matplotlib dashboard for specific episode stats. """
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"PPO Analysis - Episode {episode_num}", fontsize=16)

    # 1. 3D Trajectory (Static)
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b')
    ax3d.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.legend()

    # 2. Position
    ax_pos = fig.add_subplot(2, 3, 2)
    ax_pos.set_title("Position")
    ax_pos.plot(times, pos[:, 2], label='Z (Alt)', color='green')
    ax_pos.plot(times, np.sqrt(pos[:,0]**2 + pos[:,1]**2), label='Lateral Error', color='orange', linestyle='--')
    ax_pos.set_ylabel("m"); ax_pos.legend(); ax_pos.grid(True)

    # 3. Orientation
    ax_att = fig.add_subplot(2, 3, 3)
    ax_att.set_title("Orientation (Tilt)")
    ax_att.plot(times, att[:, 1], label='Pitch')
    ax_att.plot(times, att[:, 2], label='Yaw')
    ax_att.set_ylabel("Deg"); ax_att.legend(); ax_att.grid(True)

    # 4. Thrust
    ax_thr = fig.add_subplot(2, 3, 4)
    ax_thr.set_title("Thrust")
    ax_thr.plot(times, thrust, color='r')
    ax_thr.set_ylabel("N"); ax_thr.grid(True)

    # 5. Gimbal
    ax_gim = fig.add_subplot(2, 3, 5)
    ax_gim.set_title("Gimbal")
    ax_gim.plot(times, gimbal[:, 0], label='Yaw')
    ax_gim.plot(times, gimbal[:, 1], label='Pitch')
    ax_gim.set_ylabel("Deg"); ax_gim.legend(); ax_gim.grid(True)

    # 6. Mass
    ax_mass = fig.add_subplot(2, 3, 6)
    ax_mass.set_title("Mass")
    ax_mass.plot(times, mass, color='black')
    ax_mass.set_ylabel("kg"); ax_mass.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"episode_{episode_num}_analysis.png")
    plt.savefig(save_path)
    plt.close(fig)

# ================================================================
#   MAIN TESTING LOGIC
# ================================================================
def normalize_obs(obs, obs_rms, epsilon=1e-8):
    """ Manually normalize observation using stats from training. """
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO Rocket Agent")
    parser.add_argument("run_path", type=str, help="Path to the run folder (e.g., runs/ppo_rocket_...)")
    parser.add_argument("--model", type=str, default="best", choices=["best", "final", "latest"], help="Which model to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    # 1. Setup Directories
    if not os.path.exists(args.run_path):
        print(f"{Col.RED}Error: Run directory not found.{Col.RESET}")
        return

    data_dir = os.path.join("data", "ppo")
    os.makedirs(data_dir, exist_ok=True)
    print(f"{Col.BOLD}📂 Saving data to: {data_dir}{Col.RESET}")

    # 2. Load Normalization Statistics
    norm_path = os.path.join(args.run_path, "vec_normalize.pkl")
    if not os.path.exists(norm_path):
        print(f"{Col.RED}Error: vec_normalize.pkl not found.{Col.RESET}"); return

    # We load VecNormalize just to extract the stats (mean/var)
    # We will NOT use this wrapper for the environment loop to prevent auto-resets
    dummy_env = DummyVecEnv([lambda: RocketLandingEnv()])
    vec_norm = VecNormalize.load(norm_path, dummy_env)
    obs_rms = vec_norm.obs_rms
    
    # 3. Locate & Load Model
    if args.model == "best": model_file = "best_model/best_model.zip"
    elif args.model == "final": model_file = "final_model.zip"
    else: 
        ckpts = os.listdir(os.path.join(args.run_path, "checkpoints"))
        model_file = f"checkpoints/{sorted(ckpts)[-1]}"
    
    model_path = os.path.join(args.run_path, model_file)
    print(f"{Col.BOLD}🚀 Loading Model: {model_path}{Col.RESET}")
    
    model = PPO.load(model_path)

    # 4. Create RAW Environment (No Auto-Reset Wrapper)
    real_env = RocketLandingEnv(render_mode="human" if not args.no_render else None)

    all_histories = []

    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        
        obs, _ = real_env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # Capture Initial State (t=0)
        history = {'time': [], 'pos': [], 'attitude': [], 'thrust': [], 'gimbal': [], 'mass': [], 'reward': []}

        # Log initial state before stepping
        pos = real_env.data.xpos[real_env.rocket_bid].copy()
        quat = real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
        roll, pitch, yaw = quat_to_euler(quat)
        mass = real_env.DRY_MASS + real_env.fuel_mass
        
        history['time'].append(0)
        history['pos'].append(pos)
        history['attitude'].append([roll, pitch, yaw])
        history['thrust'].append(0.0)
        history['gimbal'].append([0.0, 0.0])
        history['mass'].append(mass)
        history['reward'].append(0.0)
        
        while True:
            step += 1
            
            # Manual Normalization
            norm_obs = normalize_obs(obs, obs_rms)
            
            # Predict
            action, _ = model.predict(norm_obs, deterministic=True)
            
            # Step Raw Env
            obs, reward, terminated, truncated, info = real_env.step(action)
            total_reward += reward
            
            # --- FETCH DATA (Now includes Terminal State) ---
            pos = real_env.data.xpos[real_env.rocket_bid].copy()
            vel = real_env.data.cvel[real_env.rocket_bid][3:].copy()
            quat = real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            mass = real_env.DRY_MASS + real_env.fuel_mass
            thrust_N = real_env.data.ctrl[real_env.thrust_act]
            g_yaw = np.degrees(real_env.data.ctrl[real_env.yaw_act])
            g_pit = np.degrees(real_env.data.ctrl[real_env.pitch_act])
            
            # --- STORE ---
            history['time'].append(step * real_env.DT)
            history['pos'].append(pos)
            history['attitude'].append([roll, pitch, yaw])
            history['thrust'].append(thrust_N)
            history['gimbal'].append([g_yaw, g_pit])
            history['mass'].append(mass)
            history['reward'].append(reward)

            # Dashboard
            if not args.no_render:
                real_env.render()
                state_str = f"Alt:{pos[2]:5.1f}m Vz:{vel[2]:5.1f} Tlt:{max(abs(pitch),abs(roll)):4.1f}°"
                ctrl_str  = f"Thr:{thrust_N:6.0f}N Gmb:{g_yaw:3.0f}/{g_pit:3.0f}"
                log_line = (
                    f"\r{step:04} | {Col.CYAN}{state_str}{Col.RESET} | "
                    f"{Col.YELLOW}{ctrl_str}{Col.RESET} | {Col.GREEN}Rew:{reward:6.2f}{Col.RESET} \033[K"
                )
                sys.stdout.write(log_line)
                sys.stdout.flush()
                time.sleep(0.01)

            # Break AFTER logging the final state
            if terminated or truncated:
                break

        # End of Episode
        all_histories.append(history)
        
        result_msg = "✅ SUCCESS" if info.get('success') else "❌ FAILURE"
        color = Col.GREEN if info.get('success') else Col.RED
        print(f"\n{color}>>> {result_msg} | Total Reward: {total_reward:.2f}{Col.RESET}")

        # 1. Save CSV
        save_to_csv(history, ep+1, data_dir)
        
        # 2. Save Static Analysis Plot
        plot_static_analysis(history, ep+1, data_dir)

    real_env.close()

    # 3. Generate Interactive Plot (Plotly)
    print(f"\n{Col.BOLD}📊 Generating Interactive 3D Plot...{Col.RESET}")
    generate_interactive_plot(all_histories, data_dir)
    
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    main()