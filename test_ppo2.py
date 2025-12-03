import os
import sys
import time
import argparse
import numpy as np
import subprocess
import csv
import matplotlib

# CRITICAL FIX: Use 'Agg' backend to prevent macOS/Linux main-thread rendering crashes
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# UPDATED IMPORT: Pointing to the fixed environment
from rocket_env.rocket_2_env import RocketLandingEnv

# ================================================================
#   UTILITIES: COLORS & MATH
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
    # Standard conversion
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
    """ Cross-platform file opener for the resulting plots """
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
    
    times = history['time']
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    rewards = np.array(history['reward'])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Step", "Time", "X", "Y", "Z", 
            "Roll", "Pitch", "Yaw", 
            "Thrust_N", "GimbalYaw_Deg", "GimbalPitch_Deg", 
            "Mass_kg", "Reward"
        ])
        for i in range(len(times)):
            writer.writerow([
                i, times[i], 
                f"{pos[i,0]:.4f}", f"{pos[i,1]:.4f}", f"{pos[i,2]:.4f}",
                f"{att[i,0]:.2f}", f"{att[i,1]:.2f}", f"{att[i,2]:.2f}",
                f"{thrust[i]:.2f}", f"{gimbal[i,0]:.2f}", f"{gimbal[i,1]:.2f}",
                f"{mass[i]:.4f}", f"{rewards[i]:.4f}"
            ])
            
    print(f"💾 Data saved to: {filename}")

def generate_interactive_plot(all_histories, save_dir):
    """ Generates an interactive 3D plot using Plotly (HTML). """
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
            mode='markers', marker=dict(size=4, color='green'),
            name=f'Start Ep{i+1}', showlegend=False
        ))
        
        # End Point
        fig.add_trace(go.Scatter3d(
            x=[pos[-1,0]], y=[pos[-1,1]], z=[pos[-1,2]],
            mode='markers', marker=dict(size=4, color='red', symbol='x'),
            name=f'End Ep{i+1}', showlegend=False
        ))

    # Target Marker at (0,0,0)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='black', symbol='diamond'),
        name='Target (0,0,0)'
    ))

    fig.update_layout(
        title="Interactive Rocket Trajectories (3D)",
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Altitude (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    filename = os.path.join(save_dir, "interactive_trajectories.html")
    fig.write_html(filename)
    print(f"\n{Col.BOLD}🌎 Interactive 3D Plot saved to: {filename}{Col.RESET}")
    open_file(filename)

def plot_static_analysis(history, episode_num, save_dir):
    """ Generates static matplotlib dashboard. """
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Episode {episode_num} Analysis", fontsize=16)

    # 1. Altitude & Descent
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(times, pos[:, 2], color='blue', label='Altitude')
    ax1.axhline(0, color='k', linestyle='--')
    ax1.set_title("Altitude (Z)")
    ax1.set_ylabel("Meters")
    ax1.grid(True)

    # 2. Lateral Deviation
    ax2 = fig.add_subplot(2, 3, 2)
    dist_xy = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
    ax2.plot(times, dist_xy, color='orange', label='Dist XY')
    ax2.set_title("Lateral Error (XY Distance)")
    ax2.set_ylabel("Meters")
    ax2.grid(True)

    # 3. Orientation (Pitch/Roll)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(times, att[:, 1], label='Pitch') # Pitch
    ax3.plot(times, att[:, 0], label='Roll')  # Roll
    ax3.set_title("Orientation (Ideally 0°)")
    ax3.set_ylabel("Degrees")
    ax3.legend()
    ax3.grid(True)

    # 4. Thrust
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(times, thrust, color='red')
    ax4.set_title("Thrust Command")
    ax4.set_ylabel("Newtons")
    ax4.grid(True)

    # 5. Gimbal Action
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(times, gimbal[:, 0], label='Yaw')
    ax5.plot(times, gimbal[:, 1], label='Pitch')
    ax5.set_title("Gimbal Deflection")
    ax5.set_ylabel("Degrees")
    ax5.legend()
    ax5.grid(True)
    
    # 6. Trajectory Top-Down
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(pos[:, 0], pos[:, 1], color='purple')
    ax6.scatter(0, 0, marker='*', color='k', s=100, label='Target')
    ax6.set_title("Top-Down View (X-Y)")
    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Y (m)")
    ax6.axis('equal')
    ax6.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"episode_{episode_num}_analysis.png")
    plt.savefig(save_path)
    plt.close(fig)

# ================================================================
#   MAIN TESTING LOGIC
# ================================================================
def normalize_obs(obs, obs_rms, epsilon=1e-8):
    """ 
    Manually normalize observation using stats from training. 
    This allows us to run the raw environment (for rendering) 
    while feeding the agent normalized data.
    """
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

def main():
    parser = argparse.ArgumentParser(description="Test a trained PPO Rocket Agent")
    parser.add_argument("run_path", type=str, help="Path to the run folder (e.g., runs/ppo_rocket_...)")
    parser.add_argument("--model", type=str, default="best", choices=["best", "final", "latest"], help="Which model to load")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    # 1. Setup Directories
    if not os.path.exists(args.run_path):
        print(f"{Col.RED}Error: Run directory '{args.run_path}' not found.{Col.RESET}")
        return

    data_dir = os.path.join(args.run_path, "test_results")
    os.makedirs(data_dir, exist_ok=True)
    print(f"{Col.BOLD}📂 Saving test data to: {data_dir}{Col.RESET}")

    # 2. Load Normalization Statistics
    norm_path = os.path.join(args.run_path, "vec_normalize.pkl")
    if not os.path.exists(norm_path):
        print(f"{Col.RED}Error: vec_normalize.pkl not found. Agent will fail.{Col.RESET}"); return

    # Load VecNormalize stats using a dummy env
    dummy_env = DummyVecEnv([lambda: RocketLandingEnv()])
    vec_norm = VecNormalize.load(norm_path, dummy_env)
    obs_rms = vec_norm.obs_rms
    print(f"{Col.GREEN}✅ Loaded Normalization Stats{Col.RESET}")

    # 3. Locate & Load Model
    # Logic to find the zip file
    if args.model == "final":
        model_file = "final_model.zip"
        model_path = os.path.join(args.run_path, model_file)
    elif args.model == "best":
        # Check if best_model is a folder (callback often saves as a folder) or file
        possible_path = os.path.join(args.run_path, "best_model")
        if os.path.isdir(possible_path):
             model_path = os.path.join(possible_path, "best_model.zip")
        else:
             model_path = os.path.join(args.run_path, "best_model.zip")
    else: # latest from checkpoints
        ckpt_dir = os.path.join(args.run_path, "checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
            if ckpts:
                # simple sort by name works if prefix is constant and numbers are zero padded
                # or creation time
                ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
                model_file = ckpts[-1]
                model_path = os.path.join(ckpt_dir, model_file)
            else:
                print("No checkpoints found.")
                return
        else:
            print("Checkpoint directory not found.")
            return

    if not os.path.exists(model_path):
        # Fallback
        model_path = os.path.join(args.run_path, "final_model.zip")
    
    print(f"{Col.BOLD}🚀 Loading Model: {model_path}{Col.RESET}")
    model = PPO.load(model_path)

    # 4. Create RAW Environment for Visualization
    # Note: We do NOT wrap this in VecNormalize because we want to see real physics,
    # and we want to control the stepping loop manually.
    real_env = RocketLandingEnv(render_mode="human" if not args.no_render else None)

    all_histories = []

    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        
        # Reset returns (obs, info) in newer Gym versions
        obs, _ = real_env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # Init History
        history = {'time': [], 'pos': [], 'attitude': [], 'thrust': [], 'gimbal': [], 'mass': [], 'reward': []}

        # Log initial state (t=0)
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
            
            # 1. Normalize Observation manually
            norm_obs = normalize_obs(obs, obs_rms)
            
            # 2. Predict Action (Deterministic)
            action, _ = model.predict(norm_obs, deterministic=True)
            
            # 3. Step Environment
            # Gym API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = real_env.step(action)
            total_reward += reward
            
            # 4. Extract Data from MuJoCo internals
            pos = real_env.data.xpos[real_env.rocket_bid].copy()
            vel = real_env.data.cvel[real_env.rocket_bid][3:].copy()
            quat = real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            mass = real_env.DRY_MASS + real_env.fuel_mass
            
            # Control Inputs
            thrust_N = real_env.data.ctrl[real_env.thrust_act]
            g_yaw = np.degrees(real_env.data.ctrl[real_env.yaw_act])
            g_pit = np.degrees(real_env.data.ctrl[real_env.pitch_act])
            
            # 5. Store
            history['time'].append(step * real_env.DT)
            history['pos'].append(pos)
            history['attitude'].append([roll, pitch, yaw])
            history['thrust'].append(thrust_N)
            history['gimbal'].append([g_yaw, g_pit])
            history['mass'].append(mass)
            history['reward'].append(reward)

            # 6. Dashboard
            if not args.no_render:
                real_env.render()
                
                # Format string for console
                state_str = f"Alt:{pos[2]:5.1f}m Vz:{vel[2]:5.1f} Tlt:{max(abs(pitch),abs(roll)):4.1f}°"
                ctrl_str  = f"Thr:{thrust_N:6.0f}N"
                
                log_line = (
                    f"\r{step:04} | {Col.CYAN}{state_str}{Col.RESET} | "
                    f"{Col.YELLOW}{ctrl_str}{Col.RESET} | {Col.GREEN}Rew:{reward:6.2f}{Col.RESET} \033[K"
                )
                sys.stdout.write(log_line)
                sys.stdout.flush()
                # time.sleep(0.01) # Uncomment to run in slow motion

            if terminated or truncated:
                break

        # Episode Summary
        all_histories.append(history)
        
        success = info.get('success', False)
        result_msg = "✅ SUCCESS" if success else "❌ FAILURE"
        color = Col.GREEN if success else Col.RED
        print(f"\n{color}>>> {result_msg} | Total Reward: {total_reward:.2f}{Col.RESET}")

        # Save artifacts
        save_to_csv(history, ep+1, data_dir)
        plot_static_analysis(history, ep+1, data_dir)

    real_env.close()

    # Generate Aggregate 3D Plot
    print(f"\n{Col.BOLD}📊 Generating Interactive 3D Plot...{Col.RESET}")
    generate_interactive_plot(all_histories, data_dir)
    
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    main()