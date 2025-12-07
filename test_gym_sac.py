import os
import sys
import time
import argparse
import numpy as np
import subprocess
import csv
import matplotlib
import importlib
import mujoco 

# CRITICAL FIX: Use 'Agg' backend to prevent macOS/Linux main-thread rendering crashes
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- IMPORT SAC MODEL INSTEAD OF PPO ---
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- IMPORT YOUR ENV CLASS ---
# Assuming the environment class is in the same relative location as the training script
from rocket_env.hover_gym import RocketLandingEnv 
from utils.plotting import generate_interactive_plot, Col

# ================================================================
#   UTILITIES: COLORS & MATH (Same as original)
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

def get_body_z_axis(quat):
    """ Calculates the Body Z-axis vector in World coordinates from a quaternion [w,x,y,z]. """
    w, x, y, z = quat
    # Formula for the 3rd column of the rotation matrix
    vec_x = 2 * (w*y + x*z)
    vec_y = 2 * (y*z - w*x)
    vec_z = 1 - 2 * (x*x + y*y)
    return np.array([vec_x, vec_y, vec_z])

# ================================================================
#   FILE & PLOTTING FUNCTIONS (Same as original)
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
    
    times = history['time']
    pos = np.array(history['pos'])
    vel = np.array(history['vel'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    rewards = np.array(history['reward'])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow([
            "Step", "Time", "X", "Y", "Z", "Vx", "Vy", "Vz",
            "Roll", "Pitch", "Yaw", 
            "Thrust", "GimbalYaw", "GimbalPitch", 
            "Mass", "Reward"
        ])
        
        # Rows
        for i in range(len(times)):
            writer.writerow([
                i, times[i], 
                pos[i,0], pos[i,1], pos[i,2],
                vel[i,0], vel[i,1], vel[i,2],
                att[i,0], att[i,1], att[i,2],
                thrust[i], gimbal[i,0], gimbal[i,1],
                mass[i], rewards[i]
            ])
            
    print(f"💾 Data saved to: {filename}")

def generate_interactive_plot(all_histories, save_dir=".", env_name="Rocket Env"):
    """ 
    Generates a high-fidelity interactive 3D plot using Plotly.
    """
    try:
        import plotly.graph_objects as go
        import plotly.colors as pc
    except ImportError:
        print(f"\n{Col.YELLOW}⚠️  'plotly' not found. Skipping interactive 3D plot.")
        print(f"   Run 'pip install plotly' to enable this feature.{Col.RESET}")
        return

    print(f"\n{Col.BOLD}📊 Generating Interactive Plotly HTML...{Col.RESET}")
    fig = go.Figure()
    palette = pc.qualitative.Plotly 

    for i, history in enumerate(all_histories):
        pos = np.array(history['pos'])
        if len(pos) < 2: continue 
        
        quats = np.array(history.get('quat', []))

        steps = np.arange(len(pos))
        
        vel_z = np.array(history['vel'])[:, 2] 
        att = np.array(history['attitude'])
        # Calculate maximum tilt (max of roll and pitch deviation from 0)
        tilt = np.maximum(np.abs(att[:,0]), np.abs(att[:,1]))
        
        # Create Hover Text
        hover_text = [
            f"Step: {s}<br>Alt: {z:.2f}m<br>Vz: {v:.2f}m/s<br>Tilt: {t:.1f}°"
            for s, z, v, t in zip(steps, pos[:,2], vel_z, tilt)
        ]

        color = palette[i % len(palette)]

        # 1. Trajectory Line
        fig.add_trace(go.Scatter3d(
            x=pos[:,0], y=pos[:,1], z=pos[:,2],
            mode='lines',
            name=f'Ep {i+1} Traj',
            text=hover_text,
            hoverinfo='text',
            line=dict(width=5, color=color),
            opacity=0.8
        ))
        
        # 2. Heading Vectors (Cone Plot)
        if len(quats) == len(pos):
            step_interval = 30 
            indices = np.arange(0, len(pos), step_interval)
            
            if len(indices) > 0:
                sub_pos = pos[indices]
                sub_quats = quats[indices]
                
                # Calculate heading vectors (rocket body Z-axis)
                headings = np.array([get_body_z_axis(q) for q in sub_quats])
                
                fig.add_trace(go.Cone(
                    x=sub_pos[:, 0], y=sub_pos[:, 1], z=sub_pos[:, 2],
                    u=headings[:, 0], v=headings[:, 1], w=headings[:, 2],
                    sizemode="scaled",
                    sizeref=0.5,
                    showscale=False,
                    anchor="tail",
                    colorscale=[[0, color], [1, color]],
                    name=f'Ep {i+1} Heading'
                ))
        
        # 3. Start Point, 4. End Point (Markers) - Skipped for brevity, assume present from original

    # --- ENVIRONMENT CONTEXT ---
    theta = np.linspace(0, 2*np.pi, 50)
    # Landing Pad (assuming tolerances from your training script: 10m pos tol, 8 deg tilt tol)
    r_pad = 1.0 # Standard landing pad radius
    r_zone = 5.0 # Target Zone for visual reference
    
    fig.add_trace(go.Scatter3d(
        x=r_pad * np.cos(theta), y=r_pad * np.sin(theta), z=np.zeros_like(theta),
        mode='lines', line=dict(color='black', width=4, dash='dash'), name='Landing Pad (1m)'
    ))
    fig.add_trace(go.Scatter3d(
        x=r_zone * np.cos(theta), y=r_zone * np.sin(theta), z=np.zeros_like(theta),
        mode='lines', line=dict(color='orange', width=2, dash='dot'), name='Target Zone (5m)'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers', marker=dict(size=5, color='black', symbol='diamond'), name='Target'
    ))

    # Layout Config (Same as original)
    fig.update_layout(
        title=f"🚀 {env_name} Trajectory Analysis",
        width=1200, height=800,
        scene=dict(
            xaxis_title='X (Lateral)', yaxis_title='Y (Lateral)', zaxis_title='Z (Altitude)',
            aspectmode='data', 
            xaxis=dict(gridcolor='lightgrey', backgroundcolor="white"),
            yaxis=dict(gridcolor='lightgrey', backgroundcolor="white"),
            zaxis=dict(gridcolor='grey', backgroundcolor="#F0F0F0"),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05)
    )

    filename = os.path.join(save_dir, f"interactive_trajectories_{env_name}.html")
    fig.write_html(filename)
    
    print(f"{Col.BOLD}🌎 Plot Saved: {Col.CYAN}{filename}{Col.RESET}")
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
    fig.suptitle(f"SAC Analysis - Episode {episode_num}", fontsize=16)

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

    # 4. Thrust (Continuous Control)
    ax_thr = fig.add_subplot(2, 3, 4)
    ax_thr.set_title("Thrust Command (SAC Output 0)")
    # SAC output is [-1, 1]. This needs to be denormalized for true thrust, but plotting the raw command is often better for analysis.
    ax_thr.plot(times, thrust, color='r') 
    ax_thr.set_ylabel("N (or Norm. Action)"); ax_thr.grid(True) 

    # 5. Gimbal (Continuous Control)
    ax_gim = fig.add_subplot(2, 3, 5)
    ax_gim.set_title("Gimbal Commands (SAC Output 1, 2)")
    # SAC action[1] is Yaw, action[2] is Pitch
    ax_gim.plot(times, gimbal[:, 0], label='Yaw Command')
    ax_gim.plot(times, gimbal[:, 1], label='Pitch Command')
    ax_gim.set_ylabel("Norm. Action [-1, 1]"); ax_gim.legend(); ax_gim.grid(True)

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
#   DYNAMIC REWARD LOADER (Same as training script)
# ================================================================
def load_reward_class(reward_name):
    try:
        module_path = f"rocket_env.rewards.{reward_name}"
        mod = importlib.import_module(module_path)
        if hasattr(mod, "RocketReward"):
            return mod.RocketReward
        else:
            for attr_name in dir(mod):
                if "Reward" in attr_name and attr_name != "RocketReward":
                    return getattr(mod, attr_name)
            raise AttributeError(f"Could not find 'RocketReward' class in {module_path}")
    except ImportError as e:
        print(f"❌ Error loading reward: {reward_name}")
        raise e

# ================================================================
#   MAIN TESTING LOGIC
# ================================================================
def normalize_obs(obs, obs_rms, epsilon=1e-8):
    """ Normalize observation using loaded statistics """
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

def main():
    parser = argparse.ArgumentParser(description="Test a trained SAC Rocket Agent")
    # Change the help message to reflect SAC paths
    parser.add_argument("run_path", type=str, help="Path to the SAC run folder (e.g., models/gym_sac_X_date)")
    parser.add_argument("--model", type=str, default="best", choices=["best", "final", "latest"], help="Which model to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes to visualize")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    # Change the default reward name to match the training script's default
    parser.add_argument("--reward", type=str, default="upright_only", help="Name of reward file in rocket_env/rewards/")
    
    args = parser.parse_args()

    # 1. Setup Directories
    if not os.path.exists(args.run_path):
        print(f"{Col.RED}Error: Run directory '{args.run_path}' not found.{Col.RESET}")
        return

    data_dir = os.path.join(args.run_path, "test_results_sac")
    os.makedirs(data_dir, exist_ok=True)
    print(f"{Col.BOLD}📂 Saving test data to: {data_dir}{Col.RESET}")

    # 2. Load Reward Class (needed for environment initialization)
    try:
        RewardClass = load_reward_class(args.reward)
        rewarder = RewardClass({'start_fuel': 4_500_000.0}) # Use the same initial config as training
        print(f"{Col.CYAN}💰 Loaded Reward Class: {args.reward}{Col.RESET}")
    except Exception as e:
        print(f"{Col.RED}Error initializing rewarder: {e}{Col.RESET}")
        return

    # 3. Load Normalization Statistics (if used, which is common for SAC/PPO)
    norm_path = os.path.join(args.run_path, "vec_normalize.pkl")
    vec_norm = None
    obs_rms = None
    
    if os.path.exists(norm_path):
        dummy_env = DummyVecEnv([lambda: RocketLandingEnv()])
        vec_norm = VecNormalize.load(norm_path, dummy_env)
        obs_rms = vec_norm.obs_rms
        print(f"{Col.GREEN}✅ Loaded VecNormalize Stats{Col.RESET}")
    else:
        print(f"{Col.YELLOW}⚠️ WARNING: vec_normalize.pkl not found. Assuming unnormalized observations.{Col.RESET}")


    # 4. Locate & Load SAC Model
    if args.model == "final":
        model_file = "final_model.zip"
        model_path = os.path.join(args.run_path, model_file)
    elif args.model == "best":
        model_path = os.path.join(args.run_path, "best_model", "best_model.zip")
    else: # latest checkpoint
        ckpt_dir = os.path.join(args.run_path, "checkpoints")
        ckpts = glob.glob(os.path.join(ckpt_dir, "sac_rocket_*.zip"))
        if ckpts:
            ckpts.sort(key=os.path.getmtime)
            model_path = ckpts[-1]
        else:
            print(f"{Col.RED}Error: No checkpoints found in {ckpt_dir}.{Col.RESET}"); return

    if not os.path.exists(model_path):
        print(f"{Col.RED}Error: Model file not found at {model_path}.{Col.RESET}")
        return
    
    print(f"{Col.BOLD}🚀 Loading SAC Model from: {model_path}{Col.RESET}")
    # --- SAC LOAD FIX ---
    model = SAC.load(model_path) 

    # 5. Create RAW Environment for Testing
    real_env = RocketLandingEnv(
        render_mode="human" if not args.no_render else None
    )
    # Assign the loaded rewarder instance
    if hasattr(real_env, "rewarder"):
        real_env.rewarder = rewarder
    
    # Cache MuJoCo IDs (assuming they are standard names in RocketLandingEnv)
    real_env.rocket_bid = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    real_env.qpos_adr = 0 # Assuming the rocket is the first body
    real_env.qvel_adr = 0
    real_env.thrust_act = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
    real_env.yaw_act = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
    real_env.pitch_act = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
    real_env.DT = real_env.model.opt.timestep

    all_histories = []

    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        
        obs, _ = real_env.reset()
        
        # --- MUJOCO STATE MODIFICATION LOGIC ---
        # NOTE: Your initial randomization logic was commented out, 
        # so we will use the environment's default reset for now.
        mujoco.mj_forward(real_env.model, real_env.data)
        # Refresh observation after reset (if needed)
        try: obs = real_env._get_obs()
        except AttributeError: pass
        # --- END MUJOCO STATE MODIFICATION ---

        step = 0
        total_reward = 0
        
        history = {
            'time': [], 'pos': [], 'vel': [], 'attitude': [], 'quat': [], 
            'thrust': [], 'gimbal': [], 'mass': [], 'reward': []
        }

        while True:
            step += 1
            
            # 6. Normalize Observation
            norm_obs = normalize_obs(obs, obs_rms) if obs_rms else obs
            
            # 7. Predict Action
            # SAC uses deterministic=True for evaluation by default, but we specify it for clarity
            action, _ = model.predict(norm_obs, deterministic=True)
            
            # 8. Step Environment
            obs, reward, terminated, truncated, info = real_env.step(action)
            total_reward += reward
            
            # 9. Log State
            pos = real_env.data.xpos[real_env.rocket_bid].copy()
            # Assuming cvel[3:] is translational velocity in world frame if using Mujoco's base velocity array format
            # A safer approach is often to use the first three elements of qvel[0:3] for world translational velocity
            vel = real_env.data.qvel[0:3].copy() 
            quat = real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            mass = real_env.DRY_MASS + real_env.fuel_mass # Uses env's internal fuel_mass
            
            # Actions are normalized [-1, 1]. Gimbal controls are also normalized actions.
            thrust_N_cmd = real_env.data.ctrl[real_env.thrust_act] 
            g_yaw_cmd = real_env.data.ctrl[real_env.yaw_act]
            g_pit_cmd = real_env.data.ctrl[real_env.pitch_act]
            
            history['time'].append(step * real_env.DT)
            history['pos'].append(pos)
            history['vel'].append(vel)
            history['attitude'].append([roll, pitch, yaw])
            history['quat'].append(quat) 
            history['thrust'].append(thrust_N_cmd) # Log the normalized action output for thrust visualization
            history['gimbal'].append([g_yaw_cmd, g_pit_cmd]) # Log the normalized action output for gimbal visualization
            history['mass'].append(mass)
            history['reward'].append(reward)

            if not args.no_render:
                real_env.render()
                state_str = f"Alt:{pos[2]:5.1f}m Dis:{np.sqrt(pos[0]**2 + pos[1]**2):5.1f}m Vz:{vel[2]:5.1f} Tlt:{max(abs(pitch),abs(roll)):4.1f}°"
                ctrl_str  = f"Thr:{thrust_N_cmd:6.2f} GimY:{g_yaw_cmd:4.2f}"
                log_line = (
                    f"\r{step:04} | {Col.CYAN}{state_str}{Col.RESET} | "
                    f"{Col.YELLOW}{ctrl_str}{Col.RESET} | {Col.GREEN}Rew:{reward:6.2f}{Col.RESET} \033[K"
                )
                sys.stdout.write(log_line)
                sys.stdout.flush()
                time.sleep(0.01) 

            if terminated or truncated:
                break

        all_histories.append(history)
        
        # End of Episode Logging
        final_pos = history['pos'][-1]
        dist_xy = np.sqrt(final_pos[0]**2 + final_pos[1]**2)
        
        is_success = info.get('is_success', False) # Use 'is_success' key from your training script
        
        # Semi-Success Check (Optional, based on your original script)
        is_semi_success = (dist_xy < 5.0) and not is_success

        result_msg = "❌ FAILURE"
        if is_success:
            result_msg = f"{Col.GREEN}✅ SUCCESS{Col.RESET}"
        elif is_semi_success:
            result_msg = f"{Col.YELLOW}⚠️ SEMI-SUCCESS (In Zone: {dist_xy:.2f}m){Col.RESET}"
        else:
            result_msg = f"{Col.RED}❌ FAILURE (Dist: {dist_xy:.2f}m){Col.RESET}"

        print(f"\n{result_msg} | Total Reward: {total_reward:.2f}")

        save_to_csv(history, ep+1, data_dir)
        plot_static_analysis(history, ep+1, data_dir)

    real_env.close()

    print(f"\n{Col.BOLD}📊 Generating Interactive 3D Plot...{Col.RESET}")
    generate_interactive_plot(all_histories, data_dir, env_name="SAC Rocket Landing")
    
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    # Example usage: mjpython test_sac_agent.py models/gym_sac_default_20240101_120000 --episodes 5
    main()