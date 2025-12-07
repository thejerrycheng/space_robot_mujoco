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

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Custom Imports
from rocket_env.polar_rocket_env import RocketLandingEnv
from discrete_action_wrapper import DiscreteActionWrapper

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
    vec_x = 2 * (w*y + x*z)
    vec_y = 2 * (y*z - w*x)
    vec_z = 1 - 2 * (x*x + y*y)
    return np.array([vec_x, vec_y, vec_z])

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
        writer.writerow([
            "Step", "Time", "X", "Y", "Z", "Vx", "Vy", "Vz",
            "Roll", "Pitch", "Yaw", 
            "Thrust", "GimbalYaw", "GimbalPitch", 
            "Mass", "Reward"
        ])
        
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

# ================================================================
#   INTERACTIVE PLOTTING (PLOTLY)
# ================================================================
def generate_interactive_plot(all_histories, save_dir=".", env_name="Rocket Env"):
    try:
        import plotly.graph_objects as go
        import plotly.colors as pc
    except ImportError:
        print(f"\n{Col.YELLOW}⚠️  'plotly' not found. Skipping interactive 3D plot.{Col.RESET}")
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
        tilt = np.maximum(np.abs(att[:,0]), np.abs(att[:,1]))
        
        hover_text = [
            f"Step: {s}<br>Alt: {z:.2f}m<br>Vz: {v:.2f}m/s<br>Tilt: {t:.1f}°"
            for s, z, v, t in zip(steps, pos[:,2], vel_z, tilt)
        ]

        color = palette[i % len(palette)]

        # Trajectory
        fig.add_trace(go.Scatter3d(
            x=pos[:,0], y=pos[:,1], z=pos[:,2],
            mode='lines',
            name=f'Ep {i+1} Traj',
            text=hover_text,
            hoverinfo='text',
            line=dict(width=5, color=color),
            opacity=0.8
        ))
        
        # Heading Vectors
        if len(quats) == len(pos):
            step_interval = 30
            indices = np.arange(0, len(pos), step_interval)
            
            if len(indices) > 0:
                sub_pos = pos[indices]
                sub_quats = quats[indices]
                headings = np.array([get_body_z_axis(q) for q in sub_quats])
                
                fig.add_trace(go.Cone(
                    x=sub_pos[:, 0], y=sub_pos[:, 1], z=sub_pos[:, 2],
                    u=headings[:, 0], v=headings[:, 1], w=headings[:, 2],
                    sizemode="scaled", sizeref=0.5, showscale=False,
                    anchor="tail", colorscale=[[0, color], [1, color]],
                    name=f'Ep {i+1} Heading'
                ))
        
        # Start/End Markers
        fig.add_trace(go.Scatter3d(
            x=[pos[0,0]], y=[pos[0,1]], z=[pos[0,2]],
            mode='markers',
            marker=dict(size=4, color=color, symbol='circle'),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter3d(
            x=[pos[-1,0]], y=[pos[-1,1]], z=[pos[-1,2]],
            mode='markers',
            marker=dict(size=6, color=color, symbol='x'),
            showlegend=False, hoverinfo='skip'
        ))

    # Environment
    theta = np.linspace(0, 2*np.pi, 50)
    r_pad = 1.0
    fig.add_trace(go.Scatter3d(
        x=r_pad * np.cos(theta), y=r_pad * np.sin(theta), z=np.zeros_like(theta),
        mode='lines', line=dict(color='black', width=4, dash='dash'), name='Landing Pad (1m)'
    ))
    
    r_zone = 5.0
    fig.add_trace(go.Scatter3d(
        x=r_zone * np.cos(theta), y=r_zone * np.sin(theta), z=np.zeros_like(theta),
        mode='lines', line=dict(color='orange', width=2, dash='dot'), name='Target Zone (5m)'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=5, color='black', symbol='diamond'), name='Target'
    ))

    fig.update_layout(
        title=f"🚀 {env_name} Trajectory Analysis (DQN)",
        width=1200, height=800,
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data', 
            xaxis=dict(gridcolor='lightgrey', backgroundcolor="white"),
            yaxis=dict(gridcolor='lightgrey', backgroundcolor="white"),
            zaxis=dict(gridcolor='grey', backgroundcolor="#F0F0F0"),
        )
    )

    filename = os.path.join(save_dir, f"interactive_trajectories_{env_name}.html")
    fig.write_html(filename)
    print(f"{Col.BOLD}🌎 Plot Saved: {Col.CYAN}{filename}{Col.RESET}")
    open_file(filename)

def plot_static_analysis(history, episode_num, save_dir):
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"DQN Analysis - Episode {episode_num}", fontsize=16)

    # 1. 3D Trajectory
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b')
    ax3d.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target')
    ax3d.legend()

    # 2. Position
    ax_pos = fig.add_subplot(2, 3, 2)
    ax_pos.set_title("Position")
    ax_pos.plot(times, pos[:, 2], label='Z (Alt)', color='green')
    ax_pos.plot(times, np.sqrt(pos[:,0]**2 + pos[:,1]**2), label='Lateral Error', color='orange', linestyle='--')
    ax_pos.legend(); ax_pos.grid(True)

    # 3. Orientation
    ax_att = fig.add_subplot(2, 3, 3)
    ax_att.set_title("Orientation (Tilt)")
    ax_att.plot(times, att[:, 1], label='Pitch')
    ax_att.plot(times, att[:, 2], label='Yaw')
    ax_att.legend(); ax_att.grid(True)

    # 4. Thrust
    ax_thr = fig.add_subplot(2, 3, 4)
    ax_thr.set_title("Thrust")
    ax_thr.plot(times, thrust, color='r')
    ax_thr.grid(True)

    # 5. Gimbal
    ax_gim = fig.add_subplot(2, 3, 5)
    ax_gim.set_title("Gimbal")
    ax_gim.plot(times, gimbal[:, 0], label='Yaw')
    ax_gim.plot(times, gimbal[:, 1], label='Pitch')
    ax_gim.legend(); ax_gim.grid(True)

    # 6. Mass
    ax_mass = fig.add_subplot(2, 3, 6)
    ax_mass.set_title("Mass")
    ax_mass.plot(times, mass, color='black')
    ax_mass.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"episode_{episode_num}_analysis.png")
    plt.savefig(save_path)
    plt.close(fig)

# ================================================================
#   DYNAMIC REWARD LOADER
# ================================================================
def load_reward_function(reward_name):
    try:
        module_path = f"rocket_env.rewards.{reward_name}"
        mod = importlib.import_module(module_path)
        return mod.compute_reward
    except ImportError as e:
        print(f"\n❌ Error loading reward: {reward_name}")
        raise e

# ================================================================
#   MAIN TESTING LOGIC
# ================================================================
def normalize_obs(obs, obs_rms, epsilon=1e-8):
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

def main():
    parser = argparse.ArgumentParser(description="Test a trained DQN Rocket Agent")
    parser.add_argument("run_path", type=str, help="Path to the run folder (e.g., runs/dqn_polar_...)")
    parser.add_argument("--model", type=str, default="best", choices=["best", "final", "latest"], help="Which model to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--reward", type=str, default="polar_vel_field", help="Name of reward file")

    # --- ARGS FOR DISCRETE WRAPPER RECONSTRUCTION ---
    # NOTE: These MUST match what you used in training!
    parser.add_argument("--thrust_bins", type=int, default=5)
    parser.add_argument("--pitch_bins", type=int, default=5)
    parser.add_argument("--roll_bins", type=int, default=5)
    parser.add_argument("--thrust_min", type=float, default=0.0)
    parser.add_argument("--thrust_max", type=float, default=1.0)
    parser.add_argument("--pitch_min", type=float, default=-0.2)
    parser.add_argument("--pitch_max", type=float, default=0.2)
    parser.add_argument("--roll_min", type=float, default=-0.2)
    parser.add_argument("--roll_max", type=float, default=0.2)
    
    args = parser.parse_args()

    # 1. Setup Directories
    if not os.path.exists(args.run_path):
        print(f"{Col.RED}Error: Run directory '{args.run_path}' not found.{Col.RESET}")
        return

    data_dir = os.path.join(args.run_path, "test_results_dqn")
    os.makedirs(data_dir, exist_ok=True)
    print(f"{Col.BOLD}📂 Saving test data to: {data_dir}{Col.RESET}")

    # 2. Load Reward Function
    reward_func = load_reward_function(args.reward)
    print(f"{Col.CYAN}💰 Loaded Reward Function: {args.reward}{Col.RESET}")

    # 3. Initialize Environments
    # We need the base environment for MuJoCo physics access
    base_env = RocketLandingEnv(
        render_mode="human" if not args.no_render else None,
        reward_func=reward_func
    )

    # We need the Wrapped environment for the Agent (Observation space matching)
    # This factory is needed for DummyVecEnv
    def make_wrapped_env():
        return DiscreteActionWrapper(
            RocketLandingEnv(reward_func=reward_func),
            thrust_bins=args.thrust_bins,
            pitch_bins=args.pitch_bins,
            roll_bins=args.roll_bins,
            thrust_range=(args.thrust_min, args.thrust_max),
            pitch_range=(args.pitch_min, args.pitch_max),
            roll_range=(args.roll_min, args.roll_max),
        )

    # We also wrap the 'base_env' strictly for stepping logic in the main loop
    # (Re-using the same wrapper class logic)
    wrapped_env = DiscreteActionWrapper(
        base_env,
        thrust_bins=args.thrust_bins,
        pitch_bins=args.pitch_bins,
        roll_bins=args.roll_bins,
        thrust_range=(args.thrust_min, args.thrust_max),
        pitch_range=(args.pitch_min, args.pitch_max),
        roll_range=(args.roll_min, args.roll_max),
    )

    # 4. Load Normalization Statistics
    norm_path = os.path.join(args.run_path, "vec_normalize.pkl")
    if not os.path.exists(norm_path):
        print(f"{Col.RED}Error: vec_normalize.pkl not found. Agent will fail.{Col.RESET}"); return

    # Load stats using a dummy vec env that mimics the TRAINING environment structure
    dummy_vec_env = DummyVecEnv([make_wrapped_env]) 
    vec_norm = VecNormalize.load(norm_path, dummy_vec_env)
    obs_rms = vec_norm.obs_rms
    print(f"{Col.GREEN}✅ Loaded Normalization Stats{Col.RESET}")

    # 5. Locate & Load DQN Model
    if args.model == "final":
        model_file = "final_model.zip"
        model_path = os.path.join(args.run_path, model_file)
    elif args.model == "best":
        possible_path = os.path.join(args.run_path, "best_model")
        if os.path.isdir(possible_path):
             model_path = os.path.join(possible_path, "best_model.zip")
        else:
             model_path = os.path.join(args.run_path, "best_model.zip")
    else: 
        ckpt_dir = os.path.join(args.run_path, "checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
            if ckpts:
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
    
    print(f"{Col.BOLD}🚀 Loading DQN Model: {model_path}{Col.RESET}")
    model = DQN.load(model_path)

    all_histories = []

    # 6. Testing Loop
    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        
        obs, _ = wrapped_env.reset()
        
        # --- CUSTOM RANDOMIZATION (MUJOCO LEVEL) ---
        # Note: We access data via base_env (the unwrapped RocketLandingEnv)
        qpos = base_env.data.qpos
        qvel = base_env.data.qvel
        
        # Randomize Position X (index 0)
        qpos[base_env.qpos_adr] = np.random.uniform(20, 25)
        
        # Randomize Velocity Magnitude
        # vel_mag = np.random.uniform(3, 4)
        # Apply velocity (if needed, uncomment below)
        # qvel[base_env.qvel_adr : base_env.qvel_adr+3] = [0, 0, -vel_mag] # Example

        # Apply to Physics Engine
        mujoco.mj_forward(base_env.model, base_env.data)
        
        # Refresh observation after manual MuJoCo modification
        # The wrapper might block direct calls, so we force a clean reset or get_obs if possible.
        # Ideally we call the internal _get_obs, but wrapped_env hides it.
        # We step with a 'no-op' or rely on the physics engine having updated.
        # For simplicity in testing, we grab the visual observation from physics directly or proceed.
        # If the environment allows, calling wrapped_env.reset() again overrides randomization.
        # The safest approach in a wrapped env without a specific API is to assume the next
        # step will correct the observation.
        
        # To be precise, let's manually re-construct the observation if we can,
        # otherwise we proceed (first frame might be slightly off).
        
        done = False
        step = 0
        total_reward = 0
        
        history = {
            'time': [], 'pos': [], 'vel': [], 'attitude': [], 'quat': [], 
            'thrust': [], 'gimbal': [], 'mass': [], 'reward': []
        }

        # Log initial state
        pos = base_env.data.xpos[base_env.rocket_bid].copy()
        vel = base_env.data.cvel[base_env.rocket_bid][3:].copy()
        quat = base_env.data.qpos[base_env.qpos_adr+3 : base_env.qpos_adr+7].copy()
        roll, pitch, yaw = quat_to_euler(quat)
        mass = base_env.DRY_MASS + base_env.fuel_mass
        
        history['time'].append(0)
        history['pos'].append(pos)
        history['vel'].append(vel)
        history['attitude'].append([roll, pitch, yaw])
        history['quat'].append(quat)
        history['thrust'].append(0.0)
        history['gimbal'].append([0.0, 0.0])
        history['mass'].append(mass)
        history['reward'].append(0.0)
        
        while True:
            step += 1
            
            # Normalize Observation (DQN expects normalized inputs if trained that way)
            norm_obs = normalize_obs(obs, obs_rms)
            
            # Predict (Deterministic=True generally maps to argmax(Q) in SB3 DQN)
            action, _ = model.predict(norm_obs, deterministic=True)
            
            # Step the WRAPPED environment (converts discrete -> continuous)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            total_reward += reward
            
            # Read Physics State from BASE ENV
            pos = base_env.data.xpos[base_env.rocket_bid].copy()
            vel = base_env.data.cvel[base_env.rocket_bid][3:].copy()
            quat = base_env.data.qpos[base_env.qpos_adr+3 : base_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            mass = base_env.DRY_MASS + base_env.fuel_mass
            
            thrust_N = base_env.data.ctrl[base_env.thrust_act]
            g_yaw = np.degrees(base_env.data.ctrl[base_env.yaw_act])
            g_pit = np.degrees(base_env.data.ctrl[base_env.pitch_act])
            
            history['time'].append(step * base_env.DT)
            history['pos'].append(pos)
            history['vel'].append(vel)
            history['attitude'].append([roll, pitch, yaw])
            history['quat'].append(quat)
            history['thrust'].append(thrust_N)
            history['gimbal'].append([g_yaw, g_pit])
            history['mass'].append(mass)
            history['reward'].append(reward)

            if not args.no_render:
                base_env.render()
                state_str = f"Alt:{pos[2]:5.1f}m Vz:{vel[2]:5.1f} Tlt:{max(abs(pitch),abs(roll)):4.1f}°"
                ctrl_str  = f"Thr:{thrust_N:6.0f}N"
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
        
        # End of Episode Stats
        final_pos = history['pos'][-1]
        dist_xy = np.sqrt(final_pos[0]**2 + final_pos[1]**2)
        
        is_success = info.get('success', False)
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

    base_env.close()

    print(f"\n{Col.BOLD}📊 Generating Interactive 3D Plot...{Col.RESET}")
    generate_interactive_plot(all_histories, data_dir, env_name="Polar Rocket (DQN)")
    
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    main()