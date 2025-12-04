import os
import sys
import time
import argparse
import numpy as np
import subprocess
import csv
import matplotlib
import importlib
import gymnasium as gym
from gymnasium import spaces

# Use Agg backend for stability
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rocket_env.rocket_2_env import RocketLandingEnv

# ==============================================================================
#   PAPER OBSERVATION WRAPPER (Must Match Training!)
# ==============================================================================
class PaperObsWrapper(gym.Wrapper):
    """
    Wraps the RocketLandingEnv to produce the specific observation vector 
    described in Equation 31 of the paper:
    obs = [v_error, q, omega, r_z, t_go]
    """
    def __init__(self, env):
        super().__init__(env)
        # New Observation Space: 3 (v_err) + 4 (q) + 3 (w) + 1 (r_z) + 1 (t_go) = 12 dims
        high = np.inf * np.ones(12, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Guidance Parameters
        self.v0 = 70.0        
        self.waypoint_z = 15.0 
        self.tau_1 = 20.0     
        self.tau_2 = 100.0    

    def _get_paper_obs(self):
        unwrapped_env = self.env.unwrapped
        data = unwrapped_env.data
        
        # 1. Extract Raw State
        pos = data.xpos[unwrapped_env.rocket_bid].copy()
        vel = data.cvel[unwrapped_env.rocket_bid][3:].copy()
        quat = data.qpos[unwrapped_env.qpos_adr+3 : unwrapped_env.qpos_adr+7].copy()
        omega = data.qvel[unwrapped_env.qvel_adr+3 : unwrapped_env.qvel_adr+6].copy()
        r_z = pos[2]
        
        # 2. Time-to-Go
        r_mag = np.linalg.norm(pos)
        v_mag = np.linalg.norm(vel) + 1e-6
        t_go = r_mag / v_mag
        
        # 3. Compute v_targ (Guidance Law)
        if r_z > self.waypoint_z:
            r_rel = pos.copy()
            r_rel[2] -= self.waypoint_z
            r_unit = r_rel / (np.linalg.norm(r_rel) + 1e-6)
            tau = self.tau_1
            factor = 1.0 - np.exp(-t_go / tau)
            v_targ = -self.v0 * r_unit * factor
        else:
            v_targ = np.array([0.0, 0.0, -2.0])

        # 4. Velocity Error
        v_error = vel - v_targ
        
        # 5. Assemble
        obs = np.concatenate([v_error, quat, omega, [r_z], [t_go]]).astype(np.float32)
        return obs

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        obs = self._get_paper_obs()
        return obs, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_paper_obs()
        return obs, reward, terminated, truncated, info

# ==============================================================================
#   UTILITIES
# ==============================================================================
class Col:
    RESET = '\033[0m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'

def quat_to_euler(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp) if np.abs(sinp) < 1 else np.copysign(np.pi / 2, sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(np.array([roll, pitch, yaw]))

def get_body_z_axis(quat):
    """ Calculates the Body Z-axis vector in World coordinates from a quaternion [w,x,y,z]. """
    w, x, y, z = quat
    # Formula for the 3rd column of the rotation matrix
    # This assumes Body Z is the thrust/longitudinal axis
    vec_x = 2 * (w*y + x*z)
    vec_y = 2 * (y*z - w*x)
    vec_z = 1 - 2 * (x*x + y*y)
    return np.array([vec_x, vec_y, vec_z])

def open_file(path):
    try:
        if sys.platform == "darwin": subprocess.call(["open", path])
        elif sys.platform == "win32": os.startfile(path)
        else: subprocess.call(["xdg-open", path])
    except: pass

def load_reward_function(reward_name):
    try:
        module_path = f"rocket_env.rewards.{reward_name}"
        mod = importlib.import_module(module_path)
        return mod.compute_reward
    except ImportError as e:
        print(f"Error loading reward: {reward_name}")
        raise e

# ==============================================================================
#   PLOTTING FUNCTIONS
# ==============================================================================
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

def plot_static_analysis(history, episode_num, save_dir):
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Paper Policy Analysis - Episode {episode_num}", fontsize=16)

    # 1. 3D Trajectory
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b')
    ax3d.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.legend()

    # 2. Position
    ax_pos = fig.add_subplot(2, 3, 2)
    ax_pos.set_title("Altitude & Lateral Error")
    ax_pos.plot(times, pos[:, 2], label='Altitude (Z)', color='green')
    ax_pos.plot(times, np.sqrt(pos[:,0]**2 + pos[:,1]**2), label='Lateral Error', color='orange', linestyle='--')
    ax_pos.axhline(0, color='k', linestyle='-')
    ax_pos.legend(); ax_pos.grid(True)

    # 3. Attitude
    ax_att = fig.add_subplot(2, 3, 3)
    ax_att.set_title("Attitude (Deg)")
    ax_att.plot(times, att[:, 0], label='Roll')
    ax_att.plot(times, att[:, 1], label='Pitch')
    ax_att.set_ylabel("Deg"); ax_att.legend(); ax_att.grid(True)

    # 4. Thrust
    ax_thr = fig.add_subplot(2, 3, 4)
    ax_thr.set_title("Total Thrust Command")
    ax_thr.plot(times, thrust, color='r')
    ax_thr.set_ylabel("N"); ax_thr.grid(True)

    # 5. Gimbal
    ax_gim = fig.add_subplot(2, 3, 5)
    ax_gim.set_title("Gimbal Deflection")
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

# ==============================================================================
#   INTERACTIVE PLOTTING (PLOTLY)
# ==============================================================================
def generate_interactive_plot(all_histories, save_dir=".", env_name="Rocket Env"):
    """ 
    Generates a high-fidelity interactive 3D plot using Plotly.
    Includes rocket heading vectors plotted at intervals.
    """
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
        quats = np.array(history['quat_raw'])
        
        if len(pos) < 2: continue 

        steps = np.arange(len(pos))
        vel_z = np.array(history['vel'])[:, 2] 
        
        # Calculate tilt from attitude (Roll/Pitch)
        att = np.array(history['attitude']) # [Roll, Pitch, Yaw]
        tilt = np.maximum(np.abs(att[:,0]), np.abs(att[:,1]))
        
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
        # Subsample to avoid visual clutter (e.g., every 30 steps)
        step_interval = 30
        indices = np.arange(0, len(pos), step_interval)
        
        if len(indices) > 0:
            sub_pos = pos[indices]
            sub_quats = quats[indices]
            
            # Calculate heading vectors for these points
            headings = np.array([get_body_z_axis(q) for q in sub_quats])
            
            # Scale vectors for visibility
            scale = 5.0
            
            fig.add_trace(go.Cone(
                x=sub_pos[:, 0], y=sub_pos[:, 1], z=sub_pos[:, 2],
                u=headings[:, 0], v=headings[:, 1], w=headings[:, 2],
                sizemode="scaled",
                sizeref=0.5, # Adjust cone size
                showscale=False,
                anchor="tail",
                colorscale=[[0, color], [1, color]], # Match trajectory color
                name=f'Ep {i+1} Heading'
            ))
        
        # 3. Start/End Markers
        fig.add_trace(go.Scatter3d(
            x=[pos[0,0]], y=[pos[0,1]], z=[pos[0,2]],
            mode='markers', marker=dict(size=4, color=color, symbol='circle'),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter3d(
            x=[pos[-1,0]], y=[pos[-1,1]], z=[pos[-1,2]],
            mode='markers', marker=dict(size=6, color=color, symbol='x'),
            showlegend=False, hoverinfo='skip'
        ))

    # Context (Pad + Target)
    theta = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter3d(
        x=5.0 * np.cos(theta), y=5.0 * np.sin(theta), z=np.zeros_like(theta),
        mode='lines', line=dict(color='black', width=4, dash='dash'), name='Target Zone (5m)'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers', 
        marker=dict(size=5, color='black', symbol='diamond'), name='Target'
    ))

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

    filename = os.path.join(save_dir, f"interactive_trajectories.html")
    fig.write_html(filename)
    print(f"{Col.BOLD}🌎 Plot Saved: {Col.CYAN}{filename}{Col.RESET}")
    open_file(filename)

# ==============================================================================
#   MAIN TESTING LOGIC
# ==============================================================================
def normalize_obs(obs, obs_rms, epsilon=1e-8):
    """ Manually normalize observation using training stats. """
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

def main():
    parser = argparse.ArgumentParser(description="Test PPO Agent with Paper Observations")
    parser.add_argument("run_path", type=str, help="Path to the run folder (e.g., runs/ppo_paper_...)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--reward", type=str, default="flip_and_fuel", help="Reward function used")
    args = parser.parse_args()

    # 1. Setup
    if not os.path.exists(args.run_path):
        print(f"{Col.RED}Error: Run path not found.{Col.RESET}"); return

    data_dir = os.path.join(args.run_path, "test_results_paper")
    os.makedirs(data_dir, exist_ok=True)
    
    # 2. Load Normalization Stats
    norm_path = os.path.join(args.run_path, "vec_normalize.pkl")
    if not os.path.exists(norm_path):
        print(f"{Col.RED}Error: vec_normalize.pkl not found.{Col.RESET}"); return

    reward_func = load_reward_function(args.reward)
    
    # Dummy creation to load stats
    def make_env():
        e = RocketLandingEnv(reward_func=reward_func)
        return PaperObsWrapper(e)
    
    dummy_vec_env = DummyVecEnv([make_env]) 
    vec_norm = VecNormalize.load(norm_path, dummy_vec_env)
    obs_rms = vec_norm.obs_rms
    print(f"{Col.GREEN}✅ Loaded Normalization Statistics{Col.RESET}")

    # 3. Load Model
    model_path = os.path.join(args.run_path, "final_model.zip")
    if not os.path.exists(model_path):
        ckpt_dir = os.path.join(args.run_path, "checkpoints")
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
            if ckpts:
                ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
                model_path = os.path.join(ckpt_dir, ckpts[-1])
            else: print("No checkpoints found."); return
        else: print("No model found."); return

    print(f"{Col.BOLD}🚀 Loading Model: {model_path}{Col.RESET}")
    model = PPO.load(model_path)

    # 4. Create Evaluation Environment
    raw_env = RocketLandingEnv(render_mode="human" if not args.no_render else None, reward_func=reward_func)
    env = PaperObsWrapper(raw_env)

    all_histories = []

    # 5. Test Loop
    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        history = {
            'time': [], 'pos': [], 'vel': [], 'attitude': [], 
            'thrust': [], 'gimbal': [], 'mass': [], 'reward': [],
            'quat_raw': [] # Need raw quat for plotting headings
        }

        while True:
            step += 1
            
            norm_obs = normalize_obs(obs, obs_rms)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Access internal mujoco data
            unwrapped = env.unwrapped
            pos = unwrapped.data.xpos[unwrapped.rocket_bid].copy()
            vel = unwrapped.data.cvel[unwrapped.rocket_bid][3:].copy()
            quat = unwrapped.data.qpos[unwrapped.qpos_adr+3 : unwrapped.qpos_adr+7].copy()
            
            thrust_val = unwrapped.data.ctrl[unwrapped.thrust_act]
            g_yaw = np.degrees(unwrapped.data.ctrl[unwrapped.yaw_act])
            g_pit = np.degrees(unwrapped.data.ctrl[unwrapped.pitch_act])
            mass_val = unwrapped.DRY_MASS + unwrapped.fuel_mass

            history['time'].append(step * unwrapped.DT)
            history['pos'].append(pos)
            history['vel'].append(vel)
            history['attitude'].append(quat_to_euler(quat))
            history['quat_raw'].append(quat) # Store for heading calc
            history['thrust'].append(thrust_val)
            history['gimbal'].append([g_yaw, g_pit])
            history['mass'].append(mass_val)
            history['reward'].append(reward)

            if not args.no_render:
                env.render()
                r, p, y = quat_to_euler(quat)
                state_str = f"Alt:{pos[2]:5.1f}m Vz:{vel[2]:5.1f} Tlt:{max(abs(r), abs(p)):4.1f}°"
                print(f"\r{step:04} | {Col.CYAN}{state_str}{Col.RESET} | Rew:{reward:6.2f}", end="")
                time.sleep(0.01) 

            if terminated or truncated:
                break
        
        all_histories.append(history)

        # End of Episode Logging
        final_pos = history['pos'][-1]
        dist_xy = np.sqrt(final_pos[0]**2 + final_pos[1]**2)
        
        # Check standard success first (which implies stability + location)
        is_success = info.get('success', False)
        
        # Semi-Success: In target zone but crashed/unstable
        # Note: We enforce dist_xy < 5.0m as the "Target Area" (matches paper)
        is_semi_success = (dist_xy < 5.0) and not is_success

        result_msg = "❌ FAILURE"
        if is_success:
            result_msg = f"{Col.GREEN}✅ SUCCESS{Col.RESET}"
        elif is_semi_success:
            result_msg = f"{Col.YELLOW}⚠️ SEMI-SUCCESS (In Zone: {dist_xy:.2f}m){Col.RESET}"
        else:
            result_msg = f"{Col.RED}❌ FAILURE (Dist: {dist_xy:.2f}m){Col.RESET}"

        print(f"\n{result_msg} | Total Reward: {total_reward:.2f}")
        
        # Save Artifacts
        save_to_csv(history, ep+1, data_dir)
        plot_static_analysis(history, ep+1, data_dir)

    env.close()
    
    # Interactive Plot
    print(f"\n{Col.BOLD}📊 Generating Interactive 3D Plot...{Col.RESET}")
    generate_interactive_plot(all_histories, data_dir)

    print("\n👋 Testing complete.")

if __name__ == "__main__":
    main()