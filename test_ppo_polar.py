#!/usr/bin/env python3
"""
Rocket Landing Agent - Test Script
Tests the Polar Rocket Agent with Body-Z aligned to Velocity.
"""
import os
import sys
import time
import argparse
import numpy as np
import subprocess
import csv
import importlib
import mujoco

# Use 'Agg' backend to prevent macOS/Linux main-thread rendering crashes
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# RL Libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import Your Polar Environment
from rocket_env.polar_rocket_env import RocketLandingEnv

# ================================================================
#   UTILITIES
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
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0: w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
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
    w, x, y, z = quat
    vec_x = 2 * (w*y + x*z)
    vec_y = 2 * (y*z - w*x)
    vec_z = 1 - 2 * (x*x + y*y)
    return np.array([vec_x, vec_y, vec_z])

def get_quat_align_z_to_vel(velocity_vector):
    """
    Calculates the quaternion [w, x, y, z] required to rotate the 
    Body Z-axis [0, 0, 1] to align with the given velocity vector.
    """
    # Normalize target vector
    v_norm = np.linalg.norm(velocity_vector)
    if v_norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0]) # Identity if no velocity
    
    target = velocity_vector / v_norm
    source = np.array([0.0, 0.0, 1.0]) # Body Z axis
    
    # 1. Compute rotation axis (cross product)
    axis = np.cross(source, target)
    axis_len = np.linalg.norm(axis)
    
    # 2. Handle parallel/anti-parallel cases
    dot = np.dot(source, target)
    
    if axis_len < 1e-6:
        # Vectors are parallel
        if dot > 0: return np.array([1.0, 0.0, 0.0, 0.0]) # Same direction
        else:       return np.array([0.0, 1.0, 0.0, 0.0]) # Opposite (180 deg about X)

    # 3. Compute Quaternion (Axis-Angle)
    # Angle between vectors
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    # Convert to quaternion [w, x, y, z]
    # w = cos(theta/2), xyz = axis * sin(theta/2)
    axis = axis / axis_len
    w = np.cos(angle / 2.0)
    xyz = axis * np.sin(angle / 2.0)
    
    return np.array([w, xyz[0], xyz[1], xyz[2]])

def open_file(path):
    try:
        if sys.platform == "darwin": subprocess.call(["open", path])
        elif sys.platform == "win32": os.startfile(path)
        else: subprocess.call(["xdg-open", path])
    except Exception: pass

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
#   PLOTTING & SAVING
# ================================================================
def save_to_csv(history, episode_num, save_dir):
    filename = os.path.join(save_dir, f"episode_{episode_num}.csv")
    data_len = len(history['time'])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Step", "Time", "X", "Y", "Z", "Vx", "Vy", "Vz", 
                   "Roll", "Pitch", "Yaw", "Thrust", "GimbalYaw", "GimbalPitch", "Mass", "Reward"]
        writer.writerow(headers)
        
        for i in range(data_len):
            pos, vel = history['pos'][i], history['vel'][i]
            att, gim = history['attitude'][i], history['gimbal'][i]
            writer.writerow([
                i, history['time'][i], pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                att[0], att[1], att[2], history['thrust'][i], gim[0], gim[1],
                history['mass'][i], history['reward'][i]
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
    fig.suptitle(f"Polar Rocket Analysis - Episode {episode_num}", fontsize=16)

    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b')
    ax3d.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target')
    ax3d.legend()

    ax_pos = fig.add_subplot(2, 3, 2); ax_pos.set_title("Position")
    ax_pos.plot(times, pos[:, 2], label='Z (Alt)', color='green')
    ax_pos.plot(times, np.sqrt(pos[:,0]**2 + pos[:,1]**2), label='Lateral Error', color='orange', linestyle='--')
    ax_pos.legend(); ax_pos.grid(True)

    ax_att = fig.add_subplot(2, 3, 3); ax_att.set_title("Orientation (Deg)")
    ax_att.plot(times, att[:, 1], label='Pitch'); ax_att.plot(times, att[:, 2], label='Yaw')
    ax_att.legend(); ax_att.grid(True)

    ax_thr = fig.add_subplot(2, 3, 4); ax_thr.set_title("Thrust (N)")
    ax_thr.plot(times, thrust, color='r'); ax_thr.grid(True)

    ax_gim = fig.add_subplot(2, 3, 5); ax_gim.set_title("Gimbal (Deg)")
    ax_gim.plot(times, gimbal[:, 0], label='Yaw'); ax_gim.plot(times, gimbal[:, 1], label='Pitch')
    ax_gim.legend(); ax_gim.grid(True)

    ax_mass = fig.add_subplot(2, 3, 6); ax_mass.set_title("Mass (kg)")
    ax_mass.plot(times, mass, color='black'); ax_mass.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"episode_{episode_num}_analysis.png"))
    plt.close(fig)

def generate_interactive_plot(all_histories, save_dir="."):
    try:
        import plotly.graph_objects as go
        import plotly.colors as pc
    except ImportError: return

    print(f"\n{Col.BOLD}📊 Generating Interactive Plotly HTML...{Col.RESET}")
    fig = go.Figure()
    palette = pc.qualitative.Plotly 

    for i, history in enumerate(all_histories):
        pos = np.array(history['pos'])
        quats = np.array(history.get('quat', []))
        if len(pos) < 2: continue 
        
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter3d(x=pos[:,0], y=pos[:,1], z=pos[:,2], mode='lines', 
                                   name=f'Ep {i+1}', line=dict(width=5, color=color), opacity=0.8))
        
        if len(quats) == len(pos):
            step_int = max(1, len(pos)//20)
            idx = np.arange(0, len(pos), step_int)
            sub_pos, sub_quats = pos[idx], quats[idx]
            headings = np.array([get_body_z_axis(q) for q in sub_quats])
            fig.add_trace(go.Cone(x=sub_pos[:,0], y=sub_pos[:,1], z=sub_pos[:,2],
                                  u=headings[:,0], v=headings[:,1], w=headings[:,2],
                                  sizemode="scaled", sizeref=0.5, showscale=False,
                                  colorscale=[[0, color], [1, color]], anchor="tail", name=f'Ep {i+1} Heading'))

    theta = np.linspace(0, 2*np.pi, 50)
    fig.add_trace(go.Scatter3d(x=1.0*np.cos(theta), y=1.0*np.sin(theta), z=np.zeros_like(theta),
        mode='lines', line=dict(color='black', dash='dash'), name='Pad (1m)'))
    
    fig.update_layout(title="Rocket Trajectories (Fixed Conditions)", width=1200, height=800, scene=dict(aspectmode='data'))
    filename = os.path.join(save_dir, "interactive_trajectories.html")
    fig.write_html(filename)
    open_file(filename)

# ================================================================
#   MAIN
# ================================================================
def normalize_obs(obs, obs_rms, epsilon=1e-8):
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

def main():
    parser = argparse.ArgumentParser(description="Test Polar Rocket Agent - Fixed Conditions")
    parser.add_argument("run_path", type=str, help="Path to run folder")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--reward", type=str, default="polar_vel_field") 
    args = parser.parse_args()

    if not os.path.exists(args.run_path): print(f"{Col.RED}Path not found.{Col.RESET}"); return
    
    data_dir = os.path.join(args.run_path, "test_fixed_conditions")
    os.makedirs(data_dir, exist_ok=True)
    
    reward_func = load_reward_function(args.reward)
    
    norm_path = os.path.join(args.run_path, "vec_normalize.pkl")
    dummy_env = DummyVecEnv([lambda: RocketLandingEnv(reward_func=reward_func)])
    vec_norm = VecNormalize.load(norm_path, dummy_env)
    obs_rms = vec_norm.obs_rms

    model_path = os.path.join(args.run_path, "final_model.zip") 
    if os.path.exists(os.path.join(args.run_path, "best_model.zip")):
        model_path = os.path.join(args.run_path, "best_model.zip")
        
    print(f"{Col.BOLD}🚀 Loading Model: {model_path}{Col.RESET}")
    model = PPO.load(model_path)

    real_env = RocketLandingEnv(render_mode="human" if not args.no_render else None, reward_func=reward_func)

    # --- CONFIGURATION CONSTANTS (From Prompt) ---
    INIT_RADIUS = 15.0
    INIT_HEIGHT = 10.0
    INITIAL_SPEED = 5.0
    
    # NOTE: Roll/Yaw are now derived from the velocity vector directly

    all_histories = []

    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        
        obs, _ = real_env.reset()
        
        # =========================================================================
        # 🔒 FIXED INITIAL CONDITIONS (NO DISTRIBUTIONS)
        # =========================================================================
        # 1. Position: r=15, h=10.
        #    We place the rocket at x=15.0, y=0.0 to satisfy r=15.0
        real_env.data.qpos[real_env.qpos_adr] = INIT_RADIUS     # X
        real_env.data.qpos[real_env.qpos_adr+1] = 0.0           # Y
        real_env.data.qpos[real_env.qpos_adr+2] = INIT_HEIGHT   # Z (Height)

        # 2. Velocity: Speed = 5.0
        #    We direct the velocity towards the landing pad (Origin).
        #    Since we are at X=15, vector towards 0 is [-1, 0, 0].
        vx = -INITIAL_SPEED
        vy = 0.0
        vz = 0.0
        
        real_env.data.qvel[real_env.qvel_adr]   = vx
        real_env.data.qvel[real_env.qvel_adr+1] = vy
        real_env.data.qvel[real_env.qvel_adr+2] = vz

        # 3. Orientation: ALIGN Z-AXIS TO VELOCITY
        #    We calculate the quaternion needed to rotate [0,0,1] to [vx, vy, vz]
        vel_vector = np.array([vx, vy, vz])
        quat = get_quat_align_z_to_vel(vel_vector)
        
        real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7] = quat

        # Apply changes to Physics
        mujoco.mj_forward(real_env.model, real_env.data)
        
        # Refresh Observation to match new state
        try: obs = real_env._get_obs()
        except AttributeError: pass
        # =========================================================================

        done = False
        step = 0
        total_reward = 0
        history = {'time':[], 'pos':[], 'vel':[], 'attitude':[], 'quat':[], 'thrust':[], 'gimbal':[], 'mass':[], 'reward':[]}

        while True:
            step += 1
            norm_obs = normalize_obs(obs, obs_rms)
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = real_env.step(action)
            total_reward += reward

            pos = real_env.data.xpos[real_env.rocket_bid].copy()
            vel = real_env.data.cvel[real_env.rocket_bid][3:].copy()
            quat = real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            mass = real_env.DRY_MASS + real_env.fuel_mass
            
            history['time'].append(step * real_env.DT)
            history['pos'].append(pos); history['vel'].append(vel)
            history['attitude'].append([roll, pitch, yaw]); history['quat'].append(quat)
            history['thrust'].append(real_env.data.ctrl[real_env.thrust_act])
            history['gimbal'].append([np.degrees(real_env.data.ctrl[real_env.yaw_act]), np.degrees(real_env.data.ctrl[real_env.pitch_act])])
            history['mass'].append(mass); history['reward'].append(reward)

            if not args.no_render:
                real_env.render()
                state_str = f"Alt:{pos[2]:5.1f}m Vz:{vel[2]:5.1f} Roll:{roll:4.1f}°"
                print(f"\r{step:04} | {Col.CYAN}{state_str}{Col.RESET} | Rew:{reward:6.2f}", end="")
                time.sleep(0.01)

            if terminated or truncated: break
        
        dist_xy = np.sqrt(history['pos'][-1][0]**2 + history['pos'][-1][1]**2)
        msg = f"{Col.GREEN}SUCCESS" if info.get('success') else f"{Col.RED}FAILURE (Dist:{dist_xy:.1f}m)"
        print(f"\n{msg}{Col.RESET} | Total Reward: {total_reward:.2f}")
        
        all_histories.append(history)
        save_to_csv(history, ep+1, data_dir)
        plot_static_analysis(history, ep+1, data_dir)

    real_env.close()
    generate_interactive_plot(all_histories, data_dir)
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    main()