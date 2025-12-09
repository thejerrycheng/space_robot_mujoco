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
import glob 

# CRITICAL FIX: Use 'Agg' backend to prevent macOS/Linux main-thread rendering crashes
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- IMPORT SAC MODEL INSTEAD OF PPO ---
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- IMPORT YOUR ENV CLASS ---
from rocket_env.rocket_gym_env import RocketLandingEnv 

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
#   FILE & PLOTTING FUNCTIONS (MODIFIED)
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
    reward = np.array(history['reward'])
    r_upright = np.array(history['r_upright'])
    r_vel = np.array(history['r_vel'])
    r_dist = np.array(history['r_dist'])

    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow([
            "Step", "Time", "X", "Y", "Z", "Vx", "Vy", "Vz",
            "Roll", "Pitch", "Yaw", 
            "Thrust", "GimbalYaw", "GimbalPitch", 
            "Mass", "TotalReward", "R_Upright", "R_Velocity", "R_Distance"
        ])
        
        # Rows
        for i in range(len(times)):
            writer.writerow([
                i, times[i], 
                pos[i,0], pos[i,1], pos[i,2],
                vel[i,0], vel[i,1], vel[i,2],
                att[i,0], att[i,1], att[i,2],
                thrust[i], gimbal[i,0], gimbal[i,1],
                mass[i], reward[i], r_upright[i], r_vel[i], r_dist[i]
            ])
            
    print(f"💾 Data saved to: {filename}")

def plot_unified_analysis(history, episode_num, model_name, save_dir):
    """ Generates a single figure with 6 subplots as requested, including the 80m tolerance circle. """
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    quat = np.array(history['quat'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    total_reward = np.array(history['reward'])
    r_upright = np.array(history['r_upright'])
    r_vel = np.array(history['r_vel'])
    r_dist = np.array(history['r_dist'])
    
    # 3. Tilt of the rocket over time
    tilt_mag = np.sqrt(att[:, 0]**2 + att[:, 1]**2)
    
    # 4. Distance to target and altitude of the rocket
    lateral_dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    altitude = pos[:, 2]

    # Create the figure with 2 rows and 3 columns
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"SAC Analysis: {model_name} - Episode {episode_num}", fontsize=16)

    # 1. Overall 3D Trajectory of the rocket with orientation arrows (MODIFIED)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_title("1. 3D Trajectory & Orientation")
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b')
    ax1.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target')
    
    # --- NEW: Add 80-meter Radius Tolerance Circle ---
    R_TOLERANCE = 80.0
    # Generate points for a circle in the XY plane (Z=0)
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = R_TOLERANCE * np.cos(theta)
    y_circle = R_TOLERANCE * np.sin(theta)
    z_circle = np.zeros_like(theta)
    
    ax1.plot(x_circle, y_circle, z_circle, color='orange', linestyle='--', 
             label=f'{R_TOLERANCE}m Tolerance Zone')
    # --------------------------------------------------
    
    # Add orientation arrows (heading vectors)
    step_interval = max(1, len(pos) // 50) # Plot about 50 arrows
    indices = np.arange(0, len(pos), step_interval)
    for i in indices:
        vec = get_body_z_axis(quat[i]) # Body Z-axis in World frame
        ax1.quiver(pos[i, 0], pos[i, 1], pos[i, 2], 
                   vec[0], vec[1], vec[2], 
                   length=10, normalize=True, color='r', arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m - Altitude)')
    ax1.legend()
    # Ensure the plot limits are set to capture the 80m circle for a complete view
    max_range = max(R_TOLERANCE, np.max(np.abs(pos))) * 1.1
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([0, np.max(pos[:,2]) * 1.1])
    ax1.set_box_aspect([1,1,1])

    # 2. The mass of the rocket over time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("2. Rocket Mass Over Time")
    ax2.plot(times, mass, color='black')
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mass (kg)"); ax2.grid(True)
    
    # 3. The tilt of the rocket over time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("3. Rocket Tilt (Angular Deviation)")
    ax3.plot(times, tilt_mag, label='Tilt Magnitude (Roll/Pitch)', color='purple')
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Tilt (Deg)"); ax3.grid(True)

    # 4. The distance to the target and the altitude of the rocket
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("4. Position: Distance to Target & Altitude")
    ax4.plot(times, altitude, label='Altitude (Z)', color='green')
    ax4.plot(times, lateral_dist, label='Lateral Distance to Target', color='orange', linestyle='--')
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Distance (m)"); ax4.legend(); ax4.grid(True)

    # 5. Control: Thrust magnitude & Gimbal pitch/roll angles (Normalized Actions)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("5. Control Commands (Normalized Actions [-1, 1])")
    ax5.plot(times, thrust, label='Thrust Command (Norm.)', color='r')
    ax5.plot(times, gimbal[:, 1], label='Gimbal Pitch Command', color='b')
    ax5.plot(times, gimbal[:, 0], label='Gimbal Yaw Command', color='c', linestyle=':')
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Normalized Action"); ax5.legend(); ax5.grid(True)
    ax5.set_ylim([-1.05, 1.05])

    # 6. The reward over time including components
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("6. Reward Breakdown Over Time")
    ax6.plot(times, total_reward, label='Total Reward', color='k', linewidth=2)
    ax6.plot(times, r_upright, label='Upright Reward', linestyle='--', color='g')
    ax6.plot(times, r_vel, label='Velocity Reward', linestyle='--', color='b')
    ax6.plot(times, r_dist, label='Position Reward', linestyle='--', color='orange')
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Reward Value"); ax6.legend(); ax6.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    model_dir = os.path.basename(os.path.normpath(save_dir))
    save_path = os.path.join(save_dir, f"episode_{episode_num}_unified_analysis.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"🖼️ Unified Plot saved to: {save_path}")


# ================================================================
#   DYNAMIC REWARD LOADER (Same as original)
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
#   MAIN TESTING LOGIC (Same as previous revised script)
# ================================================================
def normalize_obs(obs, obs_rms, epsilon=1e-8):
    """ Normalize observation using loaded statistics """
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

def main():
    parser = argparse.ArgumentParser(description="Test a trained SAC Rocket Agent")
    parser.add_argument("run_path", type=str, help="Path to the SAC run folder (e.g., models/gym_sac_X_date)")
    parser.add_argument("--model", type=str, default="best", choices=["best", "final", "latest"], help="Which model to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes to visualize")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--reward", type=str, default="upright_only", help="Name of reward file in rocket_env/rewards/")
    
    args = parser.parse_args()
    
    model_name = os.path.basename(os.path.normpath(args.run_path))

    # 1. Setup Directories
    if not os.path.exists(args.run_path):
        print(f"{Col.RED}Error: Run directory '{args.run_path}' not found.{Col.RESET}")
        return

    base_plot_dir = os.path.join("plots", "test")
    data_dir = os.path.join(base_plot_dir, model_name)
    os.makedirs(data_dir, exist_ok=True)
    print(f"{Col.BOLD}📂 Saving test data/plots to: {data_dir}{Col.RESET}")

    # 2. Load Reward Class
    try:
        RewardClass = load_reward_class(args.reward)
        rewarder = RewardClass({'start_fuel': 4_500_000.0}) 
        print(f"{Col.CYAN}💰 Loaded Reward Class: {args.reward}{Col.RESET}")
    except Exception as e:
        print(f"{Col.RED}Error initializing rewarder: {e}{Col.RESET}")
        return

    # 3. Load Normalization Statistics
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
    model = SAC.load(model_path) 

    # 5. Create RAW Environment for Testing
    real_env = RocketLandingEnv(
        render_mode="human" if not args.no_render else None
    )
    if hasattr(real_env, "rewarder"):
        real_env.rewarder = rewarder
    
    real_env.rocket_bid = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    real_env.qpos_adr = 0
    real_env.qvel_adr = 0
    real_env.thrust_act = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
    real_env.yaw_act = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
    real_env.pitch_act = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
    real_env.DT = real_env.model.opt.timestep

    all_histories = []

    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        
        obs, _ = real_env.reset()
        
        mujoco.mj_forward(real_env.model, real_env.data)
        try: obs = real_env._get_obs()
        except AttributeError: pass

        step = 0
        total_reward = 0
        
        history = {
            'time': [], 'pos': [], 'vel': [], 'attitude': [], 'quat': [], 
            'thrust': [], 'gimbal': [], 'mass': [], 'reward': [],
            'r_upright': [], 'r_vel': [], 'r_dist': [] 
        }

        while True:
            step += 1
            
            norm_obs = normalize_obs(obs, obs_rms) if obs_rms else obs
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = real_env.step(action)
            total_reward += reward
            
            # 9. Log State
            pos = real_env.data.xpos[real_env.rocket_bid].copy()
            vel = real_env.data.qvel[0:3].copy() 
            quat = real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            mass = real_env.DRY_MASS + real_env.fuel_mass
            
            thrust_N_cmd = real_env.data.ctrl[real_env.thrust_act] 
            g_yaw_cmd = real_env.data.ctrl[real_env.yaw_act]
            g_pit_cmd = real_env.data.ctrl[real_env.pitch_act]
            
            # --- EXTRACT REWARD BREAKDOWN ---
            r_up = info.get('r_upright', 0.0)
            r_vel = info.get('r_velocity', 0.0)
            r_dist = info.get('r_distance', 0.0)
            r_step = info.get('r_step', 0.0)
            r_term = info.get('terminal_reward', 0.0)
            
            history['time'].append(step * real_env.DT)
            history['pos'].append(pos)
            history['vel'].append(vel)
            history['attitude'].append([roll, pitch, yaw])
            history['quat'].append(quat) 
            history['thrust'].append(thrust_N_cmd)
            history['gimbal'].append([g_yaw_cmd, g_pit_cmd])
            history['mass'].append(mass)
            history['reward'].append(reward)
            history['r_upright'].append(r_up)
            history['r_vel'].append(r_vel)
            history['r_dist'].append(r_dist)

            if not args.no_render:
                real_env.render()
                
                state_str = f"Alt:{pos[2]:5.1f}m Dis:{np.sqrt(pos[0]**2 + pos[1]**2):5.1f}m Vel:{np.linalg.norm(vel):5.1f} Vz:{vel[2]:5.1f} Tlt:{max(abs(pitch),abs(roll)):4.1f}°"
                ctrl_str  = f"Thr:{thrust_N_cmd:6.2f} GimY:{g_yaw_cmd:4.2f}"
                reward_total_str = f"Total:{reward:6.2f}"
                reward_breakdown_str = (
                    f" Reward Upright:{r_up:5.2f}"
                    f" Reward Velocity:{r_vel:5.2f}"
                    f" Reward Distance:{r_dist:5.2f}"
                    f" Reward Step:{r_step:4.2f}"
                    f" Reward Terminal:{r_term:6.2f}"
                )

                log_line = (
                    f"\r{step:04} | {Col.CYAN}{state_str}{Col.RESET} | "
                    f"{Col.YELLOW}{ctrl_str}{Col.RESET} | "
                    f"{Col.GREEN}{reward_total_str}{Col.RESET} | "
                    f"{reward_breakdown_str}\033[K"
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
        
        is_success = info.get('is_success', False)
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
        plot_unified_analysis(history, ep+1, model_name, data_dir)

    real_env.close()

    print(f"\n{Col.BOLD}📊 Generating Interactive 3D Plot...{Col.RESET}")
    # generate_interactive_plot must be defined or imported for this to work
    # generate_interactive_plot(all_histories, data_dir, env_name="SAC Rocket Landing")
    
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    main()