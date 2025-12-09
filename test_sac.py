import os
import sys
import time
import argparse
import numpy as np
import subprocess  # For opening the saved plot
import matplotlib

# CRITICAL FIX: Use 'Agg' backend to prevent macOS main-thread crashes
# This must be set BEFORE importing pyplot
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

from stable_baselines3 import SAC
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
#   PLOTTING FUNCTION
# ================================================================
def plot_episode_data(history, episode_num, save_dir):
    """
    Generates a single window with multiple subplots visualizing the episode.
    Saves to file and attempts to open it to avoid threading crashes.
    """
    # Convert lists to numpy arrays for easier slicing
    times = np.array(history['time'])
    pos = np.array(history['pos'])        # (N, 3)
    att = np.array(history['attitude'])   # (N, 3) -> Roll, Pitch, Yaw
    thrust = np.array(history['thrust'])  # (N,)
    gimbal = np.array(history['gimbal'])  # (N, 2) -> Yaw, Pitch
    
    # Calculate Deviations
    lateral_error = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    
    # Create Figure
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Rocket Landing Analysis - Episode {episode_num}", fontsize=16)

    # ---------------------------------------------------------
    # 1. 3D TRAJECTORY
    # ---------------------------------------------------------
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b', linewidth=2)
    ax3d.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='g', marker='o', label='Start')
    ax3d.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color='r', marker='x', label='End')
    ax3d.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target Center')
    
    ax3d.set_xlabel('X Position (m)')
    ax3d.set_ylabel('Y Position (m)')
    ax3d.set_zlabel('Altitude (m)')
    ax3d.legend()

    # ---------------------------------------------------------
    # 2. POSITION STATE (X, Y, Z)
    # ---------------------------------------------------------
    ax_pos = fig.add_subplot(2, 3, 2)
    ax_pos.set_title("Position vs Time")
    ax_pos.plot(times, pos[:, 0], label='X', linestyle='--')
    ax_pos.plot(times, pos[:, 1], label='Y', linestyle='--')
    ax_pos.plot(times, pos[:, 2], label='Altitude (Z)', linewidth=2)
    ax_pos.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax_pos.set_ylabel("Position (m)")
    ax_pos.grid(True)
    ax_pos.legend()

    # ---------------------------------------------------------
    # 3. ATTITUDE (Roll, Pitch)
    # ---------------------------------------------------------
    ax_att = fig.add_subplot(2, 3, 3)
    ax_att.set_title("Attitude (Stability)")
    ax_att.plot(times, att[:, 0], label='Roll', color='purple')
    ax_att.plot(times, att[:, 1], label='Pitch', color='orange')
    ax_att.set_ylabel("Angle (Degrees)")
    ax_att.set_xlabel("Time (s)")
    ax_att.grid(True)
    ax_att.legend()

    # ---------------------------------------------------------
    # 4. THRUST MAGNITUDE
    # ---------------------------------------------------------
    ax_thrust = fig.add_subplot(2, 3, 4)
    ax_thrust.set_title("Thrust Control")
    ax_thrust.plot(times, thrust, color='r')
    ax_thrust.set_ylabel("Thrust (N)")
    ax_thrust.set_xlabel("Time (s)")
    ax_thrust.grid(True)
    ax_thrust.fill_between(times, thrust, color='r', alpha=0.1)

    # ---------------------------------------------------------
    # 5. GIMBAL ANGLES (Control Inputs)
    # ---------------------------------------------------------
    ax_gimbal = fig.add_subplot(2, 3, 5)
    ax_gimbal.set_title("Gimbal Control Angles")
    ax_gimbal.plot(times, gimbal[:, 0], label='Gimbal Yaw')
    ax_gimbal.plot(times, gimbal[:, 1], label='Gimbal Pitch')
    ax_gimbal.set_ylabel("Angle (Degrees)")
    ax_gimbal.set_xlabel("Time (s)")
    ax_gimbal.grid(True)
    ax_gimbal.legend()

    # ---------------------------------------------------------
    # 6. DERIVATION (Distance from Center)
    # ---------------------------------------------------------
    ax_err = fig.add_subplot(2, 3, 6)
    ax_err.set_title("Lateral Deviation from Target")
    ax_err.plot(times, lateral_error, color='brown')
    ax_err.set_ylabel("Distance (m)")
    ax_err.set_xlabel("Time (s)")
    ax_err.grid(True)
    ax_err.fill_between(times, lateral_error, color='brown', alpha=0.1)

    plt.tight_layout()
    
    # --- SAVE AND OPEN ---
    filename = f"episode_{episode_num}_analysis.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"{Col.BOLD}📊 Plot saved to: {save_path}{Col.RESET}")
    
    # Auto-open the image (Platform specific)
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.call(["open", save_path])
        elif sys.platform == "win32": # Windows
            os.startfile(save_path)
        else: # Linux
            subprocess.call(["xdg-open", save_path])
    except Exception:
        pass # If we can't open it, the user can just find the file.

# ================================================================
#   MAIN TESTING LOGIC
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Test a trained SAC Rocket Agent")
    parser.add_argument("run_path", type=str, help="Path to the run folder (e.g., runs/sac_rocket_2025...)")
    parser.add_argument("--model", type=str, default="best", choices=["best", "final", "latest"], help="Which model to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    # 1. Resolve Paths
    run_dir = args.run_path
    if not os.path.exists(run_dir):
        print(f"{Col.RED}Error: Run directory not found: {run_dir}{Col.RESET}")
        return

    # Locate VecNormalize stats
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

    print(f"\n{Col.BOLD}🚀 LOADING AGENT{Col.RESET}")
    print(f"   Run Dir:   {run_dir}")
    print(f"   Model:     {model_path}")
    print(f"   Norm Stats:{norm_path}")

    # 2. Setup Environment
    env = DummyVecEnv([lambda: RocketLandingEnv(render_mode="human" if not args.no_render else None)])
    env = VecNormalize.load(norm_path, env)
    env.training = False 
    env.norm_reward = False

    # 3. Load Agent
    model = SAC.load(model_path)

    # 4. Evaluation Loop
    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        print("-" * 80)
        
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0

        # --- DATA RECORDING FOR PLOTS ---
        history = {
            'time': [],
            'pos': [],
            'attitude': [],
            'thrust': [],
            'gimbal': []
        }

        while not done:
            step += 1
            
            # Predict
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            # CRITICAL FIX: Stop recording if episode is done.
            # SB3 DummyVecEnv automatically resets the environment when done=True.
            # If we record 'real_env' data now, we get the INITIAL state of the NEXT episode (teleporting to sky).
            # We want the plot to end exactly when the rocket finishes/crashes.
            if done:
                break
            
            # Access Real Environment
            real_env = env.venv.envs[0] 
            
            # --- FETCH DATA ---
            pos = real_env.data.xpos[real_env.rocket_bid].copy()
            vel = real_env.data.cvel[real_env.rocket_bid][3:].copy()
            quat = real_env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            
            thrust_N = real_env.data.ctrl[real_env.thrust_act]
            g_yaw = np.degrees(real_env.data.ctrl[real_env.yaw_act])
            g_pit = np.degrees(real_env.data.ctrl[real_env.pitch_act])
            
            # --- STORE DATA ---
            history['time'].append(step * real_env.DT) # Approx time
            history['pos'].append(pos)
            history['attitude'].append([roll, pitch, yaw])
            history['thrust'].append(thrust_N)
            history['gimbal'].append([g_yaw, g_pit])

            # --- DASHBOARD LOGGING ---
            if not args.no_render:
                real_env.render()
                
                # Format Dashboard
                state_str = f"Alt:{pos[2]:5.1f}m Vz:{vel[2]:5.1f} Tlt:{max(abs(pitch),abs(roll)):4.1f}°"
                ctrl_str  = f"Thr:{thrust_N:6.0f}N Gmb:{g_yaw:3.0f}/{g_pit:3.0f}"
                
                log_line = (
                    f"\r{step:04} | "
                    f"{Col.CYAN}{state_str}{Col.RESET} | "
                    f"{Col.YELLOW}{ctrl_str}{Col.RESET} | "
                    f"{Col.GREEN}Rew:{reward[0]:6.2f}{Col.RESET} \033[K"
                )
                sys.stdout.write(log_line)
                sys.stdout.flush()
                time.sleep(0.01) # Slightly faster for visuals

        # Episode End
        info_dict = info[0]
        result_msg = "✅ SUCCESS" if info_dict.get('success') else "❌ FAILURE"
        color = Col.GREEN if info_dict.get('success') else Col.RED
        print(f"\n{color}>>> {result_msg} | Total Reward: {total_reward:.2f}{Col.RESET}")

        # --- PLOT RESULTS ---
        print(f"{Col.BOLD}📊 Generating plots for Episode {ep+1}...{Col.RESET}")
        plot_episode_data(history, ep+1, run_dir)

    env.close()
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    main()