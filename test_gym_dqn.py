import os
import sys
import time
import argparse
import numpy as np
import subprocess
import csv
import matplotlib # Keep for backend setting
import importlib
import mujoco 
import glob
import gymnasium as gym

# CRITICAL FIX: Use 'Agg' backend to prevent macOS/Linux main-thread rendering crashes
matplotlib.use('Agg') 

# --- IMPORT DQN MODEL ---
from stable_baselines3 import DQN 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- IMPORT YOUR ENV CLASS ---
# Assuming the environment class is in the same relative location as the training script
from rocket_env.rocket_gym_discrete import RocketLandingEnv 

# ⭐️ IMPORT PLOTTING UTILITIES DIRECTLY ⭐️
from utils.plotting import (
    plot_static_analysis, 
    generate_interactive_plot, 
    get_body_z_axis,
    Col # Import the Col class for colored output
)

# Placeholder for the reward wrapper used during training
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        return self.env.step(action) 
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
# --- REWARD LOADER (Same as training script) ---
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
#   UTILITIES: MATH
# ================================================================

def quat_to_euler(quat):
    """ Convert [w, x, y, z] to [roll, pitch, yaw] in degrees. """
    # Uses scipy's Rotation internally (assumed available)
    from scipy.spatial.transform import Rotation as R
    w, x, y, z = quat
    r = R.from_quat([x, y, z, w])
    euler = r.as_euler('xyz', degrees=True)
    return euler

def save_to_csv(history, episode_num, save_dir):
    """ Saves episode trajectory data to a CSV file. """
    filename = os.path.join(save_dir, f"episode_{episode_num}.csv")
    
    times = history['time']; pos = np.array(history['pos']); vel = np.array(history['vel']); att = np.array(history['attitude'])
    thrust = np.array(history['thrust']); gimbal = np.array(history['gimbal']); mass = np.array(history['mass']); rewards = np.array(history['reward'])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Time", "X", "Y", "Z", "Vx", "Vy", "Vz", "Roll", "Pitch", "Yaw", "Thrust", "GimbalYaw", "GimbalPitch", "Mass", "Reward"])
        for i in range(len(times)):
            writer.writerow([i, times[i], pos[i,0], pos[i,1], pos[i,2], vel[i,0], vel[i,1], vel[i,2], att[i,0], att[i,1], att[i,2], thrust[i], gimbal[i,0], gimbal[i,1], mass[i], rewards[i]])
    print(f"💾 Data saved to: {filename}")


# ================================================================
#   MAIN TESTING LOGIC
# ================================================================
def normalize_obs(obs, obs_rms, epsilon=1e-8):
    """ Normalize observation using loaded statistics (If VecNormalize was used) """
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon), -10, 10)

def main():
    parser = argparse.ArgumentParser(description="Test a trained DQN Rocket Agent")
    parser.add_argument("run_path", type=str, help="Path to the run folder (e.g., models/dqn_gym_NAME_timestamp)")
    parser.add_argument("--model", type=str, default="final", choices=["best", "final", "latest"], help="Which model to load (final, best, latest checkpoint)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes to visualize")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--reward", type=str, default="landing_reward", help="Name of reward file in rocket_env/rewards/")
    
    args = parser.parse_args()

    # 1. Setup Directories
    run_dir_base = args.run_path
    if not os.path.isdir(run_dir_base):
        print(f"{Col.RED}Error: Run directory '{run_dir_base}' not found. Please provide the full path to the timestamped folder (e.g., models/dqn_gym_default_20251205_123456).{Col.RESET}"); return

    data_dir = os.path.join(run_dir_base, "test_results_dqn")
    os.makedirs(data_dir, exist_ok=True)
    print(f"{Col.BOLD}📂 Saving test data to: {data_dir}{Col.RESET}")

    # 2. Load Reward Class
    try:
        RewardClass = load_reward_class(args.reward)
        rewarder = RewardClass({'start_fuel': 4_500_000.0}) 
        print(f"{Col.CYAN}💰 Loaded Reward Class: {args.reward}{Col.RESET}")
    except Exception as e:
        print(f"{Col.RED}Error initializing rewarder: {e}{Col.RESET}"); return

    # 3. Normalization Stats (Skipped unless VecNormalize was used)
    obs_rms = None
    
    # 4. Locate & Load DQN Model (Aligned with training script paths)
    if args.model == "final":
        model_path = os.path.join(run_dir_base, "final_model.zip")
    elif args.model == "best":
        model_path = os.path.join(run_dir_base, "best_model", "best_model.zip")
    else: # latest checkpoint
        ckpt_dir = os.path.join(run_dir_base, "checkpoints")
        ckpts = glob.glob(os.path.join(ckpt_dir, "dqn_rocket_model_*.zip"))
        if ckpts:
            ckpts.sort(key=os.path.getmtime)
            model_path = ckpts[-1]
        else:
            print(f"{Col.RED}Error: No checkpoints found in {ckpt_dir}.{Col.RESET}"); return

    if not os.path.exists(model_path):
        print(f"{Col.RED}Error: Model file not found at {model_path}.{Col.RESET}")
        return
    
    print(f"{Col.BOLD}🚀 Loading DQN Model from: {model_path}{Col.RESET}")
    model = DQN.load(model_path, env=None) 

    # 5. Create RAW Environment for Testing
    env_base = RocketLandingEnv(render_mode="human" if not args.no_render else None)
    real_env = CustomRewardWrapper(env_base)
    
    # Assign the loaded rewarder instance and cache MuJoCo IDs
    if hasattr(real_env.env, "rewarder"): real_env.env.rewarder = rewarder
    
    real_env.rocket_bid = mujoco.mj_name2id(real_env.env.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    real_env.qpos_adr = 0
    real_env.thrust_act = mujoco.mj_name2id(real_env.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")
    real_env.yaw_act = mujoco.mj_name2id(real_env.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
    real_env.pitch_act = mujoco.mj_name2id(real_env.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
    real_env.DT = real_env.env.model.opt.timestep
    
    action_map = real_env.env.action_map
    
    all_histories = []

    for ep in range(args.episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{args.episodes}{Col.RESET}")
        
        obs, _ = real_env.reset()
        try: obs = real_env.env._get_obs()
        except AttributeError: pass

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
            
            # 7. Predict Action (Returns a single DISCRETE INTEGER)
            action_index, _ = model.predict(norm_obs, deterministic=True)
            
            # 8. Step Environment (Passes the integer action)
            obs, reward, terminated, truncated, info = real_env.step(action_index)
            total_reward += reward
            
            # 9. Log State and Actions
            pos = real_env.env.data.xpos[real_env.rocket_bid].copy()
            vel = real_env.env.data.qvel[0:3].copy() 
            quat = real_env.env.data.qpos[real_env.qpos_adr+3 : real_env.qpos_adr+7].copy()
            roll, pitch, yaw = quat_to_euler(quat)
            mass = real_env.env.DRY_MASS + real_env.env.fuel_mass
            
            # --- DENORMALIZE DISCRETE ACTION FOR LOGGING ---
            action_values = action_map[action_index.item()]
            
            thrust_N = action_values['thrust']
            g_yaw_norm = action_values['yaw_ctrl']
            g_pit_norm = action_values['pitch_ctrl']
            
            history['time'].append(step * real_env.DT)
            history['pos'].append(pos)
            history['vel'].append(vel)
            history['attitude'].append([roll, pitch, yaw])
            history['quat'].append(quat) 
            history['thrust'].append(thrust_N)
            history['gimbal'].append([g_yaw_norm, g_pit_norm])
            history['mass'].append(mass)
            history['reward'].append(reward)

            if not args.no_render:
                real_env.render()
                dist_xy = np.sqrt(pos[0]**2 + pos[1]**2)
                state_str = f"Alt:{pos[2]:5.1f} Dis:{dist_xy:5.1f}m Vz:{vel[2]:5.1f} Tlt:{max(abs(pitch),abs(roll)):4.1f}°"
                ctrl_str  = f"Thr:{thrust_N:6.0f}N GimY:{g_yaw_norm:4.2f}"
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

        # ⭐️ Use imported plotting functions ⭐️
        save_to_csv(history, ep+1, data_dir)
        plot_static_analysis(history, ep+1, data_dir)

    real_env.close()

    print(f"\n{Col.BOLD}📊 Generating Interactive 3D Plot...{Col.RESET}")
    # ⭐️ Use imported interactive plotting function ⭐️
    generate_interactive_plot(all_histories, data_dir, env_name="DQN Rocket Landing")
    
    print("\n👋 Testing complete.")

if __name__ == "__main__":
    # Example usage: 
    # mjpython test_dqn_agent.py models/dqn_gym_mytest_20251205_123456 --episodes 5
    main()