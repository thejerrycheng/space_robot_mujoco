import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
from collections import deque
import importlib 

# RL Libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Import your environment
from rocket_env.rocket_2_env import RocketLandingEnv

# ==============================================================================
#   ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training for Rocket Landing")
    
    # --- Experiment Config ---
    # NEW: Added --name argument for custom folder naming
    parser.add_argument("--name", type=str, default=None, 
                        help="Custom tag for the run (e.g., 'test1' -> ppo_rocket2_test1_date_time)")
    
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timesteps to train")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--viz_freq", type=int, default=20_000, help="Visualize every N episodes")
    
    # --- REWARD CONFIG ---
    parser.add_argument("--reward", type=str, default="flip_and_fuel", 
                        help="Name of the reward file in rocket_env/rewards/ (without .py)")

    # --- PPO Hyperparameters ---
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    
    return parser.parse_args()

# ==============================================================================
#   DYNAMIC REWARD LOADER
# ==============================================================================
def load_reward_function(reward_name):
    """
    Imports rocket_env.rewards.<reward_name> and returns the compute_reward function.
    """
    try:
        module_path = f"rocket_env.rewards.{reward_name}"
        mod = importlib.import_module(module_path)
        return mod.compute_reward
    except ImportError as e:
        print(f"\n❌ Error loading reward: {reward_name}")
        print(f"Make sure '{module_path}.py' exists.")
        raise e

# ==============================================================================
#   CALLBACKS
# ==============================================================================
class RocketCallback(BaseCallback):
    """
    Combines Logging (Success Rate & Semi-Success) AND Visualization.
    """
    def __init__(self, viz_freq_episodes=20000, verbose=0):
        super().__init__(verbose)
        self.success_buffer = deque(maxlen=100)
        self.semi_success_buffer = deque(maxlen=100)
        
        self.viz_freq = viz_freq_episodes
        self.episode_count = 0
        self.last_viz_count = 0

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        num_dones = np.sum(dones)
        if num_dones > 0:
            self.episode_count += num_dones
            
            for idx, done in enumerate(dones):
                if done:
                    # Log Success (Full landing)
                    if "success" in infos[idx]:
                        self.success_buffer.append(float(infos[idx]["success"]))
                    
                    # Log Semi-Success (In target area but bad orientation)
                    if "semi_success" in infos[idx]:
                        self.semi_success_buffer.append(float(infos[idx]["semi_success"]))

        # Visualization Trigger
        if self.viz_freq > 0 and (self.episode_count - self.last_viz_count >= self.viz_freq):
            self.last_viz_count = self.episode_count
            self.trigger_visualization()

        return True

    def _on_rollout_end(self) -> None:
        if len(self.success_buffer) > 0:
            success_rate = np.mean(self.success_buffer) * 100
            self.logger.record("rollout/success_rate", success_rate)
            
        if len(self.semi_success_buffer) > 0:
            semi_success_rate = np.mean(self.semi_success_buffer) * 100
            self.logger.record("rollout/semi_success_rate", semi_success_rate)

    def trigger_visualization(self):
        print(f"\n🎥 VISUALIZING EPISODE (Count: {self.episode_count})...")
        
        # Create a temp env for visualization
        # Note: We rely on default reward here for visualization, or you can pass args if needed
        viz_env = RocketLandingEnv(render_mode="human")
        obs_rms = self.training_env.obs_rms
        
        obs, _ = viz_env.reset()
        done = False
        
        while not done:
            # Normalize observation using training stats
            norm_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
            action, _ = self.model.predict(norm_obs, deterministic=True)
            obs, _, terminated, truncated, _ = viz_env.step(action)
            viz_env.render()
            done = terminated or truncated
            
        viz_env.close()
        print("🎥 Visualization complete. Resuming training...\n")

# ==============================================================================
#   PLOTTING
# ==============================================================================
def plot_training_results(log_dir):
    monitor_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_path): return

    try:
        df = pd.read_csv(monitor_path, skiprows=1)
        window_size = 100
        df['rolling_reward'] = df['r'].rolling(window=window_size).mean()
        
        success_col = None
        semi_success_col = None
        
        # Detect relevant columns dynamically
        for col in df.columns:
            if 'success' in col and 'semi' not in col: 
                success_col = col
            if 'semi_success' in col: 
                semi_success_col = col
        
        if success_col:
            df['rolling_success'] = df[success_col].rolling(window=window_size).mean() * 100
        
        if semi_success_col:
            df['rolling_semi_success'] = df[semi_success_col].rolling(window=window_size).mean() * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.set_title("Training Metrics")
        ax1.scatter(df.index, df['r'], alpha=0.1, color='gray', s=1)
        ax1.plot(df.index, df['rolling_reward'], color='blue', linewidth=2, label='Reward')
        ax1.set_ylabel("Reward"); ax1.legend(); ax1.grid(True, alpha=0.3)

        if success_col:
            ax2.plot(df.index, df['rolling_success'], color='green', linewidth=2, label='Success %')
        
        if semi_success_col:
            ax2.plot(df.index, df['rolling_semi_success'], color='orange', linewidth=2, linestyle='--', label='Semi-Success %')

        if success_col or semi_success_col:
            ax2.set_ylabel("Rate (%)"); ax2.set_ylim(-5, 105)
            ax2.axhline(y=100, color='r', linestyle=':', alpha=0.3)
            ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.xlabel("Episode")
        plt.savefig(os.path.join(log_dir, "training_metrics.png"))
        plt.close()
    except Exception: pass

# ==============================================================================
#   MAIN
# ==============================================================================
def main():
    args = parse_args()
    
    # --- UPDATED NAMING LOGIC ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.name:
        # If user provides --name "test", result is "ppo_rocket2_test_20231024_120000"
        run_name = f"ppo_rocket2_{args.name}_{timestamp}"
    else:
        # Fallback default
        run_name = f"ppo_rocket2_{args.reward}_{timestamp}"
        
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting Run: {run_name}")
    print(f"💰 Reward Function: {args.reward}")
    print(f"📂 Saving to: {log_dir}")

    # 1. Load Reward Function
    reward_func = load_reward_function(args.reward)

    # 2. Init Env (Inject Reward)
    env = make_vec_env(
        RocketLandingEnv, 
        n_envs=args.num_envs, 
        seed=args.seed,
        monitor_dir=log_dir,
        # Updated monitor keywords to capture semi_success from env info
        monitor_kwargs={'info_keywords': ('success', 'semi_success')}, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"reward_func": reward_func} 
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Init Agent
    model = PPO(
        "MlpPolicy", env,
        learning_rate=args.learning_rate, n_steps=args.n_steps,
        batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1, tensorboard_log=log_dir, device=args.device, seed=args.seed
    )

    # 4. Callbacks
    ckpt_cb = CheckpointCallback(save_freq=max(50_000 // args.num_envs, 1), save_path=os.path.join(log_dir, "checkpoints"), name_prefix="ppo_rocket")
    rocket_cb = RocketCallback(viz_freq_episodes=args.viz_freq)

    # 5. Train
    print("💪 Training started...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=[ckpt_cb, rocket_cb], progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")

    # 6. Save & Close
    model.save(os.path.join(log_dir, "final_model"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    plot_training_results(log_dir)
    env.close()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()