import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# CRITICAL: Use Agg backend to save plots without a window (avoids macOS crashes)
matplotlib.use('Agg')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Correct import path
from rocket_env.rocket_landing_env_simple import RocketLandingEnv

# ======================================================================
#   CALLBACK: TRACK & PRINT SUCCESS RATE PER ROLLOUT
# ======================================================================
class SuccessTrackingCallback(BaseCallback):
    """
    Tracks the success rate of episodes during training rollouts and 
    prints it clearly at the end of each rollout cycle.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_successes = []

    def _on_step(self) -> bool:
        # Check if any episode ended in the vectorized environments
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                # Look for 'success' in info dict (set by environment)
                success = info.get("success", False)
                self.episode_successes.append(success)
        return True

    def _on_rollout_end(self) -> None:
        """
        Called before the policy update. 
        We calculate stats for the collected rollout buffer.
        """
        if len(self.episode_successes) > 0:
            success_rate = np.mean(self.episode_successes)
            count = len(self.episode_successes)
            
            # Print to console with color
            # Green if > 80%, Yellow if > 50%, Red otherwise
            if success_rate > 0.8:
                color = "\033[92m" 
            elif success_rate > 0.5:
                color = "\033[93m"
            else:
                color = "\033[91m"
            reset = "\033[0m"
            
            print(f"   📊 Rollout Success Rate: {color}{success_rate * 100:.2f}%{reset} ({sum(self.episode_successes)}/{count} eps)")
            
            # Log to Tensorboard
            self.logger.record("rollout/success_rate_custom", success_rate)
            
            # Reset buffer for next rollout
            self.episode_successes = []

# ======================================================================
#   CALLBACK: VISUALIZE PERIODICALLY
# ======================================================================
class VisualizeCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.test_env = None

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"\n👀 Visualizing policy at step {self.num_timesteps}...")
            
            if self.test_env is None:
                self.test_env = RocketLandingEnv(render_mode="human")
            
            # Reset
            obs, _ = self.test_env.reset()
            total_reward = 0.0
            success = False

            for _ in range(1000): 
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.test_env.step(action)
                self.test_env.render()
                time.sleep(0.01)
                
                total_reward += reward
                
                if terminated or truncated:
                    success = info.get("success", False)
                    break
            
            if success:
                print(f"   Visual Test: \033[92m✅ SUCCESS\033[0m | Total Reward: {total_reward:.2f}\n")
            else:
                print(f"   Visual Test: \033[91m❌ FAILURE\033[0m | Total Reward: {total_reward:.2f}\n")
            
        return True

# ======================================================================
#   PLOTTING UTILS
# ======================================================================
def plot_training_results(log_dir, title="Rocket Landing Training Reward"):
    """ Reads monitor.csv and plots the reward curve. """
    try:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(x, y, alpha=0.3, label='Raw Reward')
            
            # Moving Average
            window_size = 50
            if len(y) > window_size:
                y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
                x_smooth = x[window_size-1:]
                plt.plot(x_smooth, y_smooth, color='red', linewidth=2, label='Moving Avg')
                
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            
            save_path = os.path.join(log_dir, "training_reward_plot.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"\n📈 Training plot saved to: {save_path}")
    except Exception as e:
        print(f"⚠️ Could not generate training plot: {e}")

# ======================================================================
#   ENV FACTORY
# ======================================================================
def make_env(log_dir=None):
    """ Create env wrapped in Monitor for stats tracking """
    def _init():
        env = RocketLandingEnv(render_mode=None)
        if log_dir is not None:
            env = Monitor(env, log_dir) # Saves to monitor.csv
        else:
            env = Monitor(env)
        return env
    return _init

# ======================================================================
#   MAIN TRAINING
# ======================================================================
def train_ppo():
    # 1. Setup Directories
    run_name = "ppo_rocket_" + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Create Vectorized Environment
    env = DummyVecEnv([make_env(log_dir)])
    # Normalization is critical for physics
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Create Evaluation Environment
    eval_env = DummyVecEnv([make_env(None)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    eval_env.training = False 
    eval_env.norm_reward = False

    # 4. Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=20_000, 
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_rocket"
    )

    visual_callback = VisualizeCallback(check_freq=100_000)
    
    # NEW: Add success tracker
    success_callback = SuccessTrackingCallback()

    callback_list = CallbackList([eval_callback, checkpoint_callback, visual_callback, success_callback])

    # 5. Define Model
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,       
        batch_size=64,     
        n_epochs=10,        
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      
        tensorboard_log=os.path.join(log_dir, "tb_logs"),
    )

    # 6. Train
    total_timesteps = 10_000_000
    print(f"\n🚀 Starting PPO Training: {run_name}")
    print(f"📂 Logging to: {log_dir}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callback_list,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted manually.")

    # 7. Save Final Model & Normalization Stats
    print("\n✅ Saving final model...")
    model.save(os.path.join(log_dir, "final_model"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    
    env.close()
    eval_env.close()

    # 8. Generate Plot
    plot_training_results(log_dir)

if __name__ == "__main__":
    train_ppo()