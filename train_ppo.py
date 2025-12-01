import os
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from rocket_env.rocket_landing_env import RocketLandingEnv

# ======================================================================
#   CALLBACK: VISUALIZE PERIODICALLY
# ======================================================================
class VisualizeCallback(BaseCallback):
    """
    Custom callback to pop up a render window every N steps
    to see how the agent is doing visually.
    """
    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.test_env = None

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"\n👀 Visualizing policy at step {self.num_timesteps}...")
            
            # Lazy load the test env to avoid conflicts at startup
            if self.test_env is None:
                self.test_env = RocketLandingEnv(render_mode="human")
            
            # We need to grab the normalization statistics from the training env
            # to ensure the agent sees the world correctly.
            # However, for a quick visual check, unnormalized usually works okay.
            obs, _ = self.test_env.reset()
            
            for _ in range(1000): # Run for one episode
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.test_env.step(action)
                self.test_env.render()
                time.sleep(0.01)
                
                if terminated or truncated:
                    break
            
        return True

# ======================================================================
#   ENV FACTORY
# ======================================================================
def make_env():
    """
    Utility to create the environment wrapped in a Monitor
    (Monitor helps SB3 track success rates and episode lengths).
    """
    env = RocketLandingEnv(render_mode=None)
    env = Monitor(env)
    return env

# ======================================================================
#   MAIN TRAINING
# ======================================================================
def train_ppo():
    # 1. Setup Directories
    run_name = "ppo_rocket_" + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Create Vectorized Environment
    # VecNormalize is CRITICAL for PPO stability on physics tasks
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Create Evaluation Environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    # Sync stats
    eval_env.training = False 
    eval_env.norm_reward = False

    # 4. Define Callbacks
    
    # A. Save the best model based on evaluation reward
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=20_000, # Check every 20k steps
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # B. Save checkpoints every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_rocket"
    )

    # C. Visualize locally (every 100k steps)
    visual_callback = VisualizeCallback(check_freq=100_000)

    # Combine callbacks
    callback_list = CallbackList([eval_callback, checkpoint_callback, visual_callback])

    # 5. Define Model
    # We increase the network size to [256, 256] to match the SAC complexity.
    # Default PPO is often too small [64, 64] for 3D physics.
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,       # Number of steps to run for each environment per update
        batch_size=64,      # Minibatch size
        n_epochs=10,        # Number of epochs when optimizing the surrogate loss
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # Slight entropy to prevent premature convergence
        tensorboard_log=os.path.join(log_dir, "tb_logs"),
    )

    # 6. Train
    total_timesteps = 2_000_000
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
    
    # Save the normalization statistics (obs mean/std). 
    # CRITICAL: You need these to run the model later!
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_ppo()