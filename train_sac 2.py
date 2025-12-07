import os
import time
import numpy as np

from stable_baselines3 import SAC
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
            
            # Use the model to predict
            # Note: We must normalize obs manually if using VecNormalize in training
            obs, _ = self.test_env.reset()
            
            # If the training env is normalized, we should ideally use the 
            # training env's running average, but for a quick visual check
            # unnormalized prediction is often 'good enough' to see behavior,
            # or we can rely on the robustness of the policy.
            
            for _ in range(800): # Run for one episode
                # Determine deterministic action
                action, _ = self.model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, _ = self.test_env.step(action)
                self.test_env.render()
                time.sleep(0.01)
                
                if terminated or truncated:
                    break
            
            # Don't close the env, keep it for next time to avoid window respawning issues
            # self.test_env.close() 
            
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
def train_sac():
    # 1. Setup Directories
    run_name = "sac_rocket_" + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Create Vectorized Environment
    # VecNormalize is CRITICAL for MuJoCo tasks to handle different input scales
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Create Evaluation Environment
    # We need a separate env for evaluation so we don't mess up training stats
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    # Important: sync the stats from training env to eval env
    eval_env.training = False 
    eval_env.norm_reward = False

    # 4. Define Callbacks
    
    # A. Save the best model based on reward
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # B. Save checkpoints every 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_rocket"
    )

    # C. Visualize locally (every 50k steps)
    visual_callback = VisualizeCallback(check_freq=50_000)

    # Combine callbacks
    callback_list = CallbackList([eval_callback, checkpoint_callback, visual_callback])

    # 5. Define Model (Optimized Hyperparameters)
    # Larger network [256, 256] is better for physics control
    policy_kwargs = dict(net_arch=[256, 256])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=512,          # Increased for smoother gradients
        tau=0.005,
        gamma=0.99,
        train_freq=1,            # Update every step (standard SAC)
        gradient_steps=1,        # 1 step per update
        learning_starts=10_000,  # Warmup steps (random actions)
        ent_coef="auto",
        tensorboard_log=os.path.join(log_dir, "tb_logs"),
    )

    # 6. Train
    total_timesteps = 3_000_000
    print(f"\n🚀 Starting SAC Training: {run_name}")
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
    train_sac()