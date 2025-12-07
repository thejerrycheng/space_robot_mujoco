import os
import sys
import time
import datetime
import argparse
import importlib
import gc 
import numpy as np
import gymnasium as gym
import glob

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# IMPORT YOUR ENV CLASS
# Assumes rocket_env/rocket_gym_env.py contains the RocketLandingEnv
from rocket_env.rocket_gym_env import RocketLandingEnv 

# ================================================================
#   HELPER: DYNAMIC REWARD LOADING
# ================================================================
def load_reward_class(reward_name):
    """
    Dynamically loads the RocketReward class from the specified reward file.
    Assumes the file is located at rocket_env/rewards/{reward_name}.py.
    """
    try:
        # Construct the module path for importlib
        module_path = f"rocket_env.rewards.{reward_name}"
        mod = importlib.import_module(module_path)
        
        # Check for the expected class name
        if hasattr(mod, "RocketReward"):
            return mod.RocketReward
        else:
            # Fallback check for any other class ending with 'Reward'
            for attr_name in dir(mod):
                if "Reward" in attr_name and attr_name != "RocketReward":
                    return getattr(mod, attr_name)
            raise AttributeError(f"Could not find 'RocketReward' class in {module_path}")
    except ImportError as e:
        print(f"❌ Error loading reward: {reward_name}. Check the path and file name.")
        raise e

# ================================================================
#   CALLBACK: METRICS LOGGING (Tensorboard)
# ================================================================
class RocketMetricsCallback(BaseCallback):
    """
    Custom callback to log specific metrics to TensorBoard at the end of a rollout.
    It collects data (success, fuel, distance) from environment info dictionaries.
    """
    def __init__(self, verbose=0):
        super(RocketMetricsCallback, self).__init__(verbose)
        self.success_history = []
        self.curriculum_history = []
        self.fuel_history = []
        self.distance_history = []

    def _on_step(self) -> bool:
        # Collect info only when an episode terminates or truncates (done)
        dones = self.locals['dones']
        infos = self.locals['infos']
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                # Note: Monitor wraps info in a list for vectorized envs
                # We extract the 'real' info dictionary
                info_dict = info if isinstance(info, dict) else info.get('episode', {})
                
                self.success_history.append(1.0 if info_dict.get('is_success', False) else 0.0)
                # Curriculum level may not exist, default to 0
                self.curriculum_history.append(info_dict.get('curriculum_level', 0))
                self.fuel_history.append(info_dict.get('fuel_remaining', 0))
                
                # 'lateral_dist' must be in the final state dict
                if 'lateral_dist' in info_dict:
                     self.distance_history.append(info_dict['lateral_dist'])
        return True

    def _on_rollout_end(self) -> None:
        """Log averages to TensorBoard."""
        if len(self.success_history) > 0:
            self.logger.record("custom/success_rate", np.mean(self.success_history))
            self.success_history = [] 
        if len(self.curriculum_history) > 0:
            self.logger.record("custom/curriculum_level", np.mean(self.curriculum_history))
            self.curriculum_history = []
        if len(self.fuel_history) > 0:
            self.logger.record("custom/avg_fuel_remaining", np.mean(self.fuel_history))
            self.fuel_history = []
        if len(self.distance_history) > 0:
            self.logger.record("custom/avg_landing_dist", np.mean(self.distance_history))
            self.distance_history = []

# ================================================================
#   MAIN TRAINING LOOP
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Train SAC Agent for Rocket Landing")
    
    # --- Resume Training ---
    parser.add_argument("--model", type=str, default=None, help="Path to a previous run folder (e.g., models/gym_sac_X_date).")

    # --- Experiment Identity ---
    parser.add_argument("--name", type=str, default="default", help="Name of the experiment")
    parser.add_argument("--reward", type=str, default="landing_reward", help="Reward file name (e.g., 'landing_reward' for rocket_env/rewards/landing_reward.py)")
    
    # --- Hyperparameters (Standard SAC) ---
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ent_coef", type=str, default="auto", help="Entropy coefficient. Use 'auto' or a fixed float.")
    
    # --- System & Logging ---
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments (SubprocVecEnv)")
    parser.add_argument("--checkpoint_freq", type=int, default=100_000, help="Save model every N steps (unvectorized)")
    parser.add_argument("--eval_freq", type=int, default=50_000, help="Evaluate model every N steps (unvectorized)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # 1. SETUP PATHS (Follows the requested naming convention)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"hover_gym_sac_{args.name}_{timestamp}" # Modified to match the requested prefix
    base_path = "./models/"
    run_dir = os.path.join(base_path, run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"📁 Saving Training Data to: {run_dir}")

    # Initialize env vars to None so finally block handles them safely
    train_env = None
    eval_env = None

    try:
        # 2. LOAD REWARD CLASS
        RewardClass = load_reward_class(args.reward)

        # 3. DEFINE ENV MAKER
        # A factory function to create environment instances
        def make_env_instance(render_mode=None):
            # The RocketLandingEnv class imports its own reward script internally.
            # We initialize the environment as is.
            env = RocketLandingEnv(render_mode=render_mode)
            
            # Note: The provided env already initializes RocketReward
            # If the loaded RewardClass is different, you could potentially overwrite it here:
            # if env.rewarder.__class__.__name__ != RewardClass.__name__:
            #     env.rewarder = RewardClass({'start_fuel': env.START_FUEL})
                
            # Wrap the environment with Monitor for logging
            env = Monitor(env) 
            return env

        def make_train_env():
            return make_env_instance(render_mode=None)

        # 4. CREATE ENVS
        # SubprocVecEnv for parallel training (faster data collection)
        train_env = SubprocVecEnv([make_train_env for _ in range(args.num_envs)])
        # DummyVecEnv for evaluation (simpler, one environment at a time)
        eval_env = DummyVecEnv([make_train_env]) 

        # 5. DEFINE CALLBACKS
        metric_cb = RocketMetricsCallback()
        
        # Checkpoint: Save model at regular intervals
        checkpoint_cb = CheckpointCallback(
            # Save frequency is divided by num_envs to get the step interval per-env
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=ckpt_dir,
            name_prefix="sac_rocket"
        )
        
        # EvalCallback: Evaluate and save the best model
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model"),
            log_path=log_dir,
            eval_freq=max(args.eval_freq // args.num_envs, 1),
            deterministic=True,
            render=False
        )
        
        # Combine callbacks
        callback_list = CallbackList([checkpoint_cb, eval_cb])

        # 6. INITIALIZE OR RESUME MODEL
        if args.model:
            potential_paths = [
                os.path.join(args.model, "best_model", "best_model.zip"),
                os.path.join(args.model, "best_model.zip"),
                os.path.join(args.model, "final_model.zip")
            ]
            
            load_path = None
            # Find the first existing model file to load
            for p in potential_paths:
                if os.path.exists(p):
                    load_path = p
                    break
            
            if load_path:
                print(f"🔄 RESUMING TRAINING from: {load_path}")
                # Load the model, overriding certain hyperparameters
                model = SAC.load(
                    load_path,
                    env=train_env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    learning_rate=args.learning_rate,
                    ent_coef=args.ent_coef,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    tau=args.tau,
                )
            else:
                print(f"❌ Could not find a model file in {args.model}. Check path.")
                return
        else:
            print(f"🆕 STARTING NEW TRAINING")
            # Initialize a new SAC model
            model = SAC(
                "MlpPolicy", # Standard multilayer perceptron policy
                train_env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                ent_coef=args.ent_coef,
                gamma=args.gamma,
                tau=args.tau,
                seed=args.seed,
                policy_kwargs=dict(net_arch=[256, 256]), # Example policy network architecture
            )

        print("------------------------------------------")
        print(f"🚀 TRAINING START | Run ID: {run_id}")
        print("------------------------------------------")

        # 7. TRAIN
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=callback_list,
            progress_bar=True 
        )

        # 8. SAVE FINAL
        final_path = os.path.join(run_dir, "final_model")
        model.save(final_path)
        
        print("------------------------------------------")
        print("🏁 TRAINING COMPLETE")
        print("------------------------------------------")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        # Re-raise the exception after logging
        raise e
    finally:
        # 9. SAFE SHUTDOWN
        print("🛑 Shutting down environments...")
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect()

if __name__ == "__main__":
    main()