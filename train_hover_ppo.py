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

# --- CHANGED: Import PPO instead of SAC ---
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# IMPORT YOUR ENV CLASS
from rocket_env.rocket_gym_env import RocketLandingEnv 

# ================================================================
#   HELPER: DYNAMIC REWARD LOADING (Remains the same)
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
        print(f"❌ Error loading reward: {reward_name}. Check the path and file name.")
        raise e

# ================================================================
#   CALLBACK: METRICS LOGGING (Remains the same)
# ================================================================
class RocketMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RocketMetricsCallback, self).__init__(verbose)
        self.success_history = []
        self.curriculum_history = []
        self.fuel_history = []
        self.distance_history = []

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        infos = self.locals['infos']
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                info_dict = info if isinstance(info, dict) else info.get('episode', {})
                
                self.success_history.append(1.0 if info_dict.get('is_success', False) else 0.0)
                self.curriculum_history.append(info_dict.get('curriculum_level', 0))
                self.fuel_history.append(info_dict.get('fuel_remaining', 0))
                
                if 'lateral_dist' in info_dict:
                     self.distance_history.append(info_dict['lateral_dist'])
        return True

    def _on_rollout_end(self) -> None:
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
    parser = argparse.ArgumentParser(description="Train PPO Agent for Rocket Landing")
    
    # --- Resume Training ---
    parser.add_argument("--model", type=str, default=None, help="Path to a previous run folder (e.g., models/gym_ppo_X_date).")

    # --- Experiment Identity ---
    parser.add_argument("--name", type=str, default="default", help="Name of the experiment")
    parser.add_argument("--reward", type=str, default="landing_reward", help="Reward file name")
    
    # --- Hyperparameters (Modified for PPO) ---
    parser.add_argument("--total_timesteps", type=int, default=10_000_000) # Often requires more steps than SAC
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99) # Higher gamma is common for PPO
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for gradient updates.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to update the policy.")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter for PPO.")

    # --- System & Logging ---
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--checkpoint_freq", type=int, default=200_000)
    parser.add_argument("--eval_freq", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 1. SETUP PATHS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # --- CHANGED: Use ppo prefix ---
    run_id = f"hover_gym_ppo_{args.name}_{timestamp}" 
    base_path = "./models/"
    run_dir = os.path.join(base_path, run_id)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"📁 Saving Training Data to: {run_dir}")

    train_env = None
    eval_env = None

    try:
        # 2. LOAD REWARD CLASS
        RewardClass = load_reward_class(args.reward)

        # 3. DEFINE ENV MAKER
        def make_env_instance(render_mode=None):
            env = RocketLandingEnv(render_mode=render_mode)
            # Reward class is initialized inside RocketLandingEnv, 
            # ensure it's the right one (or trust the env's implementation)
            env = Monitor(env) 
            return env

        def make_train_env():
            return make_env_instance(render_mode=None)

        # 4. CREATE ENVS
        # PPO benefits greatly from parallel envs (SubprocVecEnv)
        train_env = SubprocVecEnv([make_train_env for _ in range(args.num_envs)])
        eval_env = DummyVecEnv([make_train_env]) 

        # 5. DEFINE CALLBACKS
        metric_cb = RocketMetricsCallback()
        
        # Checkpoint: Save model at regular intervals
        checkpoint_cb = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=ckpt_dir,
            name_prefix="ppo_rocket"
        )
        
        # EvalCallback: Evaluate and save the best model
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model"),
            log_path=log_dir,
            eval_freq=max(args.eval_freq // args.num_envs, 1),
            deterministic=True,
            render=False
            # Removed the unsupported 'callback_on_eval' argument from previous error fix
        )
        
        # Combine callbacks
        callback_list = CallbackList([metric_cb, checkpoint_cb, eval_cb])

        # 6. INITIALIZE OR RESUME MODEL
        if args.model:
            potential_paths = [
                os.path.join(args.model, "best_model", "best_model.zip"),
                os.path.join(args.model, "best_model.zip"),
                os.path.join(args.model, "final_model.zip")
            ]
            
            load_path = None
            for p in potential_paths:
                if os.path.exists(p):
                    load_path = p
                    break
            
            if load_path:
                print(f"🔄 RESUMING TRAINING from: {load_path}")
                # --- CHANGED: Use PPO.load ---
                model = PPO.load(
                    load_path,
                    env=train_env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    learning_rate=args.learning_rate,
                    gamma=args.gamma
                )
            else:
                print(f"❌ Could not find model file in {args.model}. Check path.")
                return
        else:
            print(f"🆕 STARTING NEW TRAINING")
            # --- CHANGED: Use PPO constructor with PPO-specific HPs ---
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                clip_range=args.clip_range,
                seed=args.seed,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]), # Separate networks for PPO
            )

        print("------------------------------------------")
        print(f"🚀 PPO TRAINING START | Name: {args.name}")
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