import os
import sys
import time
import datetime
import argparse
import importlib
import gc 
import csv  # <--- Required for CSV logging
import numpy as np
import gymnasium as gym
import glob

# WRAPPERS
from gymnasium.wrappers import TimeLimit 

# ALGORITHMS
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# IMPORT YOUR ENV CLASS
# (Make sure this matches your actual file name, e.g., rocket_3_env or rocket_gym_env)
from rocket_env.rocket_gym_env import RocketLandingEnv 

# ================================================================
#   HELPER: DYNAMIC REWARD LOADING
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
#   CALLBACK: CSV EPISODE LOGGING
# ================================================================
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, log_file_path, verbose=0):
        super(EpisodeLoggerCallback, self).__init__(verbose)
        self.log_file_path = log_file_path
        
        # Create file and write header if it doesn't exist
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step",      # Total timesteps taken by model
                    "wall_time",        # Real world time
                    "episode_reward",   # Total reward for the episode
                    "episode_length",   # Steps in the episode
                    "success",          # 1 or 0
                    "fuel_remaining",   # Fuel left
                    "landing_dist",     # Distance from center
                    "curriculum_level"  # Current difficulty
                ])

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                
                # Check for Monitor wrapper stats
                if 'episode' not in info:
                    continue 

                episode_stats = info['episode']
                ep_rew = episode_stats.get('r', 0.0)
                ep_len = episode_stats.get('l', 0)
                
                # Custom environment specific info
                is_success = 1 if info.get('is_success', False) else 0
                fuel = info.get('fuel_remaining', 0.0)
                dist = info.get('lateral_dist', 0.0)
                level = info.get('curriculum_level', 0)
                
                # Write to CSV
                with open(self.log_file_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.num_timesteps,
                        time.time(),
                        ep_rew,
                        ep_len,
                        is_success,
                        fuel,
                        dist,
                        level
                    ])
        return True

# ================================================================
#   CALLBACK: METRICS LOGGING (Tensorboard)
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
                self.success_history.append(1.0 if info.get('is_success', False) else 0.0)
                self.curriculum_history.append(info.get('curriculum_level', 0))
                self.fuel_history.append(info.get('fuel_remaining', 0))
                if 'lateral_dist' in info:
                     self.distance_history.append(info['lateral_dist'])
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
    parser.add_argument("--model", type=str, default=None, help="Path to a previous run folder")

    # --- Experiment Identity ---
    parser.add_argument("--name", type=str, default="default", help="Name of the experiment")
    parser.add_argument("--reward", type=str, default="landing_reward", help="Reward file name")
    
    # --- PPO Hyperparameters ---
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--max_episode_steps", type=int, default=1500, help="Force episode finish after X steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--n_steps", type=int, default=2048, help="Steps per env per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=10, help="Optimization epochs per update")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    
    # --- System & Logging ---
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--checkpoint_freq", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 1. SETUP PATHS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"gym_ppo_{args.name}_{timestamp}"
    
    # Model Paths
    base_model_path = "./models/"
    run_dir = os.path.join(base_model_path, folder_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_log_dir = os.path.join(run_dir, "logs")
    
    # CSV Results Paths
    base_results_path = "./results/"
    results_dir = os.path.join(base_results_path, folder_name)
    csv_file_path = os.path.join(results_dir, "training_log.csv")
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"📁 Saving Models to: {run_dir}")
    print(f"📊 Saving CSV Data to: {csv_file_path}")

    # Initialize env vars
    train_env = None
    eval_env = None

    try:
        # 2. LOAD REWARD CLASS
        RewardClass = load_reward_class(args.reward)

        # 3. DEFINE ENV MAKER
        def make_env_instance(render_mode=None):
            rewarder = RewardClass({'start_fuel': 4_500_000.0}) 
            env = RocketLandingEnv(render_mode=render_mode)
            
            # 3a. Apply TimeLimit (Force episode end) - CRITICAL for CSV logging
            env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
            
            if hasattr(env, "rewarder"):
                env.rewarder = rewarder
            
            # 3b. Apply Monitor (Captures stats from finished episodes)
            env = Monitor(env) 
            return env

        def make_train_env():
            return make_env_instance(render_mode=None)

        # 4. CREATE ENVS
        train_env = SubprocVecEnv([make_train_env for _ in range(args.num_envs)])
        eval_env = DummyVecEnv([make_train_env]) 

        # 5. DEFINE CALLBACKS
        metric_cb = RocketMetricsCallback()
        csv_cb = EpisodeLoggerCallback(csv_file_path)
        
        checkpoint_cb = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=ckpt_dir,
            name_prefix="ppo_rocket"
        )
        
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model"),
            log_path=tb_log_dir,
            eval_freq=max(args.eval_freq // args.num_envs, 1),
            deterministic=True,
            render=False
        )
        
        callback_list = CallbackList([metric_cb, checkpoint_cb, eval_cb, csv_cb])

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
                model = PPO.load(
                    load_path,
                    env=train_env,
                    verbose=1,
                    tensorboard_log=tb_log_dir,
                    learning_rate=args.learning_rate,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    ent_coef=args.ent_coef,
                )
            else:
                print(f"❌ Could not find model file in {args.model}. Check path.")
                return
        else:
            print(f"🆕 STARTING NEW TRAINING")
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                tensorboard_log=tb_log_dir,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                seed=args.seed,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            )

        print("------------------------------------------")
        print(f"🚀 TRAINING START | Name: {args.name} | Algo: PPO")
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