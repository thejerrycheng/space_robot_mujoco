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

# RL
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Env
from rocket_env.polar_rocket_env import RocketLandingEnv


# ==============================================================================
#   ARGUMENT PARSER (Updated for SAC)
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SAC Training for Rocket Landing (Polar)")

    # Experiment
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--viz_freq", type=int, default=20_000)

    # Reward
    parser.add_argument("--reward", type=str, default="polar_vel_field")

    # SAC Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=2_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train_freq", type=int, default=64)
    parser.add_argument("--gradient_steps", type=int, default=64)
    parser.add_argument("--ent_coef", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


# ==============================================================================
#   REWARD LOADER
# ==============================================================================
def load_reward_function(reward_name):
    try:
        module_path = f"rocket_env.rewards.{reward_name}"
        mod = importlib.import_module(module_path)
        return mod.compute_reward
    except ImportError as e:
        print(f"\n❌ Error loading reward: {reward_name}")
        raise e


# ==============================================================================
#   CALLBACKS (reused from PPO)
# ==============================================================================
class RocketCallback(BaseCallback):
    def __init__(self, viz_freq_episodes=20000, verbose=0):
        super().__init__(verbose)
        self.success_buffer = deque(maxlen=100)
        self.semi_success_buffer = deque(maxlen=100)
        self.viz_freq = viz_freq_episodes
        self.episode_count = 0
        self.last_viz_count = 0

    def _on_step(self):
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        num_dones = np.sum(dones)
        if num_dones > 0:
            self.episode_count += num_dones

            for idx, done in enumerate(dones):
                if done:
                    if "success" in infos[idx]:
                        self.success_buffer.append(float(infos[idx]["success"]))
                    if "semi_success" in infos[idx]:
                        self.semi_success_buffer.append(float(infos[idx]["semi_success"]))

        if self.viz_freq > 0 and (self.episode_count - self.last_viz_count >= self.viz_freq):
            self.last_viz_count = self.episode_count
            self.trigger_visualization()

        return True

    def _on_rollout_end(self):
        if len(self.success_buffer) > 0:
            self.logger.record("rollout/success_rate", np.mean(self.success_buffer) * 100)
        if len(self.semi_success_buffer) > 0:
            self.logger.record("rollout/semi_success_rate", np.mean(self.semi_success_buffer) * 100)

    def trigger_visualization(self):
        print(f"\n🎥 VISUALIZING EPISODE (Count: {self.episode_count})...")

        try:
            viz_env = RocketLandingEnv(render_mode="human")
            obs_rms = self.training_env.obs_rms

            obs, _ = viz_env.reset()
            done = False

            while not done:
                norm_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                obs, _, terminated, truncated, _ = viz_env.step(action)
                viz_env.render()
                time.sleep(0.01)
                done = terminated or truncated

            viz_env.close()
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")


# ==============================================================================
#   PLOTTING (same as PPO)
# ==============================================================================
def plot_training_results(log_dir):
    monitor_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_path):
        return

    try:
        df = pd.read_csv(monitor_path, skiprows=1)
        df["rolling_reward"] = df["r"].rolling(window=100).mean()

        success_col = next((c for c in df.columns if "success" in c and "semi" not in c), None)
        semi_success_col = next((c for c in df.columns if "semi_success" in c), None)

        if success_col:
            df["rolling_success"] = df[success_col].rolling(100).mean() * 100
        if semi_success_col:
            df["rolling_semi_success"] = df[semi_success_col].rolling(100).mean() * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.set_title("Training Metrics (Polar Env)")
        ax1.plot(df["rolling_reward"], color="blue")
        ax1.set_ylabel("Reward")
        ax1.grid()

        if success_col and "rolling_success" in df:
            ax2.plot(df["rolling_success"], color="green", label="Success %")
        if semi_success_col and "rolling_semi_success" in df:
            ax2.plot(df["rolling_semi_success"], color="orange", label="Semi-Success %")

        ax2.legend()
        ax2.grid()
        plt.xlabel("Episode")
        plt.savefig(os.path.join(log_dir, "training_metrics.png"))
        plt.close()

    except Exception:
        pass


# ==============================================================================
#   MAIN — SAC VERSION
# ==============================================================================
def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"sac_polar_{args.name}_{timestamp}"
        if args.name else
        f"sac_polar_{args.reward}_{timestamp}"
    )

    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"🚀 Starting SAC Run: {run_name}")
    print(f"❄️ Env: Polar Rocket Landing")
    print(f"📂 Logging to: {log_dir}")

    # Load reward
    reward_func = load_reward_function(args.reward)

    # Create VecEnv
    env = make_vec_env(
        RocketLandingEnv,
        n_envs=args.num_envs,
        seed=args.seed,
        monitor_dir=log_dir,
        monitor_kwargs={"info_keywords": ("success", "semi_success")},
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"reward_func": reward_func},
    )

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # SAC Agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
        seed=args.seed,
    )

    # Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // args.num_envs, 1),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_polar",
    )

    rocket_cb = RocketCallback(viz_freq_episodes=args.viz_freq)

    # Train
    print("💪 Training started...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[ckpt_cb, rocket_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")

    # Save
    model.save(os.path.join(log_dir, "final_model"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    plot_training_results(log_dir)
    env.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
