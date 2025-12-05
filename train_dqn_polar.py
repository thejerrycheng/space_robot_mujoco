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
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Env + wrapper
from rocket_env.polar_rocket_env import RocketLandingEnv
from discrete_action_wrapper import DiscreteActionWrapper


# ==============================================================================
#   PARSE ARGS
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DQN Training for Rocket Landing (Polar)")

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)

    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--viz_freq", type=int, default=20_000)

    parser.add_argument("--reward", type=str, default="polar_vel_field")

    # ------------------------------
    #  NEW: Action Discretization Params
    # ------------------------------
    parser.add_argument("--thrust_bins", type=int, default=5)
    parser.add_argument("--pitch_bins", type=int, default=5)
    parser.add_argument("--roll_bins", type=int, default=5)

    parser.add_argument("--thrust_min", type=float, default=0.0)
    parser.add_argument("--thrust_max", type=float, default=1.0)

    parser.add_argument("--pitch_min", type=float, default=-0.2)
    parser.add_argument("--pitch_max", type=float, default=0.2)

    parser.add_argument("--roll_min", type=float, default=-0.2)
    parser.add_argument("--roll_max", type=float, default=0.2)

    # DQN Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--target_update_interval", type=int, default=1000)
    parser.add_argument("--exploration_fraction", type=float, default=0.3)
    parser.add_argument("--exploration_final_eps", type=float, default=0.02)

    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


# ==============================================================================
#   LOAD REWARD
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
# CALLBACKS (same as PPO/SAC)
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

    def trigger_visualization(self):
        print(f"\n🎥 VISUALIZING EPISODE (Count: {self.episode_count})...")

        try:
            viz_env = DiscreteActionWrapper(
                RocketLandingEnv(render_mode="human"),
                thrust_bins=self.model.thrust_bins,
                pitch_bins=self.model.pitch_bins,
                roll_bins=self.model.roll_bins,
            )

            obs_rms = self.training_env.obs_rms
            obs, _ = viz_env.reset()
            done = False

            while not done:
                norm_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                action, _ = self.model.predict(norm_obs, deterministic=True)
                obs, _, terminated, truncated, _ = viz_env.step(action)
                viz_env.render()
                done = terminated or truncated
                time.sleep(0.01)

            viz_env.close()

        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")


# ==============================================================================
#   PLOT TRAINING RESULTS
# ==============================================================================
def plot_training_results(log_dir):
    path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(path):
        return

    df = pd.read_csv(path, skiprows=1)
    df["rolling_reward"] = df["r"].rolling(100).mean()

    success_col = next((c for c in df.columns if "success" in c and "semi" not in c), None)
    semi_success_col = next((c for c in df.columns if "semi_success" in c), None)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["rolling_reward"], label="Reward")

    if success_col:
        ax.plot(df[success_col].rolling(100).mean() * 100, label="Success %")
    if semi_success_col:
        ax.plot(df[semi_success_col].rolling(100).mean() * 100, label="Semi-Success %")

    ax.legend()
    ax.grid()
    plt.savefig(os.path.join(log_dir, "training_metrics.png"))
    plt.close()


# ==============================================================================
#   MAIN — DQN VERSION
# ==============================================================================
def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"dqn_polar_{args.name}_{timestamp}"
        if args.name else
        f"dqn_polar_{args.reward}_{timestamp}"
    )

    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"🚀 Starting DQN Run: {run_name}")

    reward_func = load_reward_function(args.reward)

    # --- Create VecEnv with discrete action wrapper ---
    def make_env():
        return DiscreteActionWrapper(
            RocketLandingEnv(reward_func=reward_func),
            thrust_bins=args.thrust_bins,
            pitch_bins=args.pitch_bins,
            roll_bins=args.roll_bins,
            thrust_range=(args.thrust_min, args.thrust_max),
            pitch_range=(args.pitch_min, args.pitch_max),
            roll_range=(args.roll_min, args.roll_max),
        )

    env = make_vec_env(
        make_env,
        n_envs=args.num_envs,
        seed=args.seed,
        monitor_dir=log_dir,
        monitor_kwargs={"info_keywords": ("success", "semi_success")},
        vec_env_cls=SubprocVecEnv,
    )

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # --- DQN Agent ---
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=1,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
        seed=args.seed,
    )

    # store bins inside model so callback can access
    model.thrust_bins = args.thrust_bins
    model.pitch_bins = args.pitch_bins
    model.roll_bins = args.roll_bins

    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // args.num_envs, 1),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="dqn_polar",
    )

    rocket_cb = RocketCallback(viz_freq_episodes=args.viz_freq)

    # --- Train ---
    print("💪 Training started...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[ckpt_cb, rocket_cb],
        progress_bar=True,
    )

    # --- Save ---
    model.save(os.path.join(log_dir, "final_model"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    plot_training_results(log_dir)
    env.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
