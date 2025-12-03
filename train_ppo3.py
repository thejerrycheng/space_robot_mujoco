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
import gymnasium as gym
from gymnasium import spaces

# RL Libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Import your environment
from rocket_env.rocket_2_env import RocketLandingEnv

# ==============================================================================
#   SAFE ENVIRONMENT WRAPPER (Fixes KeyError)
# ==============================================================================
class SafeRocketEnv(RocketLandingEnv):
    """
    Subclass that ensures 'success' and 'semi_success' keys exist in the info dict.
    This prevents the Monitor wrapper from crashing if the base env doesn't return them.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Ensure keys exist for Monitor
        if "success" not in info:
            info["success"] = False
        if "semi_success" not in info:
            info["semi_success"] = False
            
        return obs, reward, terminated, truncated, info

# ==============================================================================
#   PAPER OBSERVATION WRAPPER
# ==============================================================================
class PaperObsWrapper(gym.Wrapper):
    """
    Wraps the RocketLandingEnv to produce the specific observation vector 
    described in Equation 31 of the paper:
    obs = [v_error, q, omega, r_z, t_go]
    
    Where:
    - v_error: Velocity error relative to a guidance field (v - v_targ)
    - q: Attitude quaternion [w, x, y, z]
    - omega: Rotational velocity [wx, wy, wz]
    - r_z: Altitude
    - t_go: Time-to-go estimate (Range / Velocity)
    """
    def __init__(self, env):
        super().__init__(env)
        
        # New Observation Space: 3 (v_err) + 4 (q) + 3 (w) + 1 (r_z) + 1 (t_go) = 12 dims
        high = np.inf * np.ones(12, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Guidance Parameters from Paper (approximate/inferred)
        self.v0 = 70.0        # Initial velocity magnitude reference
        self.waypoint_z = 15.0 # Altitude switch for guidance law
        self.tau_1 = 20.0     # Time constant for high altitude
        self.tau_2 = 100.0    # Time constant for low altitude

    def _get_paper_obs(self):
        # Access the raw MuJoCo data from the unwrapped environment
        # We need this because the standard obs might not have Quaternions or raw w
        unwrapped_env = self.env.unwrapped
        data = unwrapped_env.data
        
        # 1. Extract Raw State
        pos = data.xpos[unwrapped_env.rocket_bid].copy()
        vel = data.cvel[unwrapped_env.rocket_bid][3:].copy()
        # Quaternion [w, x, y, z]
        quat = data.qpos[unwrapped_env.qpos_adr+3 : unwrapped_env.qpos_adr+7].copy()
        # Angular Velocity [wx, wy, wz]
        omega = data.qvel[unwrapped_env.qvel_adr+3 : unwrapped_env.qvel_adr+6].copy()
        
        r_z = pos[2] # Altitude
        
        # 2. Compute Time-to-Go (t_go)
        # t_go = Range / ||Velocity||
        r_mag = np.linalg.norm(pos)
        v_mag = np.linalg.norm(vel) + 1e-6 # Avoid div/0
        t_go = r_mag / v_mag
        
        # 3. Compute v_targ (Guidance Law - Eq 29a/29d from Paper)
        if r_z > self.waypoint_z:
            # High Altitude Guidance: Gaze Heuristic
            # v_targ = -v_o * (r_hat) * (1 - exp(-t_go/tau))
            
            # Vector pointing from target (waypoint) to rocket
            # Note: Paper defines r_hat based on relative pos
            r_rel = pos.copy()
            r_rel[2] -= self.waypoint_z
            
            # Unit vector
            r_unit = r_rel / (np.linalg.norm(r_rel) + 1e-6)
            
            tau = self.tau_1
            factor = 1.0 - np.exp(-t_go / tau)
            
            v_targ = -self.v0 * r_unit * factor
        else:
            # Low Altitude: Vertical Descent
            # v_targ = [0, 0, -2.0]
            v_targ = np.array([0.0, 0.0, -2.0])

        # 4. Compute Velocity Error
        v_error = vel - v_targ
        
        # 5. Assemble Vector: [v_error(3), q(4), w(3), r_z(1), t_go(1)]
        obs = np.concatenate([v_error, quat, omega, [r_z], [t_go]]).astype(np.float32)
        return obs

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        obs = self._get_paper_obs()
        return obs, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_paper_obs()
        return obs, reward, terminated, truncated, info

# ==============================================================================
#   ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training (Paper Observation)")
    
    parser.add_argument("--name", type=str, default="paper_obs", 
                        help="Custom tag for the run")
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--viz_freq", type=int, default=20_000)
    parser.add_argument("--reward", type=str, default="flip_and_fuel")

    # Hyperparameters
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
    try:
        module_path = f"rocket_env.rewards.{reward_name}"
        mod = importlib.import_module(module_path)
        return mod.compute_reward
    except ImportError as e:
        print(f"\n❌ Error loading reward: {reward_name}")
        raise e

# ==============================================================================
#   CALLBACKS
# ==============================================================================
class RocketCallback(BaseCallback):
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
                    if "success" in infos[idx]:
                        self.success_buffer.append(float(infos[idx]["success"]))
                    if "semi_success" in infos[idx]:
                        self.semi_success_buffer.append(float(infos[idx]["semi_success"]))

        if self.viz_freq > 0 and (self.episode_count - self.last_viz_count >= self.viz_freq):
            self.last_viz_count = self.episode_count
            self.trigger_visualization()
        return True

    def _on_rollout_end(self) -> None:
        if len(self.success_buffer) > 0:
            self.logger.record("rollout/success_rate", np.mean(self.success_buffer) * 100)
        if len(self.semi_success_buffer) > 0:
            self.logger.record("rollout/semi_success_rate", np.mean(self.semi_success_buffer) * 100)

    def trigger_visualization(self):
        print(f"\n🎥 VISUALIZING EPISODE (Count: {self.episode_count})...")
        # Instantiate env with Wrapper for visualization
        raw_env = SafeRocketEnv(render_mode="human") # Use SafeRocketEnv here too
        viz_env = PaperObsWrapper(raw_env) # WRAP IT
        
        obs_rms = self.training_env.obs_rms
        obs, _ = viz_env.reset()
        done = False
        
        while not done:
            norm_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
            action, _ = self.model.predict(norm_obs, deterministic=True)
            obs, _, terminated, truncated, _ = viz_env.step(action)
            viz_env.render()
            done = terminated or truncated
        
        viz_env.close()
        print("🎥 Visualization complete.\n")

# ==============================================================================
#   MAIN
# ==============================================================================
def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_paper_{args.name}_{timestamp}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting Run: {run_name}")
    print(f"📝 Observation: Paper Implementation [v_err, q, w, r_z, t_go]")
    
    reward_func = load_reward_function(args.reward)

    # Use SafeRocketEnv to prevent Monitor crashes due to missing keys
    env = make_vec_env(
        SafeRocketEnv, 
        n_envs=args.num_envs, 
        seed=args.seed,
        monitor_dir=log_dir,
        monitor_kwargs={'info_keywords': ('success', 'semi_success')}, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"reward_func": reward_func},
        wrapper_class=PaperObsWrapper  # <--- APPLY WRAPPER HERE
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=args.learning_rate, n_steps=args.n_steps,
        batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1, tensorboard_log=log_dir, device=args.device, seed=args.seed
    )

    ckpt_cb = CheckpointCallback(save_freq=max(50_000 // args.num_envs, 1), save_path=os.path.join(log_dir, "checkpoints"), name_prefix="ppo_paper")
    rocket_cb = RocketCallback(viz_freq_episodes=args.viz_freq)

    print("💪 Training started...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=[ckpt_cb, rocket_cb], progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")

    model.save(os.path.join(log_dir, "final_model"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    env.close()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()