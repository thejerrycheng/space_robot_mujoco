import os
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from rocket_env.rocket_landing_env import RocketLandingEnv



# ======================================================================
#   ENV FACTORY
# ======================================================================
def make_env(render_mode="human"):
    def _init():
        return RocketLandingEnv(render_mode=render_mode)
    return _init



# ======================================================================
#   CUSTOM CALLBACK: SAVE + VISUALIZE EVERY 200 TRAINING ITERATIONS
# ======================================================================
class SaveAndVisualizeCallback(BaseCallback):

    def __init__(self, save_freq=200, log_dir="", vec_env=None, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.vec_env_ref = vec_env
        self.counter = 0

        os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)

    def _on_step(self) -> bool:

        n_steps = self.model.n_steps

        if self.model.num_timesteps % n_steps == 0:
            self.counter += 1

            if self.counter % self.save_freq == 0:

                print(f"\n===== ITERATION {self.counter}: Saving model + Visualizing policy =====\n")

                # SAVE MODEL
                save_path = f"{self.log_dir}/checkpoints/ppo_step_{self.counter}.zip"
                self.model.save(save_path)

                if hasattr(self.vec_env_ref, "save"):
                    self.vec_env_ref.save(f"{self.log_dir}/checkpoints/vecnorm_step_{self.counter}.pkl")

                # VISUALIZE POLICY
                self._visualize_policy()

        return True


    def _visualize_policy(self):
        """
        Uses mujoco-python-viewer for rendering.
        """
        from mujoco_python_viewer import Viewer  # <- NEW viewer

        env = RocketLandingEnv(render_mode=None)   # turn off built-in render
        obs, _ = env.reset()

        # norm obs if needed
        if isinstance(self.vec_env_ref, VecNormalize):
            obs = self.vec_env_ref.normalize_obs(obs)

        viewer = Viewer(env.model, env.data)

        for _ in range(600):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            viewer.render()  # <--- NEW viewer API

            time.sleep(0.01)

            if terminated or truncated:
                break

        viewer.close()
        env.close()



# ======================================================================
#   MAIN TRAINING PIPELINE
# ======================================================================
def train():

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    # ---------------------------------------------
    # 1. Create training environment
    # ---------------------------------------------
    train_env = DummyVecEnv([make_env(render_mode=None)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---------------------------------------------
    # 2. Logging
    # ---------------------------------------------
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # ---------------------------------------------
    # 3. Create PPO agent
    # ---------------------------------------------
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=f"{log_dir}/tb/",
    )
    model.set_logger(logger)

    # ---------------------------------------------
    # 4. Custom callback for save + visualization
    # ---------------------------------------------
    callback = SaveAndVisualizeCallback(
        save_freq=200,
        log_dir=log_dir,
        vec_env=train_env
    )

    # ---------------------------------------------
    # 5. Train
    # ---------------------------------------------
    total_steps = 2_000_000

    print("\n🚀 Starting PPO training with save + visualize pipeline...\n")

    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        progress_bar=True
    )

    print("\n🎉 Training complete. Saving final model.\n")

    model.save(f"{log_dir}/final_model")
    train_env.save(f"{log_dir}/vecnorm_final.pkl")


if __name__ == "__main__":
    print("\n=== Rocket Landing PPO Training Pipeline ===\n")
    train()
