import os
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from rocket_env.rocket_landing_env import RocketLandingEnv


# ======================================================================
#   ENV FACTORY
# ======================================================================
def make_env(render_mode=None):
    def _init():
        return RocketLandingEnv(render_mode=render_mode)
    return _init


# ======================================================================
#   CUSTOM CALLBACK: SAVE + VISUALIZE EVERY N STEPS
# ======================================================================
class SaveAndVisualizeCallback(BaseCallback):
    def __init__(self, save_freq_steps: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq_steps = save_freq_steps
        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    def _on_step(self) -> bool:
        # num_timesteps is global across training
        if self.model.num_timesteps % self.save_freq_steps == 0:
            step = self.model.num_timesteps
            if self.verbose:
                print(f"\n===== SAC SAVE & VISUALIZE at step {step} =====")

            # Save checkpoint
            ckpt_path = os.path.join(self.log_dir, "checkpoints", f"sac_rocket_step_{step}.zip")
            self.model.save(ckpt_path)

            # Visualize policy briefly
            self._visualize_policy()

        return True

    def _visualize_policy(self):
        """
        Short rollout in a human-rendered env.
        """
        from rocket_env.rocket_landing_env import RocketLandingEnv
        env = RocketLandingEnv(render_mode="human")
        obs, _ = env.reset()

        for _ in range(600):  # ~6 seconds if dt=0.01 with 10ms sleep
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.01)
            if terminated or truncated:
                break

        env.close()


# ======================================================================
#   MAIN SAC TRAINING
# ======================================================================
def train_sac():
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs_sac", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create single environment (off-policy algorithms like SAC don't need VecEnv)
    env = make_env(render_mode=None)()

    # Configure logging (stdout, csv, tensorboard)
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=(64, "step"),
        gradient_steps=64,
        ent_coef="auto",
        target_entropy="auto",
        tensorboard_log=os.path.join(log_dir, "tb"),
    )
    model.set_logger(logger)

    # Callback: save + visualize every 200k steps (tune as you like)
    callback = SaveAndVisualizeCallback(
        save_freq_steps=200_000,
        log_dir=log_dir,
        verbose=1,
    )

    total_timesteps = 2_000_000
    print(f"\n🚀 Starting SAC training for {total_timesteps} timesteps...\n")

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10,
        callback=callback,
        progress_bar=True,
    )

    print("\n✅ SAC training finished. Saving final model.\n")
    final_path = os.path.join(log_dir, "sac_rocket_final")
    model.save(final_path)

    env.close()
    print(f"📁 Final model saved to: {final_path}")


if __name__ == "__main__":
    train_sac()
