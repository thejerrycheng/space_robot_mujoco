from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rocket_env.rocket_landing_env import RocketLandingEnv


def make_env():
    return RocketLandingEnv(render_mode=None)


if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="logs/",
    )

    model.learn(total_timesteps=1_000_000)
    model.save("rocket_ppo")
    env.close()
