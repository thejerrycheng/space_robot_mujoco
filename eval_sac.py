import time
from stable_baselines3 import SAC
from rocket_env.rocket_landing_env import RocketLandingEnv


def evaluate_sac(model_path: str):
    env = RocketLandingEnv(render_mode="human")
    model = SAC.load(model_path)

    while True:
        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            time.sleep(0.01)


if __name__ == "__main__":
    # point this at either:
    #  - a checkpoint, e.g. "runs_sac/2025-01-01_12-00-00/checkpoints/sac_rocket_step_200000.zip"
    #  - or the final model, e.g. "runs_sac/2025-01-01_12-00-00/sac_rocket_final.zip"
    MODEL_PATH = "runs_sac/your_run/sac_rocket_final.zip"
    evaluate_sac(MODEL_PATH)
