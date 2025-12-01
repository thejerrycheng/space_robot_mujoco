import time
from stable_baselines3 import PPO
from rocket_env.rocket_landing_env import RocketLandingEnv

if __name__ == "__main__":
    env = RocketLandingEnv(render_mode="human")
    model = PPO.load("rocket_ppo")

    obs, info = env.reset()
    done = False
    trunc = False

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        time.sleep(0.01)

    env.close()
