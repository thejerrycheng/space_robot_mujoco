from rocket_env.rocket_3_env import RocketLandingEnv
import time

env = RocketLandingEnv(render_mode="human")

obs, _ = env.reset()

for i in range(1000):
    # Random actions for now
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        print(f"Episode End. Outcome: {info.get('outcome', 'Unknown')} | Level: {info['curriculum_level']}")
        obs, _ = env.reset()
        
    time.sleep(0.01)