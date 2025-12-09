import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Dict, Any


from rocket_env.rocket_gym_discrete import RocketLandingEnv

# --- Configuration ---
NUM_EPISODES = 10
RENDER_DELAY = 0.01  # Delay to slow down the visualization

def run_test_episodes(num_episodes: int, render: bool = True) -> List[Dict[str, Any]]:
    """
    Runs a specified number of episodes using a random discrete agent 
    and collects performance data.
    """
    render_mode = "human" if render else None
    
    # NOTE: Adjust the XML path if necessary.
    env = RocketLandingEnv(
        xml_path="assets/mjcf/realistic_param.xml", 
        render_mode=render_mode
    )
    
    print(f"--- Running {num_episodes} Test Episodes (Discrete Agent) ---")
    print(f"Action Space Size: {env.action_space.n}")
    print("-" * 50)

    results = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=int(time.time()))
        
        terminated = False
        truncated = False
        episode_reward = 0.0
        step_count = 0
        
        start_time = time.time()
        
        print(f"▶️  Episode {episode}/{num_episodes} started...")

        while not terminated and not truncated:
            # ⭐️ Discrete Action Sampling: Gets a single integer (0 to 97)
            action_index = env.action_space.sample() 
            
            obs, reward, terminated, truncated, info = env.step(action_index)
            
            episode_reward += reward
            step_count += 1
            
            if render:
                env.render()
                time.sleep(RENDER_DELAY)
        
        # --- Episode Logging ---
        duration = time.time() - start_time
        is_success = info.get("is_success", False)
        fuel_remaining = info['fuel_remaining']
        
        status_emoji = "✅ SUCCESS" if is_success else ("💥 FAILED (Grounded)" if terminated else "⏱️ TRUNCATED")
        
        print(f"   {status_emoji} | Steps: {step_count:4d} | Reward: {episode_reward:8.2f} | Fuel Left: {fuel_remaining:.2f} kg | Time: {duration:.2f}s")

        results.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': step_count,
            'success': is_success,
            'fuel_remaining': fuel_remaining
        })

    env.close()
    return results

def visualize_results(results: List[Dict[str, Any]]):
    """
    Creates a simple visualization plot for the test results. 
    
    """
    episodes = [r['episode'] for r in results]
    rewards = [r['reward'] for r in results]
    successes = [r['success'] for r in results]
    
    colors = ['g' if s else 'r' for s in successes]
    
    plt.figure(figsize=(10, 5))
    plt.bar(episodes, rewards, color=colors)
    plt.xlabel("Episode Number")
    plt.ylabel("Total Episode Reward")
    plt.title(f"Random Discrete Agent Performance over {len(episodes)} Episodes")
    
    # Add success markers
    for i, s in enumerate(successes):
        plt.text(episodes[i], rewards[i], '✅' if s else '❌', ha='center', va='bottom')
        
    plt.xticks(episodes)
    plt.grid(axis='y', linestyle='--')
    plt.show()

if __name__ == "__main__":
    # --- Execute and Visualize ---
    
    # Collect data from 10 test runs (with rendering enabled)
    test_results = run_test_episodes(NUM_EPISODES, render=True)
    
    # Generate the plot
    visualize_results(test_results)
    
    total_successes = sum(r['success'] for r in test_results)
    print("\n--- Summary ---")
    print(f"Total Successes: {total_successes}/{NUM_EPISODES}")