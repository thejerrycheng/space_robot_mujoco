import numpy as np
from scipy.spatial.transform import Rotation as R

class RocketReward:
    """
    Simplified reward function focused only on landing performance:
    1. Low lateral position error.
    2. Low final velocity.
    3. Near-zero tilt angle (upright).
    """

    def __init__(self, config):
        # Configuration is typically passed from the environment
        # No fuel state tracking needed anymore
        
        # Reward Weights (Tune these based on training results)
        self.W_POS = 10.0      # Weight for lateral position (distance from target)
        self.W_VEL = 5.0       # Weight for vertical/horizontal speed
        self.W_TILT = 10.0      # Weight for tilt angle
        self.W_SUCCESS = 5000.0 # Large reward for successful termination
        self.W_CRASH = -500.0  # Penalty for crash (non-successful termination)
        self.W_ALIVE = 0.0     # Small survival bonus (can be set to 0)

    def compute(self, state, action, terminated, truncated, success):
        """
        Computes the reward for the current time step.

        Args:
            state (dict): Dictionary of state variables from _get_state_dict().
            action (np.array): The action taken by the agent (thrust, yaw, pitch).
            terminated (bool): True if episode ended due to crash/success/boundary.
            truncated (bool): True if episode ended due to step limit or out-of-bounds.
            success (bool): True if the landing was successful.

        Returns:
            (float, dict): Total reward and a dictionary of reward components.
        """
        total_reward = 0.0
        r_info = {}

        # 1. Survival/Progress Bonus
        total_reward += self.W_ALIVE
        
        # --- 2. Step Reward (Continuous Shaping) ---
        
        # Penalize Lateral Distance
        lateral_dist = state['lateral_dist']
        # Normalized by a large factor for smoother scaling
        pos_reward = -self.W_POS * (lateral_dist / 1000.0) 
        total_reward += pos_reward
        r_info['r/pos'] = pos_reward

        # Penalize Total Velocity
        vel_mag = np.linalg.norm(state['vel'])
        # Quadratic penalty encourages very low speeds
        vel_reward = -self.W_VEL * (vel_mag / 100.0)**2 
        total_reward += vel_reward
        r_info['r/vel'] = vel_reward

        # Penalize Tilt Angle
        tilt_deg = state['tilt']
        # Quadratic penalty for tilt angle normalized by max tilt (90 deg)
        tilt_reward = -self.W_TILT * (tilt_deg / 90.0)**2
        total_reward += tilt_reward
        r_info['r/tilt'] = tilt_reward

        # --- 3. Terminal Reward ---
        if terminated or truncated:
            
            # Penalize any remaining lateral distance at termination
            final_pos_penalty = -self.W_POS * (lateral_dist) 
            total_reward += final_pos_penalty
            r_info['r/final_pos_penalty'] = final_pos_penalty
            
            # Penalize any remaining velocity at termination
            final_vel_penalty = -self.W_VEL * (vel_mag)**2 
            total_reward += final_vel_penalty
            r_info['r/final_vel_penalty'] = final_vel_penalty
            
            if success:
                # Big positive reward for meeting all criteria
                final_reward = self.W_SUCCESS
                total_reward += final_reward
                r_info['r/terminal'] = final_reward
            else:
                # Big negative reward for crashing or exceeding boundaries
                final_reward = self.W_CRASH
                total_reward += final_reward
                r_info['r/terminal'] = final_reward
            
            # Since fuel rewards are removed, no need to reset last_fuel. 
            # We still include the fuel_remaining in r_info for logging, though it won't affect the reward.

        return total_reward, r_info