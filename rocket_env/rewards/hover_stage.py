import numpy as np
from scipy.spatial.transform import Rotation as R

class RocketReward:
    """
    Simplified reward function focused only on landing performance.
    
    MODIFICATION: Speed penalty is only active when altitude is below 100 meters.
    """

    def __init__(self, config):
        # Configuration is typically passed from the environment
        
        # Reward Weights (Tune these based on training results)
        self.W_POS = 10.0      # Weight for lateral position (distance from target)
        self.W_VEL = 5.0       # Weight for vertical/horizontal speed
        self.W_TILT = 10.0      # Weight for tilt angle
        self.W_SUCCESS = 5000.0 # Large reward for successful termination
        self.W_CRASH = -500.0  # Penalty for crash (non-successful termination)
        self.W_ALIVE = 0.0     # Small survival bonus (can be set to 0)
        
        # NEW: Altitude threshold for activating speed penalty
        self.SPEED_PENALTY_ALT_THRESHOLD = 100.0 

    def compute(self, state, action, terminated, truncated, success):
        """
        Computes the reward for the current time step.
        """
        total_reward = 0.0
        r_info = {}

        # 1. Survival/Progress Bonus
        total_reward += self.W_ALIVE
        
        # --- 2. Step Reward (Continuous Shaping) ---
        
        # Penalize Lateral Distance (Position shaping is always active)
        lateral_dist = state['lateral_dist']
        pos_reward = -self.W_POS * (lateral_dist / 1000.0) 
        total_reward += pos_reward
        r_info['r/pos'] = pos_reward
        
        # Penalize Tilt Angle (Attitude shaping is always active)
        tilt_deg = state['tilt']
        tilt_reward = -self.W_TILT * (tilt_deg / 90.0)**2
        total_reward += tilt_reward
        r_info['r/tilt'] = tilt_reward


        # --- MODIFIED: Penalize Total Velocity only when low altitude ---
        vel_mag = np.linalg.norm(state['vel'])
        current_alt = state['alt']
        vel_reward = 0.0 # Initialize to 0

        if current_alt < self.SPEED_PENALTY_ALT_THRESHOLD:
            # Quadratic penalty encourages very low speeds
            vel_reward = -self.W_VEL * (vel_mag / 100.0)**2 
            
        total_reward += vel_reward
        r_info['r/vel'] = vel_reward

        # --- 3. Terminal Reward (Final penalties are always applied upon termination) ---
        if terminated or truncated:
            
            # Penalize any remaining lateral distance at termination
            final_pos_penalty = -self.W_POS * (lateral_dist) 
            total_reward += final_pos_penalty
            r_info['r/final_pos_penalty'] = final_pos_penalty
            
            # Penalize any remaining velocity at termination (This penalty is crucial and overrides the altitude logic)
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
            
        # Add logging info
        r_info['is_success'] = success
        r_info['lateral_dist'] = lateral_dist 
        r_info['alt'] = current_alt
        r_info['vel_mag'] = vel_mag
        r_info['fuel_remaining'] = state['fuel_mass']

        return total_reward, r_info