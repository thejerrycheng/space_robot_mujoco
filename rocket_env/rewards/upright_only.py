import numpy as np

class UprightOnlyReward:
    def __init__(self, config):
        # We still need start_fuel for the final bonus, but the shaping is simple.
        self.start_fuel = config.get('start_fuel', 400_000.0) 
        
        # --- Weights ---
        self.w = {
            # Highest weight on Uprightness (Cosine of tilt angle)
            'upright_scale': 10.0,
            
            # Small penalty to encourage movement/progress
            'step_penalty': -0.01,
            
            # Terminal rewards remain standard
            'success_bonus': 1000.0,
            # 'crash_penalty': -50.0,
        }
        
        # Max tilt is 90 degrees for normalization
        self.MAX_TILT = 90.0

    def _calculate_upright_reward(self, state):
        """
        Calculates reward based on how close the rocket is to vertical (0 degrees tilt).
        Uses the cosine function, where cos(0 deg) = 1 (Max Reward)
        and cos(90 deg) = 0.
        """
        tilt_deg = state['tilt']
        tilt_rad = np.radians(tilt_deg)
        
        # Use cos(tilt) which is 1.0 when upright, falling to 0.0 at 90 degrees.
        upright_score = np.cos(tilt_rad)
        
        # Apply scaling. The reward is always positive (or zero) if the rocket is mostly upright.
        return self.w['upright_scale'] * upright_score

    def compute(self, state_dict, action, terminated, truncated, success):
        """
        Computes the total reward, prioritizing the upright position.
        """
        
        total_reward = 0.0
        r_info = {}
        
        # --- 1. Primary Shaping: Uprightness (Positive Reward) ---
        r_upright = self._calculate_upright_reward(state_dict)
        total_reward += r_upright
        
        # --- 2. Step Penalty (Time) ---
        # A small negative penalty ensures the agent doesn't stay still forever.
        total_reward += self.w['step_penalty']

        # --- 3. Terminal Reward ---
        if terminated:
            if success:
                # Big boost for successful landing
                total_reward += self.w['success_bonus']
                # Small bonus for fuel remaining
                total_reward += (state_dict['fuel_mass'] / self.start_fuel) * 5.0
                
            # elif self.
            # else:
            #     # Penalty for crashing (loss)
            #     total_reward += self.w['crash_penalty']
                
        r_info = {
            "r_upright": r_upright
        }
        
        return total_reward, r_info