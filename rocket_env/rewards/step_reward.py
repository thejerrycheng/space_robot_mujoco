import numpy as np

class RocketReward:
    def __init__(self, config):
        self.start_fuel = config.get('start_fuel', 1000.0)
        
        # --- WEIGHTS (The "Knobs" you can turn) ---
        self.w = {
            'upright':      2.0,   # High priority on orientation
            'dist':         1.0,   # Moving toward target
            'vel':          1.5,   # Slowing down
            'ang_vel':      0.5,   # Reducing spin
            'control':      0.05,  # Save fuel / smooth control
            'terminal':     100.0, # Big bonus for success
            'crash':       -50.0   # Penalty for crashing
        }

    def compute(self, state, action, terminated, truncated, success):
        """
        Returns: (total_reward, info_dict)
        """
        
        # 1. COMPUTE RAW COMPONENTS
        
        # --- A. Uprightness (Cosine Similarity) ---
        # tilt is in degrees. 0 is upright. 
        # Convert to a 0.0 to 1.0 score (1.0 = perfect upright)
        # We use a sharp curve: tilting a little is okay, tilting a lot is bad.
        tilt_rad = np.radians(state['tilt'])
        upright_score = np.cos(tilt_rad) # 1 at 0deg, 0 at 90deg, -1 at 180deg
        
        # Clip: If we are upside down, score is 0, not negative
        upright_score = max(0.0, upright_score)
        
        # Sharpen the curve: (x^4) makes it very sensitive to small tilts
        # 10 deg tilt -> 0.98^4 = 0.92 (Good)
        # 45 deg tilt -> 0.70^4 = 0.24 (Bad)
        upright_term = upright_score ** 4

        # --- B. Distance to Target (Normalized) ---
        # We want a value between 0 and 1.
        # Let's say max interesting distance is 1000m.
        dist = state['lateral_dist']
        dist_term = 1.0 / (1.0 + (dist / 100.0))

        # --- C. Velocity (Soft Landing) ---
        # We penalize velocity, but we penalize it MORE if we are close to the ground.
        velocity = np.linalg.norm(state['vel'])
        
        # Target velocity depends on altitude. 
        # High up: 100 m/s is fine. Low down: 5 m/s is required.
        target_v = 5.0 + (state['alt'] / 20.0) # Simple linear glide path
        
        vel_penalty = 0.0
        if velocity > target_v:
            # Penalize the excess velocity
            excess = velocity - target_v
            vel_penalty = -excess / 50.0 # Scaling factor

        # --- D. Angular Velocity (Spinning) ---
        # Hard to land if you are spinning like a top
        # state doesn't have ang_vel in dict, assuming it's passed or derived
        # For now, let's assume stable approach means tilt doesn't change rapidly
        # (If you need explicit ang_vel, add it to state_dict in env)
        ang_vel_penalty = 0.0 

        # --- E. Control Effort ---
        # Small penalty for blasting engines to prevent jittery policies
        control_penalty = -np.sum(np.abs(action)) / 3.0

        # 2. COMBINE (Shaped Reward)
        
        # GATING MECHANISM:
        # If the rocket is tilting > 45 degrees, IGNORE distance reward.
        # This forces the agent to fix orientation before trying to move.
        if state['tilt'] > 45.0:
            dist_term = 0.0
        
        step_reward = (
            self.w['upright'] * upright_term +
            self.w['dist']    * dist_term +
            self.w['vel']     * vel_penalty +
            self.w['control'] * control_penalty
        )

        # 3. TERMINAL REWARDS
        if terminated:
            if success:
                step_reward += self.w['terminal']
                # Bonus for fuel remaining?
                step_reward += (state['fuel_mass'] / self.start_fuel) * 20.0
            else:
                step_reward += self.w['crash']
                
                # Crash Shaping:
                # If you crashed, but you were upright and close, punish less.
                # If you crashed upside down, punish max.
                if state['tilt'] < 20.0:
                    step_reward += 20.0 # "Good try" bonus

        info = {
            "reward_upright": upright_term,
            "reward_dist": dist_term,
            "reward_vel": vel_penalty
        }
        
        return step_reward, info