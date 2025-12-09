import numpy as np

class Modular3DReward:
    def __init__(self, config):
        # Initial fuel mass is required for the fuel bonus calculation
        self.start_fuel = config.get('start_fuel', 400_000.0) 
        
        # --- Default Weights & Scales (Tune these based on training results) ---
        self.w = {
            # WEIGHTS for continuous penalty (applied to normalized state error)
            # Order: [Lateral Pos, Altitude, Velocity, Tilt, Angular Vel]
            'state_weights': np.array([3.0, 1.0, 4.0, 5.0, 1.0]),
            'state_scale': -1.0,        # Scales the overall continuous penalty

            # Terminal Rewards
            'success_bonus': 100.0,
            'crash_penalty': -50.0,
            
            # Step Penalty (Encourages faster solutions, similar to step_scale)
            'step_penalty': -0.05,
            
            # Heuristic penalty for fuel waste (optional)
            'thrust_penalty_scale': -0.001
        }
        
        # --- Success Criteria (Used to check for success, defined in your Env) ---
        # Note: These values are read from your Env's constraints
        self.POS_TOL = 10.0
        self.VEL_TOL = 5.0 
        self.TILT_TOL = 8.0
        
        # Maximum expected distance/velocity for normalizing penalties
        self.MAX_LAT_DIST = 500.0 
        self.MAX_VELOCITY = 100.0
        self.MAX_TILT = 90.0
        

    def _calculate_distance_penalty(self, state):
        """
        Calculates a weighted penalty based on the current state's deviation
        from the ideal target state (0 for all components).
        """
        
        # 1. Normalize State Errors (0 = perfect, 1 = max deviation)
        
        # Lateral Distance: Goal is 0m. Max is 500m (used for normalization)
        lat_dist_err = state['lateral_dist'] / self.MAX_LAT_DIST
        
        # Altitude: Goal is 0m. We don't penalize altitude *until* it's low,
        # so let's only penalize lateral distance during flight.
        # We set alt_err to 0 to prevent encouraging diving at the ground.
        alt_err = 0.0 

        # Velocity: Goal is 0 m/s. Max is 100 m/s (for normalization)
        vel_err = np.linalg.norm(state['vel']) / self.MAX_VELOCITY

        # Attitude (Tilt): Goal is 0 deg. Max is 90 deg.
        tilt_err = state['tilt'] / self.MAX_TILT
        
        # Angular Velocity (No data in state_dict, but we can proxy with tilt change)
        # Since state_dict doesn't contain ang_vel, we use tilt_err for high weight
        ang_vel_err = 0.0 
        
        # Combine errors into a single vector
        errors = np.array([
            lat_dist_err, 
            alt_err, 
            vel_err, 
            tilt_err, 
            ang_vel_err
        ])
        
        # 2. Apply Weights and Scale (Similar to the provided 2D logic)
        weighted_error = np.dot(errors, self.w['state_weights'])
        
        # Apply L2 norm approximation and scale
        penalty = self.w['state_scale'] * np.power(weighted_error, 0.5)
        
        return penalty
        
    def compute(self, state_dict, action, terminated, truncated, success):
        """
        Computes the total reward for the step.
        """
        
        total_reward = 0.0
        r_info = {} # To pass back debug rewards
        
        # --- 1. Continuous Shaping Reward (Distance Penalty) ---
        r_dist = self._calculate_distance_penalty(state_dict)
        total_reward += r_dist
        
        # --- 2. Step Penalty (Time/Fuel) ---
        # This replaces the 2D script's 'step_scale' * not_success
        total_reward += self.w['step_penalty']

        # --- 3. Terminal Reward ---
        if terminated:
            if success:
                # Success Reward
                total_reward += self.w['success_bonus']
                
                # Bonus for fuel remaining
                total_reward += (state_dict['fuel_mass'] / self.start_fuel) * 5.0
                
            else:
                # Crash Penalty
                # Note: We don't need a separate crash check like the 2D script 
                # because your Env only sets terminated=True if alt<50m. 
                # If success is False, it was a crash.
                total_reward += self.w['crash_penalty']
                
        r_info = {
            "r_dist_shaping": r_dist
        }
        
        return total_reward, r_info