import numpy as np

class RocketReward:
    def __init__(self, config):
        # Configuration and Weights
        self.START_FUEL = config.get('start_fuel', 200_000.0) 
        
        # Shaping Weights (Continuous)
        self.W_UPRIGHT  = 10.0  
        self.W_VELOCITY = -0.1   
        self.W_DISTANCE = -0.05  
        self.W_STEP     = -0.01  

        # Terminal Weights (Sparse)
        self.W_SUCCESS  = 1000.0 
        self.W_FUEL     = 5.0    

                # Penalty 
        self.P_CRASH = -100.0 
        self.P_BOUNDARY_VIOLATION = -500.0

        # Bonus 
        self.B_CRASH_TARGET_ZONE = 10.0  

    # ==========================================================
    # SHAPING REWARD FUNCTIONS
    # ==========================================================

    def _calculate_upright_reward(self, state):
        """Rewards vertical alignment (0 degrees tilt)."""
        tilt_deg = state['tilt']
        tilt_rad = np.radians(tilt_deg)
        return self.W_UPRIGHT * np.cos(tilt_rad)

    def _calculate_velocity_reward(self, state):
        """Penalizes the magnitude of the velocity vector (speed) in the Z (altitude) direction only."""
        vel_z = state['vel'][2]
        return self.W_VELOCITY * vel_z

    def _calculate_distance_reward(self, state):
        """Penalizes the lateral distance from the target (0,0)."""
        lateral_dist = state['lateral_dist']
        return self.W_DISTANCE * lateral_dist
    
    def _calculate_step_reward(self):
        """Penalizes elapsed time."""
        return self.W_STEP

    # ==========================================================
    # TERMINAL REWARD FUNCTION (NEW)
    # ==========================================================

    def _calculate_terminal_reward(self, state, terminated, success):
        """
        Calculates the large, sparse reward upon successful termination,
        and applies penalties for various failure modes.
        """
        r_terminal = 0.0
        
        # --- Check for Success ---
        if terminated and success:
            # 1. Success Bonus
            r_terminal += self.W_SUCCESS
            
            # 2. Fuel Bonus
            fuel_mass = state['fuel_mass']
            fuel_ratio = fuel_mass / self.START_FUEL
            r_terminal += self.W_FUEL * fuel_ratio
            
        # --- Check for Failure Penalties & Crash Bonus ---
        elif terminated and not success:
            
            # Flag for whether the termination was due to boundary violation (Tilt/Speed/Distance)
            boundary_violation = False

            # A. Tilt Over 100 Degrees (Catastrophic Failure)
            if abs(state['tilt']) > 100.0:
                r_terminal += self.P_BOUNDARY_VIOLATION
                boundary_violation = True
                # print(f"💰 Terminal Reward: Excessive Tilt Penalty ({self.P_BOUNDARY_VIOLATION:.2f})")
                
            # B. Max Lateral Distance Exceeded (Catastrophic Failure)
            if state['lateral_dist'] > 700.0:
                r_terminal += self.P_BOUNDARY_VIOLATION
                boundary_violation = True
                # print(f"💰 Terminal Reward: Max Distance Penalty ({self.P_BOUNDARY_VIOLATION:.2f})")
            
            # C. Max Speed Exceeded (Catastrophic Failure)
            if np.linalg.norm(state['vel']) > 500.0:
                r_terminal += self.P_BOUNDARY_VIOLATION
                boundary_violation = True
                # print(f"💰 Terminal Reward: Max Speed Penalty ({self.P_BOUNDARY_VIOLATION:.2f})")

            # ----------------------------------------------------
            # D. Crash Landing Penalty (Not due to boundary violation)
            if not boundary_violation and \
               state['alt'] < 60.0:
                r_terminal += self.P_CRASH
                # print(f"💰 Terminal Reward: Base Crash Penalty ({self.P_CRASH:.2f})")
                # --- NEW: Small Bonus for Crashing in the Target Area ---
                if state['lateral_dist'] < 100.0:
                    r_terminal += self.B_CRASH_TARGET_ZONE
                    print(f"💰 Terminal Reward: Crash within Target Zone Bonus ({self.B_CRASH_TARGET_ZONE:.2f})")
                # ----------------------------------------------------
            
        return r_terminal 

    def compute(self, state_dict, action, terminated, truncated, success):
        
        # --- 1. Shaping Rewards (Continuous) ---
        r_upright  = self._calculate_upright_reward(state_dict)
        r_velocity = self._calculate_velocity_reward(state_dict)
        r_distance = self._calculate_distance_reward(state_dict)
        r_step     = self._calculate_step_reward()
        
        r_shaping = r_velocity + r_step

        # --- 2. Terminal Reward (Sparse) ---
        r_terminal = self._calculate_terminal_reward(state_dict, terminated, success)
        
        total_reward = r_shaping + r_terminal

        # Info for logging
        r_info = {
            "r_upright": r_upright,
            "r_velocity": r_velocity,
            "r_distance": r_distance,
            "r_step": r_step,
            "terminal_reward": r_terminal
        }
        
        return total_reward, r_info