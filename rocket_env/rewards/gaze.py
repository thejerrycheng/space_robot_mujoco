import numpy as np

class RocketReward:
    def __init__(self, config):
        self.start_fuel = config.get('start_fuel', 200_000.0)
        self.v0 = 50.0 # Ref velocity magnitude (approx initial descent speed)
        
        # --- Paper Hyperparameters (Table 5) ---
        self.params = {
            'alpha': -0.01,  # Velocity tracking error weight
            'beta':  -0.05,  # Control effort weight
            'delta': -20.0,  # Attitude margin weight
            'eta':    0.01,  # Survival bonus
            'kappa':  10.0,  # Success bonus
            'crash': -100.0, # Failure penalty (gamma in paper)
            
            # Constraints
            'q_lim': 0.2,       # ~11.4 degrees (Hard Limit)
            'q_mgn': 0.098,     # ~5.6 degrees (Margin start)
            'tau_1': 20.0,      # Time constant (High altitude)
            'tau_2': 100.0,     # Time constant (Low altitude)
            'h_switch': 15.0    # Altitude to switch guidance logic (meters)
        }

    def _get_target_velocity(self, state):
        """
        Calculates the Gaze Heuristic target velocity field.
        Paper Eq (29a - 29e)
        """
        # Current State
        r_pos = state['pos'] # [x, y, z]
        v_vel = state['vel'] # [vx, vy, vz]
        alt   = r_pos[2]
        
        # 1. Determine Target Point & Time Constant (Eq 29c, 29e)
        if alt > self.params['h_switch']:
            # High Altitude: Aim for a point 15m above ground
            target_pos = np.array([0.0, 0.0, self.params['h_switch']])
            tau = self.params['tau_1']
        else:
            # Low Altitude: Aim for ground, vertical descent
            target_pos = np.array([0.0, 0.0, -5.0]) # Target slightly below ground to force touchdown
            tau = self.params['tau_2']

        # 2. Vector to Target
        # relative position vector
        r_hat = target_pos - r_pos 
        dist = np.linalg.norm(r_hat)
        
        # Avoid division by zero
        if dist < 1e-3: 
            return np.zeros(3)

        # 3. Time-to-Go Estimate (Eq 29b)
        # t_go = range / speed
        speed = np.linalg.norm(v_vel)
        if speed < 0.1: speed = 0.1
        t_go = dist / speed

        # 4. Calculate Desired Velocity Vector (Eq 29a)
        # v_targ = -v0 * (direction) * (1 - exp(-t_go/tau))
        direction = r_hat / dist
        factor = 1.0 - np.exp(-t_go / tau)
        
        v_targ = direction * self.v0 * factor
        
        # If below switch altitude, kill horizontal velocity (force vertical)
        if alt <= self.params['h_switch']:
            v_targ[0] = 0.0 # x
            v_targ[1] = 0.0 # y
            # Eq 29d suggests specific vertical targets, 
            # but standard gaze heuristic works well here.
            
        return v_targ

    def compute(self, state, action, terminated, truncated, success):
        """
        Calculates reward based on Eq (30).
        """
        # --- 1. Compute Components ---
        
        # A. Velocity Tracking Error (The Gaze Heuristic)
        v_targ = self._get_target_velocity(state)
        v_error = state['vel'] - v_targ
        # Eq 30 Term 1: alpha * ||v - v_targ||
        reward_track = self.params['alpha'] * np.linalg.norm(v_error)

        # B. Control Effort
        # Eq 30 Term 2: beta * ||Force|| (We use action norm as proxy)
        reward_ctrl = self.params['beta'] * np.linalg.norm(action)

        # C. Attitude Margin (Shaping)
        # Eq 30 Term 4: delta * -max(0, q - q_margin)
        # Convert tilt degrees to radians for paper consistency
        tilt_rad = np.radians(state['tilt'])
        excess_tilt = max(0.0, tilt_rad - self.params['q_mgn'])
        reward_att = self.params['delta'] * excess_tilt

        # D. Survival Bonus
        # Eq 30 Term 5: eta
        reward_survive = self.params['eta']

        # --- 2. Sum Step Reward ---
        step_reward = reward_track + reward_ctrl + reward_att + reward_survive

        # --- 3. Terminal Logic ---
        if terminated:
            if success:
                # Success Bonus (Kappa)
                step_reward += self.params['kappa']
                # Fuel Bonus (Not in Eq 30 but helpful)
                step_reward += (state['fuel_mass'] / self.start_fuel) * 5.0
            else:
                # Crash Penalty
                # Paper handles this via 'gamma' term on constraint violation
                step_reward += self.params['crash']

        info = {
            "r_track": reward_track,
            "r_ctrl": reward_ctrl,
            "r_att": reward_att,
            "v_targ_z": v_targ[2] # Useful for debugging
        }
        
        return step_reward, info