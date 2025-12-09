import numpy as np

class RocketReward:
    def __init__(self, config=None):
        # --- CONSTANTS FROM PAPER TABLE 5 [cite: 500] ---
        # Alpha: Weight for velocity tracking error (The "Gaze Heuristic")
        self.alpha = -0.01 
        
        # Beta: Weight for control effort (Fuel usage)
        self.beta  = -0.05 
        
        # Gamma: Penalty for violating constraints (Attitude > limit)
        self.gamma = -100.0 
        
        # Delta: Penalty for approaching attitude limits (Margin penalty)
        self.delta = -20.0 
        
        # Eta: "Alive" bonus to encourage staying in the air / making progress
        self.eta   = 0.01 
        
        # Kappa: Terminal Success Bonus
        self.kappa = 10.0 

        # --- SHAPING PARAMETERS [cite: 467, 500] ---
        self.v0 = 50.0       # Reference velocity magnitude (approx starting vel)
        self.tau_1 = 20.0    # Time constant when > 15m
        self.tau_2 = 100.0   # Time constant when < 15m
        self.h_switch = 15.0 # Altitude to switch strategies (meters) [cite: 465]
        
        # Limits [cite: 491, 504]
        self.q_lim = 0.35    # Approx 20 deg (0.349 rad) hard limit
        self.q_mgn = 0.28    # Approx 16 deg (0.279 rad) soft margin start
        
    def compute(self, state, action, terminated, truncated, success):
        """
        Calculates reward based on Eq. 30 in the paper.
        Args:
            state (dict): Must contain 'pos', 'vel', 'tilt' (deg or rad), 'fuel_mass'
            action (np.array): The raw action vector
        """
        # 1. COMPUTE TARGET VELOCITY (The "Velocity Field")
        # [cite: 461-473]
        v_targ = self._compute_v_targ(state['pos'], state['vel'])
        
        # 2. COMPUTE COMPONENT REWARDS
        
        # A. Tracking Reward (Alpha) [cite: 479]
        # Minimize error between actual velocity and the "ideal" gaze velocity
        v_error = np.linalg.norm(state['vel'] - v_targ)
        r_track = self.alpha * v_error
        
        # B. Control Effort (Beta) [cite: 480]
        # Penalize high thrust usage. Assuming action is normalized [-1, 1]
        # We approximate ||F_B|| by the norm of the action.
        force_mag = np.linalg.norm(action)
        r_fuel = self.beta * force_mag
        
        # C. Attitude Margin (Delta) [cite: 481, 476]
        # Penalize if tilt gets close to the limit. 
        # The paper uses quaternion components; here we use 'tilt' angle (in radians)
        # We assume state['tilt'] is provided in degrees, convert to rads.
        tilt_rad = np.radians(state['tilt'])
        r_margin = 0.0
        if tilt_rad > self.q_mgn:
            # "max(0, q_i - q_mgn)" logic from Eq 30
            r_margin = self.delta * (tilt_rad - self.q_mgn)
            
        # D. Constraint Violation (Gamma) [cite: 481]
        # If we exceed limits, huge penalty.
        r_constraint = 0.0
        if tilt_rad > self.q_lim:
            r_constraint = self.gamma
            
        # E. Alive/Progress Bonus (Eta) 
        # "Constant positive term that encourages the agent to keep making progress"
        r_alive = self.eta
        
        # F. Terminal Bonus (Kappa) [cite: 489]
        r_terminal = 0.0
        if terminated:
            if success:
                # Paper uses 10.0, but scaling might be needed depending on your env
                r_terminal = self.kappa 
            # Note: The paper mentions "large negative reward" for crashing [cite: 219]
            # This is usually handled by the constraint violation (Gamma), 
            # but we can add a specific crash penalty if needed.
            elif state['alt'] < 0: 
                r_terminal = -10.0 # Crash penalty

        # 3. SUMMATION [cite: 476]
        total_reward = r_track + r_fuel + r_margin + r_constraint + r_alive + r_terminal
        
        return total_reward, {
            "r_track": r_track,
            "r_fuel": r_fuel,
            "r_margin": r_margin,
            "r_term": r_terminal,
            "v_error": v_error
        }

    def _compute_v_targ(self, pos, vel):
        """
        Implements the Velocity Field (Eq 29a - 29e)
        """
        x, y, z = pos
        
        # Relative position vector
        # The paper targets a point 15m above the ground (z=15)
        r_vec = np.array([x, y, z])
        v_mag = np.linalg.norm(vel) + 1e-6
        
        # SWITCHING LOGIC [cite: 462, 467]
        if z > self.h_switch:
            # --- PHASE 1: Above 15m ---
            # Target a point at [0, 0, 15]
            target_pos = np.array([0.0, 0.0, 15.0])
            r_hat = r_vec - target_pos # Vector pointing FROM target TO lander (approx)
            
            # Distance to that virtual target
            dist = np.linalg.norm(r_hat)
            
            # Time to go (Eq 29b)
            t_go = dist / v_mag
            
            # Direction vector (Unit vector pointing to target)
            # The paper defines r_hat slightly differently in Eq 29c, 
            # but the logic implies a vector pointing towards the origin.
            direction = -r_hat / (dist + 1e-6) 
            
            # Exponential decay (Eq 29a)
            decay = 1.0 - np.exp(-t_go / self.tau_1)
            
            # Resulting Target Velocity
            v_targ = self.v0 * direction * decay
            
            # Paper mentions target z-velocity of -2 m/s specifically 
            # We blend the vector field with this requirement.
            # However, Eq 29a suggests the field handles it all. 
            # We stick to the vector field for smoothness.
            
        else:
            # --- PHASE 2: Below 15m (Vertical Descent) ---
            # "Below 15 m, the downrange and crossrange velocity components... set to zero" 
            # "Target a z-component... equal to -2 m/s" 
            v_targ = np.array([0.0, 0.0, -2.0])
            
        return v_targ