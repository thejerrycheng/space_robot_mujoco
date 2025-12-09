import numpy as np

def compute_reward(env, m, thrust, terminated, success):
    """
    Implements the "Velocity Field" reward shaping from Gaudet et al. (2018).
    
    The core idea is to NOT reward position directly, but to calculate a 
    'Target Velocity' (v_targ) that points toward the landing site, and 
    reward the agent for matching that velocity vector.
    """
    rewards = {}
    
    # --- CONSTANTS FROM PAPER (Table 5) ---
    alpha = -0.01   # Velocity tracking penalty
    beta  = -0.05   # Control effort penalty
    gamma = -100.0  # Constraint violation penalty (soft)
    
    # 1. CALCULATE V_TARG (The Ideal Velocity)
    # The paper targets a point 15m above the ground (z=15)
    # to ensure a vertical descent for the final phase.
    
    # Relative position to the "Virtual Target" (0, 0, 15)
    # Note: m["pos"] is [x, y, z]. We want vector pointing TO target.
    # Target is at [0, 0, 15]. Vector FROM lander TO target is:
    r_to_targ = np.array([0.0, 0.0, 15.0]) - m["pos"]
    
    dist_to_targ = np.linalg.norm(r_to_targ)
    
    # Prevent division by zero
    vel_mag = m["vel_err"] + 1e-6
    
    # Time-to-Go estimate (Eq 29b)
    t_go = dist_to_targ / vel_mag
    
    # Parameters for the curve (Table 5)
    tau = 20.0 if m["z"] > 15.0 else 100.0
    v0  = 50.0  # Reference max velocity magnitude
    
    # Calculate Target Velocity Vector (Eq 29a)
    # Direction * Magnitude_Shaping
    direction = r_to_targ / (dist_to_targ + 1e-6)
    
    # This curve starts high and decays to 0 as t_go -> 0
    mag_curve = 1.0 - np.exp(-t_go / tau)
    
    v_targ = v0 * direction * mag_curve
    
    # If below 100m, simply target vertical descent at -2 m/s
    if m["z"] < 100:
        v_targ = np.array([0.0, 0.0, -2.0])

    # 2. VELOCITY TRACKING REWARD (The main driver)
    # Minimize difference between actual vel and ideal vel
    v_error = np.linalg.norm(m["vel"] - v_targ)
    rewards["tracking"] = alpha * v_error

    # 3. CONTROL EFFORT
    norm_thrust = thrust / env.MAX_THRUST
    # rewards["fuel"] = beta * norm_thrust

    # 4. ORIENTATION (Added for stability) - 'upright' bonus 
    rewards["upright"] = 1.0 * (m["quat_w"] ** 2)

    # 5. TERMINAL REWARDS (Kappa)
    rewards["terminal"] = 0.0
    
    if terminated:
        if success:
            rewards["terminal"] = 500.0  # Paper uses 10, but our scale is different
            print(f"🌟 SUCCESS! Fuel: {env.fuel_mass:.2f}")
        elif m["z"] < 0.1:
            rewards["terminal"] = -100.0 # Crash
        elif m["dist_xy"] > env.MAX_LATERAL_DIST:
            rewards["terminal"] = -50.0  # Drift
        elif m["vel_err"] > env.MAX_VELOCITY:
            rewards["terminal"] = -50.0  # Unstable

    # Sum
    total_reward = sum(rewards.values())
    
    # "Alive" bonus
    total_reward += 1
    
    return total_reward, rewards