import numpy as np

def compute_reward(env, state, thrust_cmd, terminated, success):
    """
    Physics-Guided Reward Function: Ballistic Guidance.
    
    Reference Velocity Logic:
    Calculates the 'Required Velocity' (Lambert Solution) to coast ballistically 
    from the current position to the target (0,0,0) under gravity alone (no thrust).
    
    Time-of-Flight (T) Constraint:
    We use the free-fall time (T = sqrt(2h/g)) as the duration constraint. 
    This generates a reference trajectory that asks the rocket to align itself 
    such that gravity naturally pulls it into the target.
    """
    rewards = {}
    
    # --- CONFIGURATION ---
    # Gravity constant (Must match MuJoCo env, usually 1.62 for Moon)
    G_MOON = 1.62
    GRAVITY_VEC = np.array([0.0, 0.0, -G_MOON])
    
    # Weights
    w_track     = 2.0   # High priority: Match the ballistic trajectory
    w_upright   = 1.0   # Keep orientation upright
    w_fuel      = 0.05  # Minimize control effort
    
    # Terminal Rewards
    r_success   = 100.0
    r_semi      = 20.0
    r_crash     = -100.0
    r_drift     = -50.0
    
    # --- 1. EXTRACT STATE ---
    pos = state["pos"] # [x, y, z]
    vel = state["vel"] # [vx, vy, vz]
    dist_xy = state["dist_xy"]
    z = state["z"]
    
    # --- 2. CALCULATE BALLISTIC REFERENCE VELOCITY (Lambert Guidance) ---
    # We want to find v_ref such that: 0 = pos + v_ref*T + 0.5*g*T^2
    # Implies: v_ref = (-pos - 0.5*g*T^2) / T
    
    # A. Determine Time-of-Flight (T)
    # We use the time it would take to fall from this height as the natural cadence.
    # T = sqrt(2*h/g).
    # We clamp T_min to avoid division by zero near the ground.
    T_min = 1.0 
    if z > 0.1:
        T_fall = np.sqrt(2 * z / G_MOON)
        T = max(T_fall, T_min)
    else:
        T = T_min

    # B. Solve for Velocity Vector
    # Target position is (0,0,0), so delta_p = -pos
    delta_p = -pos 
    gravity_displacement = 0.5 * GRAVITY_VEC * (T ** 2)
    
    # v_ref = (Target - Current - Gravity_Effect) / Time
    v_ref = (delta_p - gravity_displacement) / T
    
    # C. Ground Proximity Override
    # If we are very close to the ground (< 5m), physics might demand weird things 
    # to hit 0,0,0 exactly (like high speed). Switch to soft landing mode.
    if z < 5.0:
        v_ref = np.array([0.0, 0.0, -1.0]) # Gentle vertical descent
    
    # --- 3. REWARD COMPONENTS ---
    
    # A. Velocity Tracking
    v_error = np.linalg.norm(vel - v_ref)
    # Exp kernel: Reward is 1.0 if error is 0, decays as error grows
    rewards["tracking"] = w_track * np.exp(-0.5 * v_error)
    
    # B. Orientation (Upright)
    # state['quat_w'] approaches 1.0 when upright
    rewards["upright"] = w_upright * (state["quat_w"] ** 2)
    
    # C. Fuel Penalty
    # Assuming thrust_cmd is normalized magnitude or raw force
    # We penalize usage to encourage efficient ballistic coasting
    norm_thrust = np.abs(thrust_cmd) / env.MAX_THRUST if hasattr(env, 'MAX_THRUST') else 0.0
    rewards["fuel"] = -w_fuel * norm_thrust
    
    # D. Alive Bonus
    rewards["alive"] = 0.05
    
    # E. Terminal Conditions
    rewards["terminal"] = 0.0
    if terminated:
        if success:
            rewards["terminal"] = r_success
        elif z < 0.4:
            # Ground Impact
            if dist_xy < 5.0:
                rewards["terminal"] = r_semi # Hit target, but crashed/tilted
            else:
                rewards["terminal"] = r_crash
        elif dist_xy > env.MAX_LATERAL_DIST:
            rewards["terminal"] = r_drift
            
    # --- 4. RETURN ---
    total_reward = sum(rewards.values())
    
    info = {
        "v_ref_error": v_error,
        "v_ref_mag": np.linalg.norm(v_ref),
        "calc_time_flight": T
    }
    
    return total_reward, info