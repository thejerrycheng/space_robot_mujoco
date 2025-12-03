import numpy as np

def compute_reward(env, m, thrust, terminated, success):
    """
    Staged Reward Function:
    Stage 1: Flip Upright (Ignores landing logic until upright)
    Stage 2: Land Softly (Activated only when upright)
    """
    rewards = {}
    
    # --- STAGE 1: ORIENTATION ---
    # This is the "Key" to unlock the rest of the rewards.
    # We always reward being upright, as it's the prerequisite for everything.
    # quat_w = 1.0 (upright), 0.7 (45 deg), 0.0 (90 deg)
    rewards["upright"] = 5.0 * (m["quat_w"] ** 2)

    # --- GATING LOGIC ---
    # If the rocket is not roughly upright (w < 0.8, approx 35 deg tilt),
    # we DO NOT reward approach or descent. This prevents the agent from 
    # trying to dive at the target head-first or sideways.
    landing_gate = 1.0 if m["quat_w"] > 0.8 else 0.0

    # --- STAGE 2: APPROACH & LANDING (Gated) ---
    
    # Distance Bonus (Only valid if we are upright-ish)
    # We want to be close to (0,0,0)
    dist_bonus = 1.0 / (1.0 + m["target_dist_3d"])
    rewards["dist_bonus"] = landing_gate * (3.0 * dist_bonus)

    # Descent Profile (Only valid if upright-ish)
    # Guidance to slow down near the ground
    desired_vz = -1.0 * max(m["z"] - env.LANDING_Z, 0.0)
    desired_vz = np.clip(desired_vz, -2.0, -0.1) # Slower descent target
    
    # We use a tighter exp curve for descent precision
    rewards["descent"] = landing_gate * (1.0 * np.exp(-2.0 * abs(m["vz"] - desired_vz)))

    # --- GLOBAL COSTS ---
    
    # Fuel Efficiency: Penalize high throttle usage
    norm_thrust = thrust / env.MAX_THRUST
    rewards["fuel"] = -0.05 * norm_thrust 

    # Velocity Constraint: Prevent supersonic suicide dives
    rewards["vel_pen"]  = -0.01 * m["vel_err"]
    
    # Survival Bonus: Keep flying!
    rewards["alive"] = 0.1

    # --- TERMINAL REWARDS ---
    rewards["terminal"] = 0.0
    
    if terminated:
        if success:
            rewards["terminal"] = 2000.0 # Massive success reward
            print(f"🌟 SUCCESS LANDING! Fuel Rem: {env.fuel_mass:.2f}")
        elif m["z"] < 0.1:  # Crash
            rewards["terminal"] = -100.0
        elif m["dist_xy"] > env.MAX_LATERAL_DIST:  # Drifted away
            rewards["terminal"] = -50.0
        elif m["vel_err"] > env.MAX_VELOCITY:  # Unstable
            rewards["terminal"] = -50.0

    total_reward = sum(rewards.values())
    
    return total_reward, rewards