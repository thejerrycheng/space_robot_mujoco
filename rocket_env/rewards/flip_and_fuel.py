import numpy as np

def compute_reward(env, m, thrust, terminated, success):
    """
    Args:
        env: The RocketLandingEnv instance
        m: Dict of state metrics (pos, vel, tilt, etc)
        thrust: Current thrust command in Newtons
        terminated: Boolean
        success: Boolean
    """
    rewards = {}
    
    # 1. ORIENTATION (Highest Priority)
    # We want quat_w -> 1.0 (Upright).
    # Weight: 2.0 * square creates a strong gradient near the top.
    rewards["upright"] = 2.0 * (m["quat_w"] ** 2)

    # 2. DISTANCE (Change from Penalty to BONUS)
    # OLD: -0.5 * dist (Caused suicide bug)
    # NEW: +1.0 / (1.0 + dist) (Rewards getting closer, bounded 0 to 1)
    # This ensures simply existing isn't painful.
    dist_bonus = 1.0 / (1.0 + m["target_dist_3d"])
    rewards["dist_bonus"] = 2.0 * dist_bonus

    # 3. SURVIVAL BONUS
    # Encourages the agent to stay alive longer (hover/control) rather than crash.
    rewards["alive"] = 0.1

    # 4. FUEL EFFICIENCY
    # Penalize based on % of MAX_THRUST used.
    norm_thrust = thrust / env.MAX_THRUST
    rewards["fuel"] = -0.05 * norm_thrust # Reduced slightly

    # 5. VELOCITY CONSTRAINT
    # Gentle penalty for moving too fast
    rewards["vel_pen"]  = -0.01 * m["vel_err"]

    # 6. DESCENT PROFILE
    # # Guidance to slow down near the ground
    # desired_vz = -1.0 * max(m["z"] - env.LANDING_Z, 0.0)
    # desired_vz = np.clip(desired_vz, -5.0, -0.5)
    # rewards["descent"] = 0.5 * np.exp(-1.0 * abs(m["vz"] - desired_vz))

    # --- B. Terminal Rewards (Sparse) ---
    rewards["terminal"] = 0.0
    
    if terminated:
        if success:
            rewards["terminal"] = 1000.0
            print(f"🌟 SUCCESS LANDING! Fuel Rem: {env.fuel_mass:.2f}")
        elif m["z"] < 0.4:  # Crash
            # Penalty is manageable, not so huge it dwarfs the shaping
            rewards["terminal"] = -100.0
        elif m["dist_xy"] > env.MAX_LATERAL_DIST:  # Drifted away
            rewards["terminal"] = -50.0
        elif m["vel_err"] > env.MAX_VELOCITY:  # Supersonic/Unstable
            rewards["terminal"] = -50.0

    total_reward = sum(rewards.values())
    
    return total_reward, rewards