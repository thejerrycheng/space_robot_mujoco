import numpy as np
import time
import sys
from rocket_env.rocket_landing_env import RocketLandingEnv

# ================================================================
#   UTILITIES: COLORS & MATH
# ================================================================
class Col:
    RESET = '\033[0m'
    CYAN = '\033[96m'   # For Physics State
    YELLOW = '\033[93m' # For Controls
    GREEN = '\033[92m'  # For Success/Reward
    RED = '\033[91m'    # For Crash
    BOLD = '\033[1m'

def quat_to_euler(quat):
    """ Convert [w, x, y, z] to [roll, pitch, yaw] in degrees. """
    w, x, y, z = quat
    
    # Roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(np.array([roll, pitch, yaw]))

def randomize_initial_state(env):
    """ Applies randomization to the environment. """
    # Position
    env.data.qpos[env.qpos_adr : env.qpos_adr+3] = [
        np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(15, 30)
    ]
    # Orientation (Max 30 deg tilt)
    tilt = np.deg2rad(np.random.uniform(0, 30))
    axis = np.random.randn(3); axis[2]=0; axis/=np.linalg.norm(axis)
    env.data.qpos[env.qpos_adr+3 : env.qpos_adr+7] = [
        np.cos(tilt/2), axis[0]*np.sin(tilt/2), axis[1]*np.sin(tilt/2), 0
    ]
    # Velocity
    env.data.qvel[env.qvel_adr : env.qvel_adr+3] = [
        np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-5, -1)
    ]
    import mujoco
    mujoco.mj_forward(env.model, env.data)

# ================================================================
#   MAIN LOOP
# ================================================================
def test_env(random_actions=True, episodes=5):
    env = RocketLandingEnv(render_mode="human")

    for ep in range(episodes):
        print(f"\n{Col.BOLD}🚀 EPISODE {ep+1}/{episodes}{Col.RESET}")
        print("-" * 140)
        
        # Header for readability
        print(f"{'STEP':<5} | {Col.CYAN}{'STATE (Alt/Vel/Tilt/Mass)':<40}{Col.RESET} | "
              f"{Col.YELLOW}{'CONTROLS (Thrust/Gimbal)':<30}{Col.RESET} | {Col.GREEN}{'REWARD':<10}{Col.RESET}")

        env.reset()
        env.render()
        randomize_initial_state(env)
        env.render()

        done = False
        truncated = False
        step = 0

        while not (done or truncated):
            step += 1
            
            # --- 1. GET ACTION ---
            if random_actions:
                action = env.action_space.sample()
            else:
                action = np.zeros(3) # Hover

            # --- 2. STEP ---
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            
            # --- 3. EXTRACT DATA FOR LOGGING ---
            
            # Physics State
            pos = env.data.xpos[env.rocket_bid]
            vel = env.data.cvel[env.rocket_bid][3:]
            quat = env.data.qpos[env.qpos_adr+3 : env.qpos_adr+7]
            roll, pitch, yaw = quat_to_euler(quat)
            
            # Mass Calculation (Dry + Fuel)
            # Note: We access env.fuel_mass because that's where the changing value is stored
            current_mass = env.DRY_MASS + env.fuel_mass

            # Controls (Actual Actuator Outputs)
            thrust_N = env.data.ctrl[env.thrust_act]
            g_yaw    = np.degrees(env.data.ctrl[env.yaw_act])
            g_pit    = np.degrees(env.data.ctrl[env.pitch_act])

            # --- 4. FORMAT DASHBOARD STRING ---
            # Using fixed width {:X.Yf} to prevent jitter
            
            state_str = (
                f"Alt:{pos[2]:5.1f}m "
                f"Vz:{vel[2]:5.1f} "
                f"Tlt:{max(abs(pitch), abs(roll)):4.1f}° "
                f"Kg:{current_mass:5.1f}"
            )
            
            ctrl_str = (
                f"Thr:{thrust_N:6.0f}N "
                f"Gmb:{g_yaw:3.0f}/{g_pit:3.0f}"
            )

            # Assemble with Colors
            log_line = (
                f"\r{step:04}  | "
                f"{Col.CYAN}{state_str}{Col.RESET} | "
                f"{Col.YELLOW}{ctrl_str}{Col.RESET}     | "
                f"{Col.GREEN}{reward:6.1f}{Col.RESET} \033[K" # \033[K clears rest of line
            )

            sys.stdout.write(log_line)
            sys.stdout.flush()

            time.sleep(0.02) # Slow down to make it readable

        # Episode Result
        result_color = Col.GREEN if info.get('success') else Col.RED
        result_msg = "✅ SUCCESS" if info.get('success') else "❌ FAILURE"
        print(f"\n{result_color}>>> RESULT: {result_msg}{Col.RESET}")

    env.close()
    print("\n🎉 Test complete.\n")

if __name__ == "__main__":
    test_env(random_actions=True, episodes=10)