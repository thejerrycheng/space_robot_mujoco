import numpy as np
import time
import sys
import mujoco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    mujoco.mj_forward(env.model, env.data)

def plot_all_trajectories(trajectories):
    """ Plots all collected trajectories in a 3D space. """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=colors[i], label=f'Episode {i+1}')
        
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position (Altitude)')
    ax.set_title('Rocket Trajectories')
    ax.legend()
    plt.show()

# ================================================================
#   MAIN LOOP
# ================================================================
def test_env(episodes=5):
    env = RocketLandingEnv(render_mode="human")
    all_trajectories = []

    for ep in range(episodes):
        print(f"\n{Col.BOLD}🚀 EPISODE {ep+1}/{episodes}{Col.RESET}")
        print("-" * 140)
        
        # Header for readability
        print(f"{'STEP':<5} | {Col.CYAN}{'STATE (Alt/Vel/Tilt/Mass)':<40}{Col.RESET} | "
              f"{Col.YELLOW}{'CONTROLS (Thrust/Gimbal)':<30}{Col.RESET} | {Col.GREEN}{'REWARD':<10}{Col.RESET}")

        env.reset()
        env.render()
        
        # Apply extra randomization if desired (optional)
        randomize_initial_state(env)
        env.render()

        done = False
        truncated = False
        step = 0
        episode_trajectory = []

        while not (done or truncated):
            step += 1
            
            # --- 1. NO ACTION (PASSIVE DROP TEST) ---
            # Action [-1, 0, 0] maps to 0% Thrust (Free fall)
            # Action [0, 0, 0] maps to 50% Thrust (Hover-ish)
            action = np.array([-1.0, 0.0, 0.0]) 

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
            current_mass = env.DRY_MASS + env.fuel_mass

            # Controls (Actual Actuator Outputs)
            thrust_N = env.data.ctrl[env.thrust_act]
            g_yaw    = np.degrees(env.data.ctrl[env.yaw_act])
            g_pit    = np.degrees(env.data.ctrl[env.pitch_act])

            # Collect trajectory data
            episode_trajectory.append(pos.copy())

            # --- 4. FORMAT DASHBOARD STRING ---
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
                f"{Col.GREEN}{reward:6.1f}{Col.RESET} \033[K"
            )

            sys.stdout.write(log_line)
            sys.stdout.flush()

            time.sleep(0.02) 

        all_trajectories.append(episode_trajectory)

        # Episode Result
        result_color = Col.GREEN if info.get('success') else Col.RED
        result_msg = "✅ SUCCESS" if info.get('success') else "❌ FAILURE"
        print(f"\n{result_color}>>> RESULT: {result_msg}{Col.RESET}")

    env.close()
    print("\n🎉 Test complete. Plotting trajectories...")
    plot_all_trajectories(all_trajectories)

if __name__ == "__main__":
    test_env(episodes=10)