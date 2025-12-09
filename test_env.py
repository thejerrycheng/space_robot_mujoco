import numpy as np
import time
import sys
import os
import argparse
import importlib
import mujoco
import csv

# --- CRITICAL FIX FOR MACOS CRASH ---
# We must set the backend to 'Agg' before importing pyplot.
# This prevents Matplotlib from trying to open a window on a non-main thread
# or conflicting with the MuJoCo viewer, which causes NSWindow errors.
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

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
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(np.array([roll, pitch, yaw]))

def get_body_z_axis(quat):
    """ 
    Calculates the Body Z-axis vector in World Frame given a quaternion [w,x,y,z].
    This typically corresponds to the rocket's longitudinal/thrust axis.
    """
    w, x, y, z = quat
    # Formula for the Z column of the Rotation Matrix from Quaternion
    vx = 2 * (x*z + w*y)
    vy = 2 * (y*z - w*x)
    vz = 1 - 2 * (x*x + y*y)
    return np.array([vx, vy, vz])

def open_file(path):
    """ Cross-platform file opener helper. """
    import subprocess
    try:
        if sys.platform == "win32": os.startfile(path)
        elif sys.platform == "darwin": subprocess.call(["open", path])
        else: subprocess.call(["xdg-open", path])
    except Exception as e:
        print(f"Could not open file automatically: {e}")

def save_to_csv(history, episode_num, save_dir):
    """ Saves episode trajectory data to a CSV file. """
    filename = os.path.join(save_dir, f"episode_{episode_num}.csv")
    
    # Ensure all arrays are same length
    min_len = min(len(history[k]) for k in ['time', 'pos', 'vel', 'attitude', 'thrust', 'gimbal', 'mass', 'reward'])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow([
            "Step", "Time", "X", "Y", "Z", "Vx", "Vy", "Vz",
            "Roll", "Pitch", "Yaw", 
            "Thrust", "GimbalYaw", "GimbalPitch", 
            "Mass", "Reward"
        ])
        
        # Rows
        for i in range(min_len):
            pos = history['pos'][i]
            vel = history['vel'][i]
            att = history['attitude'][i]
            gim = history['gimbal'][i]
            
            writer.writerow([
                i, history['time'][i], 
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                att[0], att[1], att[2],
                history['thrust'][i], gim[0], gim[1],
                history['mass'][i], history['reward'][i]
            ])
            
    print(f"💾 Data saved to: {filename}")

# ================================================================
#   PLOTTING FUNCTIONS
# ================================================================
def plot_static_analysis(history, episode_num, save_dir):
    """ Generates static matplotlib dashboard for specific episode stats. """
    # No local import of plt here; rely on the global Agg-configured plt
    
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Analysis - Episode {episode_num}", fontsize=16)

    # 1. 3D Trajectory (Static)
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.set_title("3D Trajectory")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b')
    ax3d.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.legend()

    # 2. Position
    ax_pos = fig.add_subplot(2, 3, 2)
    ax_pos.set_title("Position")
    ax_pos.plot(times, pos[:, 2], label='Z (Alt)', color='green')
    ax_pos.plot(times, np.sqrt(pos[:,0]**2 + pos[:,1]**2), label='Lateral Error', color='orange', linestyle='--')
    ax_pos.set_ylabel("m"); ax_pos.legend(); ax_pos.grid(True)

    # 3. Orientation
    ax_att = fig.add_subplot(2, 3, 3)
    ax_att.set_title("Orientation (Tilt)")
    ax_att.plot(times, att[:, 1], label='Pitch')
    ax_att.plot(times, att[:, 2], label='Yaw')
    ax_att.set_ylabel("Deg"); ax_att.legend(); ax_att.grid(True)

    # 4. Thrust
    ax_thr = fig.add_subplot(2, 3, 4)
    ax_thr.set_title("Thrust")
    ax_thr.plot(times, thrust, color='r')
    ax_thr.set_ylabel("N"); ax_thr.grid(True)

    # 5. Gimbal
    ax_gim = fig.add_subplot(2, 3, 5)
    ax_gim.set_title("Gimbal")
    ax_gim.plot(times, gimbal[:, 0], label='Yaw')
    ax_gim.plot(times, gimbal[:, 1], label='Pitch')
    ax_gim.set_ylabel("Deg"); ax_gim.legend(); ax_gim.grid(True)

    # 6. Mass
    ax_mass = fig.add_subplot(2, 3, 6)
    ax_mass.set_title("Mass")
    ax_mass.plot(times, mass, color='black')
    ax_mass.set_ylabel("kg"); ax_mass.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"episode_{episode_num}_analysis.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"📊 Static Plot saved to: {save_path}")

def generate_interactive_plot(all_histories, save_dir=".", env_name="Rocket Env"):
    """ 
    Generates a high-fidelity interactive 3D plot using Plotly.
    Expects 'all_histories' to be a list of dicts.
    """
    try:
        import plotly.graph_objects as go
        import plotly.colors as pc
    except ImportError:
        print(f"\n{Col.YELLOW}⚠️  'plotly' not found. Skipping interactive 3D plot.")
        print(f"   Run 'pip install plotly' to enable this feature.{Col.RESET}")
        return

    print(f"\n{Col.BOLD}📊 Generating Interactive Plotly HTML...{Col.RESET}")
    fig = go.Figure()
    palette = pc.qualitative.Plotly 

    for i, history in enumerate(all_histories):
        pos = np.array(history['pos'])
        if len(pos) < 2: continue 

        # Get quaternions for heading calc
        quats = np.array(history.get('quat', []))

        # Extract data for hover tooltips
        steps = np.arange(len(pos))
        vel_z = np.array(history['vel'])[:, 2] 
        att = np.array(history['attitude'])
        # Max tilt (approx)
        tilt = np.maximum(np.abs(att[:,0]), np.abs(att[:,1]))
        
        # Create Hover Text
        hover_text = [
            f"Step: {s}<br>Alt: {z:.2f}m<br>Vz: {v:.2f}m/s<br>Tilt: {t:.1f}°"
            for s, z, v, t in zip(steps, pos[:,2], vel_z, tilt)
        ]

        color = palette[i % len(palette)]

        # 1. Trajectory Line
        fig.add_trace(go.Scatter3d(
            x=pos[:,0], y=pos[:,1], z=pos[:,2],
            mode='lines',
            name=f'Ep {i+1} Traj',
            text=hover_text,
            hoverinfo='text',
            line=dict(width=5, color=color),
            opacity=0.8
        ))
        
        # 2. Heading Vectors (Cone Plot)
        # Check if we have quaternion data to calculate headings
        if len(quats) == len(pos):
            step_interval = 20 # Plot a cone every 20 steps to avoid clutter
            indices = np.arange(0, len(pos), step_interval)
            
            if len(indices) > 0:
                sub_pos = pos[indices]
                sub_quats = quats[indices]
                
                # Calculate heading vectors (Rocket Z-axis)
                headings = np.array([get_body_z_axis(q) for q in sub_quats])
                
                # Add Cones
                fig.add_trace(go.Cone(
                    x=sub_pos[:, 0], y=sub_pos[:, 1], z=sub_pos[:, 2],
                    u=headings[:, 0], v=headings[:, 1], w=headings[:, 2],
                    sizemode="scaled",
                    sizeref=0.5, # Adjust size of cones
                    showscale=False,
                    anchor="tail", # Cone tail at the position
                    colorscale=[[0, color], [1, color]],
                    name=f'Ep {i+1} Heading',
                    hoverinfo='skip'
                ))
        
        # 3. Start Point
        fig.add_trace(go.Scatter3d(
            x=[pos[0,0]], y=[pos[0,1]], z=[pos[0,2]],
            mode='markers',
            marker=dict(size=4, color=color, symbol='circle'),
            showlegend=False, hoverinfo='skip'
        ))
        
        # 4. End Point
        fig.add_trace(go.Scatter3d(
            x=[pos[-1,0]], y=[pos[-1,1]], z=[pos[-1,2]],
            mode='markers',
            marker=dict(size=6, color=color, symbol='x'),
            showlegend=False, hoverinfo='skip'
        ))

    # --- ENVIRONMENT CONTEXT ---
    # Landing Pad (1m radius circle)
    theta = np.linspace(0, 2*np.pi, 50)
    r_pad = 1
    fig.add_trace(go.Scatter3d(
        x=r_pad * np.cos(theta), y=r_pad * np.sin(theta), z=np.zeros_like(theta),
        mode='lines',
        line=dict(color='black', width=4, dash='dash'),
        name='Landing Pad (1m)'
    ))
    
    # Target Zone (20m radius)
    r_zone = 20.0
    fig.add_trace(go.Scatter3d(
        x=r_zone * np.cos(theta), y=r_zone * np.sin(theta), z=np.zeros_like(theta),
        mode='lines',
        line=dict(color='orange', width=2, dash='dot'),
        name='Target Zone (20m)'
    ))
    
    # Center Point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=5, color='black', symbol='diamond'),
        name='Target'
    ))

    # Layout Config
    fig.update_layout(
        title=f"🚀 {env_name} Trajectory Analysis",
        width=1200, height=800,
        scene=dict(
            xaxis_title='X (Lateral)',
            yaxis_title='Y (Lateral)',
            zaxis_title='Z (Altitude)',
            aspectmode='data', # CRITICAL: Keeps physics scale 1:1
            xaxis=dict(gridcolor='lightgrey', backgroundcolor="white"),
            yaxis=dict(gridcolor='lightgrey', backgroundcolor="white"),
            zaxis=dict(gridcolor='grey', backgroundcolor="#F0F0F0"),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05)
    )

    filename = os.path.join(save_dir, f"interactive_trajectories_{env_name}.html")
    fig.write_html(filename)
    
    print(f"{Col.BOLD}🌎 Plot Saved: {Col.CYAN}{filename}{Col.RESET}")
    open_file(filename)

# ================================================================
#   DYNAMIC ENV LOADER
# ================================================================
def get_env_class(env_name):
    """ Dynamically imports the RocketLandingEnv class. """
    env_map = {
        "default": "rocket_env.rocket_landing_env",
        "env2":    "rocket_env.rocket_landing_env_2",
        "env3":    "rocket_env.rocket_landing_env_3",
        "simple":  "rocket_env.rocket_landing_env_simple",
        "new":     "rocket_env.rocket_landing_env_new",
        "real":     "rocket_env.rocket_realistic_env",
        "default2": "rocket_env.rocket_2_env",
        "polar": "rocket_env.polar_rocket_env",
        "default3": "rocket_env.rocket_3_env",
        # Assuming your file is named RocketLandingEnv.py and in root or rocket_env
        "current": "RocketLandingEnv" 
    }
    
    # Try direct import if not in map
    module_path = env_map.get(env_name, env_name)

    try:
        module = importlib.import_module(module_path)
        return getattr(module, "RocketLandingEnv")
    except ImportError as e:
        # Fallback to current directory for "RocketLandingEnv"
        try:
             module = importlib.import_module("RocketLandingEnv")
             return getattr(module, "RocketLandingEnv")
        except:
            print(f"{Col.RED}Error importing {module_path}: {e}{Col.RESET}")
            sys.exit(1)
    except AttributeError:
        print(f"{Col.RED}Error: 'RocketLandingEnv' class not found in {module_path}{Col.RESET}")
        sys.exit(1)

def randomize_initial_state(env):
    """ Applies randomization to the environment. """
    env.reset() # Ensure base reset first
    # Example logic (adjust based on your env structure)
    # env.data.qpos[...] = ...
    mujoco.mj_forward(env.model, env.data)

# ================================================================
#   MAIN LOOP
# ================================================================
def test_env(env_name, episodes=5):
    # 1. Load the correct environment class
    EnvClass = get_env_class(env_name)
    env = EnvClass(render_mode="human")
    
    print(f"\n{Col.BOLD}🚀 Testing Environment: {env_name} ({EnvClass.__module__}){Col.RESET}")
    
    # Force Gravity Check
    if hasattr(env.model.opt, 'gravity'):
        env.model.opt.gravity[:] = [0, 0, -1.62]

    all_histories = []

    for ep in range(episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{episodes}{Col.RESET}")
        print("-" * 140)
        print(f"{'STEP':<5} | {Col.CYAN}{'STATE (Alt/Vel/Tilt/Mass)':<40}{Col.RESET} | "
              f"{Col.YELLOW}{'CONTROLS (Thrust/Gimbal)':<30}{Col.RESET} | {Col.GREEN}{'REWARD':<10}{Col.RESET}")

        obs, _ = env.reset()
        env.render()
        
        # randomize_initial_state(env) # Uncomment if needed
        env.render()

        done = False
        truncated = False
        step = 0
        
        # Store full history for Plotly and CSV
        episode_history = {
            'time': [], 'pos': [], 'vel': [], 'angle': [], 'quat': [],
            'attitude': [], 'thrust': [], 'gimbal': [], 'mass': [], 'reward': []
        }

        while not (done or truncated):
            step += 1
            
            # --- 1. ACTION (Passive or Random) ---
            # action = np.array([-1.0, 0.0, 0.0]) # Free fall (0 thrust)
            action = env.action_space.sample() * 0.1 + np.array([-1, 0, 0]) # Low thrust random

            # --- 2. STEP ---
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            
            # --- 3. DATA EXTRACTION ---
            pos = env.data.qpos[env.qpos_adr : env.qpos_adr+3].copy()
            vel = env.data.qvel[env.qvel_adr : env.qvel_adr+3].copy()
            quat = env.data.qpos[env.qpos_adr+3 : env.qpos_adr+7].copy()
            
            # Calculate Euler Angles
            euler_deg = quat_to_euler(quat) # [roll, pitch, yaw]
            tilt_deg = max(abs(euler_deg[0]), abs(euler_deg[1]))

            # Collect for Plot/CSV
            episode_history['time'].append(step * env.model.opt.timestep)
            episode_history['pos'].append(pos)
            episode_history['vel'].append(vel)
            episode_history['quat'].append(quat)
            episode_history['attitude'].append(euler_deg)
            episode_history['angle'].append(np.deg2rad(tilt_deg)) # Keep legacy format for Plotly
            episode_history['reward'].append(reward)

            # Logging Vars
            dry_mass = getattr(env, 'DRY_MASS', 100)
            fuel_mass = getattr(env, 'fuel_mass', 0)
            current_mass = dry_mass + fuel_mass
            episode_history['mass'].append(current_mass)
            
            # Controls (Reverse engineer from action or take directly if available)
            # Assuming env.data.ctrl indices match thrust/yaw/pitch actuators
            # Note: This depends on actuator order in XML
            try:
                thrust_N = env.data.ctrl[env.thrust_act]
                g_yaw    = np.degrees(env.data.ctrl[env.yaw_act])
                g_pit    = np.degrees(env.data.ctrl[env.pitch_act])
            except:
                thrust_N, g_yaw, g_pit = 0, 0, 0
                
            episode_history['thrust'].append(thrust_N)
            episode_history['gimbal'].append([g_yaw, g_pit])

            # Format String
            state_str = (
                f"Alt:{pos[2]:5.1f}m "
                f"Vz:{vel[2]:5.1f} "
                f"Tlt:{tilt_deg:4.1f}° "
                f"Kg:{current_mass:5.1f}"
            )
            ctrl_str = f"Thr:{thrust_N:6.0f}N Gmb:{g_yaw:3.0f}/{g_pit:3.0f}"

            log_line = (
                f"\r{step:04}  | "
                f"{Col.CYAN}{state_str}{Col.RESET} | "
                f"{Col.YELLOW}{ctrl_str}{Col.RESET}     | "
                f"{Col.GREEN}{reward:6.1f}{Col.RESET} \033[K"
            )

            sys.stdout.write(log_line)
            sys.stdout.flush()
            time.sleep(0.005) # Speed up slightly

        all_histories.append(episode_history)

        # Episode Result
        result_color = Col.GREEN if info.get('success') else Col.RED
        result_msg = "✅ SUCCESS" if info.get('success') else "❌ FAILURE"
        print(f"\n{result_color}>>> RESULT: {result_msg}{Col.RESET}")
        
        # Save per-episode CSV and Static Plot
        save_to_csv(episode_history, ep+1, ".")
        plot_static_analysis(episode_history, ep+1, ".")

    env.close()
    
    # Call the Interactive Plotly function
    generate_interactive_plot(all_histories, env_name=env_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rocket Landing Environment")
    parser.add_argument("--env", type=str, default="current", 
                        help="Which environment file to load (module name)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    test_env(args.env, args.episodes)