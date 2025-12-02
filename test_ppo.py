import numpy as np
import time
import sys
import os
import argparse
import importlib
import mujoco

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
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees(np.array([roll, pitch, yaw]))

def open_file(filename):
    """ Cross-platform file opener helper. """
    import subprocess
    try:
        if sys.platform == "win32": os.startfile(filename)
        elif sys.platform == "darwin": subprocess.call(["open", filename])
        else: subprocess.call(["xdg-open", filename])
    except Exception as e:
        print(f"Could not open file automatically: {e}")

# ================================================================
#   INTERACTIVE PLOTTING (PLOTLY)
# ================================================================
def generate_interactive_plot(all_histories, save_dir=".", env_name="Rocket Env"):
    """ 
    Generates a high-fidelity interactive 3D plot using Plotly.
    Expects 'all_histories' to be a list of dicts with keys: 'pos', 'vel', 'angle'.
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

        # Extract data for hover tooltips
        steps = np.arange(len(pos))
        vel_z = np.array(history['vel'])[:, 2] 
        tilt = np.degrees(history['angle']) # Convert stored radians to degrees
        
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
            name=f'Ep {i+1}',
            text=hover_text,
            hoverinfo='text',
            line=dict(width=5, color=color),
            opacity=0.8
        ))
        
        # 2. Start Point
        fig.add_trace(go.Scatter3d(
            x=[pos[0,0]], y=[pos[0,1]], z=[pos[0,2]],
            mode='markers',
            marker=dict(size=4, color=color, symbol='circle'),
            showlegend=False, hoverinfo='skip'
        ))
        
        # 3. End Point
        fig.add_trace(go.Scatter3d(
            x=[pos[-1,0]], y=[pos[-1,1]], z=[pos[-1,2]],
            mode='markers',
            marker=dict(size=6, color=color, symbol='x'),
            showlegend=False, hoverinfo='skip'
        ))

    # --- ENVIRONMENT CONTEXT ---
    # Landing Pad (1m radius circle)
    theta = np.linspace(0, 2*np.pi, 50)
    r_pad = 1.0
    fig.add_trace(go.Scatter3d(
        x=r_pad * np.cos(theta), y=r_pad * np.sin(theta), z=np.zeros_like(theta),
        mode='lines',
        line=dict(color='black', width=4, dash='dash'),
        name='Landing Pad (1m)'
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

    filename = f"interactive_trajectories_{env_name}.html"
    fig.write_html(filename)
    
    print(f"{Col.BOLD}🌎 Plot Saved: {Col.CYAN}{filename}{Col.RESET}")
    open_file(filename)

# ================================================================
#   DYNAMIC ENV LOADER
# ================================================================
def get_env_class(env_name):
    """ Dynamically imports the RocketLandingEnv class. """
    env_map = {
        "default": "runs.rocket_landing_env",
        "env2":    "runs.rocket_landing_env_2",
        "env3":    "runs.rocket_landing_env_3",
        "simple":  "runs.rocket_landing_env_simple",
        "new":     "runs.rocket_landing_env_new",
    }

    if env_name not in env_map:
        print(f"{Col.RED}Error: Unknown environment '{env_name}'. Available: {list(env_map.keys())}{Col.RESET}")
        sys.exit(1)

    module_path = env_map[env_name]
    try:
        module = importlib.import_module(module_path)
        return getattr(module, "RocketLandingEnv")
    except ImportError as e:
        print(f"{Col.RED}Error importing {module_path}: {e}{Col.RESET}")
        sys.exit(1)
    except AttributeError:
        print(f"{Col.RED}Error: 'RocketLandingEnv' class not found in {module_path}{Col.RESET}")
        sys.exit(1)

def randomize_initial_state(env):
    """ Applies randomization to the environment. """
    # Position
    env.data.qpos[env.qpos_adr : env.qpos_adr+3] = [
        np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(10, 15)
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

# ================================================================
#   MAIN LOOP
# ================================================================
def test_env(env_name, episodes=5):
    # 1. Load the correct environment class
    EnvClass = get_env_class(env_name)
    env = EnvClass(render_mode="human")
    
    print(f"\n{Col.BOLD}🚀 Testing Environment: {env_name} ({EnvClass.__module__}){Col.RESET}")
    
    # Force Gravity Check
    env.model.opt.gravity[:] = [0, 0, -9.81]

    all_histories = []

    for ep in range(episodes):
        print(f"\n{Col.BOLD}▶ EPISODE {ep+1}/{episodes}{Col.RESET}")
        print("-" * 140)
        print(f"{'STEP':<5} | {Col.CYAN}{'STATE (Alt/Vel/Tilt/Mass)':<40}{Col.RESET} | "
              f"{Col.YELLOW}{'CONTROLS (Thrust/Gimbal)':<30}{Col.RESET} | {Col.GREEN}{'REWARD':<10}{Col.RESET}")

        env.reset()
        env.render()
        
        randomize_initial_state(env)
        env.render()

        done = False
        truncated = False
        step = 0
        
        # Store full history for Plotly (Pos, Vel, Angle)
        episode_history = {'pos': [], 'vel': [], 'angle': []}

        while not (done or truncated):
            step += 1
            
            # --- 1. NO ACTION (PASSIVE DROP TEST) ---
            action = np.array([-1.0, 0.0, 0.0]) # Free fall (0 thrust)

            # --- 2. STEP ---
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            
            # --- 3. DATA EXTRACTION ---
            pos = env.data.qpos[env.qpos_adr : env.qpos_adr+3].copy()
            vel = env.data.qvel[env.qvel_adr : env.qvel_adr+3].copy()
            quat = env.data.qpos[env.qpos_adr+3 : env.qpos_adr+7]
            roll, pitch, yaw = quat_to_euler(quat)
            
            # Collect for Plot
            episode_history['pos'].append(pos)
            episode_history['vel'].append(vel)
            episode_history['angle'].append(np.deg2rad(max(abs(pitch), abs(roll))))

            # Logging Vars
            dry_mass = getattr(env, 'DRY_MASS', 1000)
            fuel_mass = getattr(env, 'fuel_mass', 0)
            current_mass = dry_mass + fuel_mass
            
            thrust_N = env.data.ctrl[env.thrust_act]
            g_yaw    = np.degrees(env.data.ctrl[env.yaw_act])
            g_pit    = np.degrees(env.data.ctrl[env.pitch_act])

            # Format String
            state_str = (
                f"Alt:{pos[2]:5.1f}m "
                f"Vz:{vel[2]:5.1f} "
                f"Tlt:{max(abs(pitch), abs(roll)):4.1f}° "
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
            time.sleep(0.01) 

        all_histories.append(episode_history)

        # Episode Result
        result_color = Col.GREEN if info.get('success') else Col.RED
        result_msg = "✅ SUCCESS" if info.get('success') else "❌ FAILURE"
        print(f"\n{result_color}>>> RESULT: {result_msg}{Col.RESET}")

    env.close()
    
    # Call the new Plotly function
    generate_interactive_plot(all_histories, env_name=env_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rocket Landing Environment")
    parser.add_argument("--env", type=str, default="default", 
                        choices=["default", "env2", "env3", "simple", "new"],
                        help="Which environment file to load")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    test_env(args.env, args.episodes)