import os
import sys
import time
import argparse
import numpy as np
import subprocess
import csv
import importlib
import mujoco

# --- CRITICAL FIX FOR MACOS/LINUX RENDERING ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================================================
#   IMPORT ENVIRONMENT
# ================================================================
# Attempt to load the specific Polar Environment requested
try:
    mod = importlib.import_module("rocket_env.polar_rocket_env")
    RocketLandingEnv = getattr(mod, "RocketLandingEnv")
    print(f"✅ Successfully loaded RocketLandingEnv from 'rocket_env.polar_rocket_env'")
except ImportError as e:
    print(f"⚠️  Could not import 'rocket_env.polar_rocket_env': {e}")
    print("   Attempting fallback to local 'RocketLandingEnv.py'...")
    try:
        from RocketLandingEnv import RocketLandingEnv
        print("✅ Loaded local RocketLandingEnv.")
    except ImportError:
        print("❌ Error: Could not find RocketLandingEnv class anywhere.")
        sys.exit(1)

# ================================================================
#   UTILITIES
# ================================================================
class Col:
    RESET = '\033[0m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
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

def get_body_z_axis(quat):
    """ Calculates the Body Z-axis (Heading) vector from quaternion. """
    w, x, y, z = quat
    vx = 2 * (x*z + w*y)
    vy = 2 * (y*z - w*x)
    vz = 1 - 2 * (x*x + y*y)
    return np.array([vx, vy, vz])

def open_file(path):
    try:
        if sys.platform == "darwin": subprocess.call(["open", path])
        elif sys.platform == "win32": os.startfile(path)
        else: subprocess.call(["xdg-open", path])
    except: pass

# ================================================================
#   PLOTTING & SAVING
# ================================================================
def save_to_csv(history, episode_num, save_dir):
    filename = os.path.join(save_dir, f"episode_{episode_num}.csv")
    min_len = min(len(history[k]) for k in ['time', 'pos', 'vel', 'attitude', 'thrust', 'mass', 'reward'])
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Time", "X", "Y", "Z", "Vx", "Vy", "Vz", "Roll", "Pitch", "Yaw", "Thrust", "Mass", "Reward"])
        for i in range(min_len):
            p, v, a = history['pos'][i], history['vel'][i], history['attitude'][i]
            writer.writerow([
                i, history['time'][i], 
                p[0], p[1], p[2], v[0], v[1], v[2], a[0], a[1], a[2],
                history['thrust'][i], history['mass'][i], history['reward'][i]
            ])

def plot_static_analysis(history, episode_num, save_dir):
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    mass = np.array(history['mass'])
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Analysis - Episode {episode_num}", fontsize=16)

    # 1. Trajectory
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Traj', color='b')
    ax3d.scatter(0, 0, 0, c='k', marker='*', s=100, label='Target')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    # Force Aspect Ratio Hack
    max_range = np.array([pos[:,0].max()-pos[:,0].min(), pos[:,1].max()-pos[:,1].min(), pos[:,2].max()-pos[:,2].min()]).max() / 2.0
    mid_x, mid_y, mid_z = (pos[:,0].max()+pos[:,0].min())*0.5, (pos[:,1].max()+pos[:,1].min())*0.5, (pos[:,2].max()+pos[:,2].min())*0.5
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
    ax3d.legend()

    # 2. Altitude vs Radius
    ax_pos = fig.add_subplot(2, 3, 2)
    radius = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
    ax_pos.plot(times, pos[:, 2], label='Height (Z)', color='g')
    ax_pos.plot(times, radius, label='Radius (XY)', color='orange', linestyle='--')
    ax_pos.set_title("Position Metrics"); ax_pos.legend(); ax_pos.grid(True)

    # 3. Orientation
    ax_att = fig.add_subplot(2, 3, 3)
    ax_att.plot(times, att[:, 1], label='Pitch')
    ax_att.plot(times, att[:, 0], label='Roll')
    ax_att.set_title("Orientation"); ax_att.set_ylabel("Deg"); ax_att.legend(); ax_att.grid(True)

    # 4. Thrust
    ax_thr = fig.add_subplot(2, 3, 4)
    ax_thr.plot(times, thrust, color='r')
    ax_thr.set_title("Thrust (N)"); ax_thr.grid(True)

    # 5. Mass
    ax_mass = fig.add_subplot(2, 3, 5)
    ax_mass.plot(times, mass, color='k')
    ax_mass.set_title("Fuel Mass (kg)"); ax_mass.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"episode_{episode_num}_analysis.png")
    plt.savefig(save_path)
    plt.close(fig)

def generate_interactive_plot(all_histories, save_dir=".", env_name="Polar Env", termination_radius=100.0):
    try:
        import plotly.graph_objects as go
        import plotly.colors as pc
    except ImportError:
        print("Plotly not found.")
        return

    print(f"\n{Col.BOLD}📊 Generating Interactive Plotly HTML...{Col.RESET}")
    fig = go.Figure()
    palette = pc.qualitative.Plotly 
    all_x, all_y, all_z = [], [], []

    for i, history in enumerate(all_histories):
        pos = np.array(history['pos'])
        if len(pos) < 2: continue 
        all_x.extend(pos[:,0]); all_y.extend(pos[:,1]); all_z.extend(pos[:,2])
        
        quats = np.array(history.get('quat', []))
        color = palette[i % len(palette)]

        # Trajectory
        fig.add_trace(go.Scatter3d(
            x=pos[:,0], y=pos[:,1], z=pos[:,2],
            mode='lines', name=f'Ep {i+1}', line=dict(width=4, color=color), opacity=0.7
        ))

        # Heading Cones (Every 30 steps)
        if len(quats) == len(pos):
            step_int = 30
            idx = np.arange(0, len(pos), step_int)
            if len(idx) > 0:
                h = np.array([get_body_z_axis(q) for q in quats[idx]])
                fig.add_trace(go.Cone(
                    x=pos[idx,0], y=pos[idx,1], z=pos[idx,2],
                    u=h[:,0], v=h[:,1], w=h[:,2],
                    sizemode="scaled", sizeref=0.5, showscale=False, anchor="tail",
                    colorscale=[[0, color], [1, color]], name=f'Ep {i+1} Heading', hoverinfo='skip'
                ))

        # Start/End
        fig.add_trace(go.Scatter3d(x=[pos[0,0]], y=[pos[0,1]], z=[pos[0,2]], mode='markers', marker=dict(size=4, color=color), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[pos[-1,0]], y=[pos[-1,1]], z=[pos[-1,2]], mode='markers', marker=dict(size=6, color=color, symbol='x'), showlegend=False))

    # Context
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter3d(x=np.cos(theta), y=np.sin(theta), z=np.zeros_like(theta), mode='lines', line=dict(color='black', width=5), name='Pad (1m)'))
    fig.add_trace(go.Scatter3d(x=termination_radius*np.cos(theta), y=termination_radius*np.sin(theta), z=np.zeros_like(theta), mode='lines', line=dict(color='red', dash='dash'), name='Limit'))
    
    # 1:1 Scaling Calculation
    all_x = np.concatenate([all_x, [-termination_radius, termination_radius]])
    all_y = np.concatenate([all_y, [-termination_radius, termination_radius]])
    all_z = np.concatenate([all_z, [0, 20]])
    
    mx, my, mz = (np.max(all_x)+np.min(all_x))/2, (np.max(all_y)+np.min(all_y))/2, (np.max(all_z)+np.min(all_z))/2
    rng = max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)) / 2
    
    fig.update_layout(
        title=f"🚀 {env_name} Analysis (25 Trajectories)", width=1200, height=800,
        scene=dict(
            xaxis=dict(range=[mx-rng, mx+rng], backgroundcolor="white"),
            yaxis=dict(range=[my-rng, my+rng], backgroundcolor="white"),
            zaxis=dict(range=[mz-rng, mz+rng], backgroundcolor="#F0F0F0"),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    filename = os.path.join(save_dir, "interactive_polar_random_test.html")
    fig.write_html(filename)
    print(f"{Col.BOLD}🌎 Saved: {filename}{Col.RESET}")
    open_file(filename)

# ================================================================
#   MAIN TEST LOGIC
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=25, help="Number of trajectories to simulate")
    parser.add_argument("--no-render", action="store_true", help="Disable Mujoco rendering")
    parser.add_argument("--out-dir", type=str, default="test_polar_random", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Init Environment (Using the imported class from rocket_env.polar_rocket_env)
    env = RocketLandingEnv(render_mode="human" if not args.no_render else None)
    
    print(f"{Col.BOLD}🚀 Running {args.episodes} Randomized Episodes (No Policy)...{Col.RESET}")
    
    all_histories = []

    for ep in range(args.episodes):
        # --- POLAR RANDOMIZATION (Before Reset) ---
        # INIT_RADIUS: 12-20
        # INIT_HEIGHT: 8-14
        # INITIAL_SPEED: 0-7
        # INITIAL_ROLL_DEG: -20 to +20
        
        r_val = np.random.uniform(12.0, 20.0)
        h_val = np.random.uniform(8.0, 14.0)
        s_val = np.random.uniform(0.0, 7.0)
        roll_val = np.random.uniform(-20.0, 20.0)
        
        # Inject values into environment instance
        env.INIT_RADIUS = r_val
        env.INIT_HEIGHT = h_val
        env.INITIAL_SPEED = s_val
        env.INITIAL_ROLL_DEG = roll_val
        
        print(f"Ep {ep+1:02d}: {Col.YELLOW}R={r_val:4.1f}m, H={h_val:4.1f}m, Spd={s_val:3.1f}m/s, Roll={roll_val:5.1f}°{Col.RESET}", end=" | ")

        obs, _ = env.reset()
        
        done = False
        hist = {'time':[], 'pos':[], 'vel':[], 'attitude':[], 'quat':[], 'thrust':[], 'mass':[], 'reward':[]}
        step = 0
        
        while not done:
            # NO CONTROL: Passive free fall
            action = np.array([-1.0, 0.0, 0.0]) 
            
            obs, reward, term, trunc, info = env.step(action)
            
            # Rendering
            if not args.no_render:
                env.render()
                # Speed up rendering for batch processing
                if step % 2 == 0: time.sleep(0.001) 

            # Data Logging
            pos = env.data.xpos[env.rocket_bid].copy()
            vel = env.data.qvel[env.qvel_adr:env.qvel_adr+3].copy()
            quat = env.data.qpos[env.qpos_adr+3:env.qpos_adr+7].copy()
            att = quat_to_euler(quat)
            thr = env.data.ctrl[env.thrust_act]
            
            hist['time'].append(step * env.DT)
            hist['pos'].append(pos); hist['vel'].append(vel); hist['attitude'].append(att); hist['quat'].append(quat)
            hist['thrust'].append(thr); hist['mass'].append(env.fuel_mass + env.DRY_MASS); hist['reward'].append(reward)
            
            step += 1
            if term or trunc: done = True

        all_histories.append(hist)
        
        # Result
        final_dist = np.linalg.norm(hist['pos'][-1][:2])
        res = f"{Col.RED}CRASH{Col.RESET}" if hist['pos'][-1][2] <= 0.4 else f"{Col.YELLOW}TERM{Col.RESET}"
        print(f"Res: {res} | Dist: {final_dist:.1f}m")
        
        save_to_csv(hist, ep+1, args.out_dir)
        plot_static_analysis(hist, ep+1, args.out_dir)

    env.close()
    
    # Generate the big interactive plot
    term_radius = getattr(env, 'MAX_LATERAL_DIST', 100)
    generate_interactive_plot(all_histories, args.out_dir, env_name="Polar Rocket Env", termination_radius=term_radius)
    print(f"\n{Col.GREEN}✅ Completed {args.episodes} episodes.{Col.RESET}")

if __name__ == "__main__":
    main()