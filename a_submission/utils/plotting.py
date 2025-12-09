import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plots
from scipy.spatial.transform import Rotation as R
import platform
import subprocess

# ================================================================
#   HELPER CLASSES & FUNCTIONS
# ================================================================

class Col:
    """ Simple terminal colors for logging """
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def get_body_z_axis(quat):
    """ 
    Calculates the body frame Z-axis vector from a quaternion [w, x, y, z].
    Used to visualize where the rocket is pointing.
    """
    # Scipy Rotation expects [x, y, z, w]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    # Apply rotation to the standard Z vector [0, 0, 1]
    return r.apply([0, 0, 1])

def open_file(path):
    """ Opens a file with the default system application. """
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin": # macOS
        subprocess.call(('open', path))
    else: # Linux
        subprocess.call(('xdg-open', path))

# ================================================================
#   PLOTTING FUNCTIONS
# ================================================================

def plot_static_analysis(history, episode_num, save_dir):
    """ 
    Generates static matplotlib dashboard for specific episode stats. 
    Saves a .png file to the save_dir.
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

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
    # Mark Start (Green) and End (Red)
    ax3d.scatter(pos[0,0], pos[0,1], pos[0,2], c='g', marker='o', label='Start')
    ax3d.scatter(pos[-1,0], pos[-1,1], pos[-1,2], c='r', marker='x', label='End')
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
    # Assuming att is [roll, pitch, yaw] or similar
    if att.shape[1] >= 3:
        ax_att.plot(times, att[:, 1], label='Pitch')
        ax_att.plot(times, att[:, 2], label='Yaw')
    else:
        # Fallback if just tilt magnitude
        ax_att.plot(times, att[:, 0], label='Tilt')
    ax_att.set_ylabel("Deg"); ax_att.legend(); ax_att.grid(True)

    # 4. Thrust
    ax_thr = fig.add_subplot(2, 3, 4)
    ax_thr.set_title("Thrust")
    ax_thr.plot(times, thrust, color='r')
    ax_thr.set_ylabel("N"); ax_thr.grid(True)

    # 5. Gimbal
    ax_gim = fig.add_subplot(2, 3, 5)
    ax_gim.set_title("Gimbal")
    if gimbal.shape[1] >= 2:
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
        
        # Get quaternions if available (added in main loop)
        quats = np.array(history.get('quat', []))

        # Extract data for hover tooltips
        steps = np.arange(len(pos))
        
        vel_z = np.array(history['vel'])[:, 2] 
        att = np.array(history['attitude'])
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
            opacity=0.75
        ))
        
        # 2. Heading Vectors (Cone Plot)
        # Check if we have quaternion data to calculate headings
        if len(quats) == len(pos):
            step_interval = 30 # Plot a cone every 30 steps
            indices = np.arange(0, len(pos), step_interval)
            
            if len(indices) > 0:
                sub_pos = pos[indices]
                sub_quats = quats[indices]
                
                # Calculate heading vectors
                headings = np.array([get_body_z_axis(q) for q in sub_quats])
                
                fig.add_trace(go.Cone(
                    x=sub_pos[:, 0], y=sub_pos[:, 1], z=sub_pos[:, 2],
                    u=headings[:, 0], v=headings[:, 1], w=headings[:, 2],
                    sizemode="scaled",
                    sizeref=1,
                    showscale=False,
                    anchor="tail",
                    colorscale=[[0, color], [1, color]],
                    name=f'Ep {i+1} Heading'
                ))
        
        # 3. Start Point
        fig.add_trace(go.Scatter3d(
            x=[pos[0,0]], y=[pos[0,1]], z=[pos[0,2]],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle'),
            showlegend=False, hoverinfo='skip'
        ))
        
        # 4. End Point
        fig.add_trace(go.Scatter3d(
            x=[pos[-1,0]], y=[pos[-1,1]], z=[pos[-1,2]],
            mode='markers',
            marker=dict(size=5, color=color, symbol='x'),
            showlegend=False, hoverinfo='skip'
        ))

    # --- ENVIRONMENT CONTEXT ---
    # Landing Pad (1m radius circle)
    theta = np.linspace(0, 2*np.pi, 50)
    r_pad = 20.0
    fig.add_trace(go.Scatter3d(
        x=r_pad * np.cos(theta), y=r_pad * np.sin(theta), z=np.zeros_like(theta),
        mode='lines',
        line=dict(color='black', width=10, dash='dash'),
        name='Landing Pad (20m)'
    ))
    
    # Target Zone (5m radius) for Semi-Success Visual
    r_zone = 100.0
    fig.add_trace(go.Scatter3d(
        x=r_zone * np.cos(theta), y=r_zone * np.sin(theta), z=np.zeros_like(theta),
        mode='lines',
        line=dict(color='orange', width=5, dash='dot'),
        name='Target Zone (100m)'
    ))
    
    # Center Point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='black', symbol='diamond'),
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
            aspectmode='data', 
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