import numpy as np
import plotly.graph_objects as go

def calculate_target_velocity(pos, current_vel_mag):
    """
    Replicates the logic from the user's compute_reward snippet 
    and the Gaudet (2018) paper.
    """
    # 1. SETUP
    # Target is virtual point at 15m (Paper Eq 29c)
    target_pos = np.array([0.0, 0.0, 15.0]) 
    
    # Vector FROM lander TO target
    r_to_targ = target_pos - pos
    dist_to_targ = np.linalg.norm(r_to_targ)
    
    # Constants from Paper/Snippet
    tau = 20.0 if pos[2] > 15.0 else 100.0
    v0 = 50.0  # Max reference velocity
    
    # 2. CALCULATE Time-to-Go (t_go)
    # Note: We use current_vel_mag to estimate t_go
    # Safe divide
    vel_mag_safe = max(current_vel_mag, 1e-3)
    t_go = dist_to_targ / vel_mag_safe
    
    # 3. COMPUTE V_TARG
    # Direction
    if dist_to_targ > 1e-6:
        direction = r_to_targ / dist_to_targ
    else:
        direction = np.array([0, 0, -1]) # Down
        
    # Magnitude Curve (Eq 29a)
    mag_curve = 1.0 - np.exp(-t_go / tau)
    
    v_targ = v0 * direction * mag_curve
    
    # 4. LOW ALTITUDE OVERRIDE
    # If below 15m, vertical descent
    if pos[2] < 15.0:
        v_targ = np.array([0.0, 0.0, -2.0])
        
    return v_targ

def solve_cylindrical_field(r_vals, theta_vals, z_vals):
    """
    Generates a grid of consistent velocity vectors based on Cylindrical coordinates.
    This emphasizes the symmetry of the problem.
    """
    # Create meshgrid using Cylindrical Coordinates
    # R: Radius, THETA: Angle, Z: Height
    R, THETA, Z = np.meshgrid(r_vals, theta_vals, z_vals, indexing='ij')
    
    # Convert grid points to Cartesian for Physics calculations
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    W = np.zeros_like(X)
    
    # Iterate through grid
    it = np.nditer([X, Y, Z], flags=['multi_index'])
    for x, y, z in it:
        idx = it.multi_index
        pos = np.array([x, y, z])
        
        # Iteratively solve for consistent velocity
        v_mag_guess = 20.0 
        for _ in range(3): # Converge on consistent velocity
            v_targ = calculate_target_velocity(pos, v_mag_guess)
            v_mag_guess = np.linalg.norm(v_targ)
        
        U[idx] = v_targ[0]
        V[idx] = v_targ[1]
        W[idx] = v_targ[2]
            
    return X, Y, Z, U, V, W

def main():
    # --- CONFIG ---
    Z_MAX = 1000   # Start altitude (m)
    R_MAX = 500    # Max Radius (m)
    
    # Resolution parameters for Cylindrical Grid
    R_RES = 6      # Number of concentric rings (radial resolution)
    THETA_RES = 16 # Number of spokes (angular resolution)
    Z_RES = 15     # Vertical resolution
    
    print("🚀 Generating Interactive 3D Velocity Field (Cylindrical)...")
    
    # 1. Generate Cylindrical Grid Points
    r = np.linspace(50, R_MAX, R_RES)     # Avoid r=0 to keep arrows distinct
    theta = np.linspace(0, 2*np.pi, THETA_RES, endpoint=False) # Full circle
    z = np.linspace(0, Z_MAX, Z_RES)
    
    # Solve field and get Cartesian coordinates back for plotting
    X, Y, Z, U, V, W = solve_cylindrical_field(r, theta, z)
    
    # Calculate magnitude for coloring
    M = np.sqrt(U**2 + V**2 + W**2)
    
    # 2. Create Plotly Figure
    fig = go.Figure()
    
    # A. 3D Cones (Vector Field)
    fig.add_trace(go.Cone(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        u=U.flatten(),
        v=V.flatten(),
        w=W.flatten(),
        colorscale='Plasma',
        sizemode="scaled",
        sizeref=0.7, 
        colorbar=dict(title='Speed (m/s)'),
        name='Velocity Field'
    ))
    
    # B. Virtual Target Marker (Green Diamond)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[15],
        mode='markers',
        marker=dict(size=8, color='green', symbol='diamond'),
        name='Virtual Target (15m)'
    ))

    # C. Ground / Landing Pad (Circle)
    theta_pad = np.linspace(0, 2*np.pi, 100)
    r_pad = 50.0 # Visual pad size
    x_pad = r_pad * np.cos(theta_pad)
    y_pad = r_pad * np.sin(theta_pad)
    z_pad = np.zeros_like(theta_pad)
    
    fig.add_trace(go.Scatter3d(
        x=x_pad, y=y_pad, z=z_pad,
        mode='lines',
        line=dict(color='black', width=5),
        name='Landing Zone'
    ))

    # D. Layout Settings
    fig.update_layout(
        title="Interactive 3D Guidance Velocity Field (Cylindrical View)",
        scene=dict(
            xaxis_title='Downrange (X)',
            yaxis_title='Crossrange (Y)',
            zaxis_title='Altitude (Z)',
            aspectmode='data', # Ensures 1:1:1 scale
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.6)"
        )
    )
    
    filename = "velocity_field_3d_cylindrical.html"
    fig.write_html(filename)
    print(f"✅ Saved interactive plot to '{filename}'")
    
    # Try to open automatically
    import sys, subprocess, os
    try:
        if sys.platform == "darwin": subprocess.call(["open", filename])
        elif sys.platform == "win32": os.startfile(filename)
        else: subprocess.call(["xdg-open", filename])
    except: pass

if __name__ == "__main__":
    main()