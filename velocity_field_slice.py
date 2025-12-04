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

def solve_slice_field(r_max, z_max, r_res, z_res):
    """
    Generates a single 2D planar slice of the velocity field.
    We map this to the X-Z plane (Y=0) for X >= 0.
    """
    # Create meshgrid for Radius and Height
    # Start r at a small offset to avoid the singularity at r=0 for direction calculation
    r_vals = np.linspace(10, r_max, r_res)
    z_vals = np.linspace(0, z_max, z_res)
    
    R, Z = np.meshgrid(r_vals, z_vals, indexing='ij')
    
    # Map to Cartesian: This slice lies on the X-axis (Theta = 0)
    X = R
    Y = np.zeros_like(R)
    
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
    R_MAX = 400    # Max Radius (m)
    
    # Resolution parameters for the Slice
    R_RES = 15     # Points along the radius
    Z_RES = 15     # Points along the height
    
    print("🚀 Generating Velocity Field Slice (X-Z Plane)...")
    
    X, Y, Z, U, V, W = solve_slice_field(R_MAX, Z_MAX, R_RES, Z_RES)
    
    # Calculate magnitude for coloring verification
    M = np.sqrt(U**2 + V**2 + W**2)
    min_speed = np.min(M)
    max_speed = np.max(M)
    print(f"   ► Speed Range: {min_speed:.2f} m/s to {max_speed:.2f} m/s")
    
    # 2. Create Plotly Figure
    fig = go.Figure()
    
    # A. 3D Cones (Vector Field Slice)
    fig.add_trace(go.Cone(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        u=U.flatten(),
        v=V.flatten(),
        w=W.flatten(),
        colorscale='Plasma',
        # Explicitly lock the color scale to the calculated magnitudes
        cmin=min_speed,
        cmax=max_speed,
        sizemode="scaled",
        sizeref=0.5, 
        colorbar=dict(title='Speed (m/s)'),
        name='Velocity Profile'
    ))
    
    # B. Virtual Target Marker (Green Diamond)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[15],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name='Virtual Target (15m)'
    ))
    
    # C. Center Line (Z-axis) - Visual Reference for Symmetry
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, Z_MAX],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name='Central Axis'
    ))

    # D. Ground / Landing Pad (Circle)
    theta_pad = np.linspace(0, 2*np.pi, 100)
    r_pad = 50.0 
    x_pad = r_pad * np.cos(theta_pad)
    y_pad = r_pad * np.sin(theta_pad)
    z_pad = np.zeros_like(theta_pad)
    
    fig.add_trace(go.Scatter3d(
        x=x_pad, y=y_pad, z=z_pad,
        mode='lines',
        line=dict(color='black', width=5),
        name='Landing Zone'
    ))

    # E. Layout Settings
    fig.update_layout(
        title="Guidance Velocity Profile (Single Slice)",
        scene=dict(
            xaxis_title='Downrange (X)',
            yaxis_title='Crossrange (Y)',
            zaxis_title='Altitude (Z)',
            aspectmode='data',
            camera=dict(
                # Set camera to view the slice side-on by default
                eye=dict(x=0, y=-2.5, z=0.5),
                up=dict(x=0, y=0, z=1)
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
    
    filename = "velocity_field_slice.html"
    fig.write_html(filename)
    print(f"✅ Saved interactive slice to '{filename}'")
    
    # Try to open automatically
    import sys, subprocess, os
    try:
        if sys.platform == "darwin": subprocess.call(["open", filename])
        elif sys.platform == "win32": os.startfile(filename)
        else: subprocess.call(["xdg-open", filename])
    except: pass

if __name__ == "__main__":
    main()