import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plots
from scipy.spatial.transform import Rotation as R
import platform
import subprocess
import os
import sys
import time
import argparse
import numpy as np
import os, sys
import csv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# =====================
# ===========================================
#   FILE & PLOTTING FUNCTIONS (MODIFIED)
# ================================================================

def open_file(path):
    """ Cross-platform file opener """
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.call(["open", path])
        elif sys.platform == "win32": # Windows
            os.startfile(path)
        else: # Linux
            subprocess.call(["xdg-open", path])
    except Exception:
        pass

def save_to_csv(history, episode_num, save_dir):
    """ Saves episode trajectory data to a CSV file. """
    filename = os.path.join(save_dir, f"episode_{episode_num}.csv")
    
    times = history['time']
    pos = np.array(history['pos'])
    vel = np.array(history['vel'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    reward = np.array(history['reward'])
    r_upright = np.array(history['r_upright'])
    r_vel = np.array(history['r_vel'])
    r_dist = np.array(history['r_dist'])

    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        writer.writerow([
            "Step", "Time", "X", "Y", "Z", "Vx", "Vy", "Vz",
            "Roll", "Pitch", "Yaw", 
            "Thrust", "GimbalYaw", "GimbalPitch", 
            "Mass", "TotalReward", "R_Upright", "R_Velocity", "R_Distance"
        ])
        
        # Rows
        for i in range(len(times)):
            writer.writerow([
                i, times[i], 
                pos[i,0], pos[i,1], pos[i,2],
                vel[i,0], vel[i,1], vel[i,2],
                att[i,0], att[i,1], att[i,2],
                thrust[i], gimbal[i,0], gimbal[i,1],
                mass[i], reward[i], r_upright[i], r_vel[i], r_dist[i]
            ])
            
    print(f"💾 Data saved to: {filename}")
    
def get_body_z_axis(quat):
    """ 
    Calculates the body frame Z-axis vector from a quaternion [w, x, y, z].
    Used to visualize where the rocket is pointing.
    """
    # Scipy Rotation expects [x, y, z, w]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    # Apply rotation to the standard Z vector [0, 0, 1]
    return r.apply([0, 0, 1])

def plot_unified_analysis(history, episode_num, model_name, save_dir):
    """ Generates a single figure with 6 subplots as requested, including the 80m tolerance circle. """
    times = np.array(history['time'])
    pos = np.array(history['pos'])
    quat = np.array(history['quat'])
    att = np.array(history['attitude'])
    thrust = np.array(history['thrust'])
    gimbal = np.array(history['gimbal'])
    mass = np.array(history['mass'])
    total_reward = np.array(history['reward'])
    r_upright = np.array(history['r_upright'])
    r_vel = np.array(history['r_vel'])
    r_dist = np.array(history['r_dist'])
    
    # 3. Tilt of the rocket over time
    tilt_mag = np.sqrt(att[:, 0]**2 + att[:, 1]**2)
    
    # 4. Distance to target and altitude of the rocket
    lateral_dist = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    altitude = pos[:, 2]

    # Create the figure with 2 rows and 3 columns
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"SAC Analysis: {model_name} - Episode {episode_num}", fontsize=16)

    # 1. Overall 3D Trajectory of the rocket with orientation arrows (MODIFIED)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_title("1. 3D Trajectory & Orientation")
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], label='Trajectory', color='b')
    ax1.scatter(0, 0, 0, color='k', marker='*', s=100, label='Target')
    
    # --- NEW: Add 80-meter Radius Tolerance Circle ---
    R_TOLERANCE = 80.0
    # Generate points for a circle in the XY plane (Z=0)
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = R_TOLERANCE * np.cos(theta)
    y_circle = R_TOLERANCE * np.sin(theta)
    z_circle = np.zeros_like(theta)
    
    ax1.plot(x_circle, y_circle, z_circle, color='orange', linestyle='--', 
             label=f'{R_TOLERANCE}m Tolerance Zone')
    # --------------------------------------------------
    
    # Add orientation arrows only if quat is valid
    if quat is not None and hasattr(quat, "shape") and quat.shape == (len(pos), 4):
        # 如果全部四元数都是零，则跳过
        if not np.allclose(quat, 0):
            step_interval = max(1, len(pos) // 50)
            indices = np.arange(0, len(pos), step_interval)
            for i in indices:
                try:
                    vec = get_body_z_axis(quat[i])
                    ax1.quiver(pos[i, 0], pos[i, 1], pos[i, 2],
                            vec[0], vec[1], vec[2],
                            length=10, normalize=True, color='r', arrow_length_ratio=0.3)
                except Exception:
                    pass

    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m - Altitude)')
    ax1.legend()
    # Ensure the plot limits are set to capture the 80m circle for a complete view
    max_range = max(R_TOLERANCE, np.max(np.abs(pos))) * 1.1
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([0, np.max(pos[:,2]) * 1.1])
    ax1.set_box_aspect([1,1,1])

    # 2. The mass of the rocket over time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("2. Rocket Mass Over Time")
    ax2.plot(times, mass, color='black')
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Mass (kg)"); ax2.grid(True)
    
    # 3. The tilt of the rocket over time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("3. Rocket Tilt (Angular Deviation)")
    ax3.plot(times, tilt_mag, label='Tilt Magnitude (Roll/Pitch)', color='purple')
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Tilt (Deg)"); ax3.grid(True)

    # 4. The distance to the target and the altitude of the rocket
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("4. Position: Distance to Target & Altitude")
    ax4.plot(times, altitude, label='Altitude (Z)', color='green')
    ax4.plot(times, lateral_dist, label='Lateral Distance to Target', color='orange', linestyle='--')
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Distance (m)"); ax4.legend(); ax4.grid(True)

        # 5. Control: Thrust magnitude & Gimbal pitch/roll angles (Normalized Actions)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("5. Control Commands (Normalized Actions [-1, 1])")

    # ------------------------
    # Normalize helper
    # ------------------------
    def normalize(x):
        x_min, x_max = np.min(x), np.max(x)
        if x_max - x_min < 1e-6:
            return np.zeros_like(x)
        return 2 * (x - x_min) / (x_max - x_min) - 1  # scale → [-1, 1]

    # thrust 已经是 normalized，可直接使用
    thrust_norm = thrust  

    # gimbal 两个通道：yaw = gimbal[:,0], pitch = gimbal[:,1]
    pitch_norm = normalize(gimbal[:, 1])
    yaw_norm   = normalize(gimbal[:, 0])

    # ------------------------
    # Draw
    # ------------------------
    ax5.plot(times, thrust_norm, label='Thrust Command (Norm.)', color='r')
    ax5.plot(times, pitch_norm, label='Gimbal Pitch Command (Norm.)', color='b')
    ax5.plot(times, yaw_norm,   label='Gimbal Yaw Command (Norm.)', color='c', linestyle=':')

    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Normalized Action")
    ax5.legend()
    ax5.grid(True)
    ax5.set_ylim([-1.05, 1.05])


    # 6. The reward over time including components
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("6. Reward Breakdown Over Time")
    ax6.plot(times, total_reward, label='Total Reward', color='k', linewidth=2)
    ax6.plot(times, r_upright, label='Upright Reward', linestyle='--', color='g')
    ax6.plot(times, r_vel, label='Velocity Reward', linestyle='--', color='b')
    ax6.plot(times, r_dist, label='Position Reward', linestyle='--', color='orange')
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Reward Value"); ax6.legend(); ax6.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    model_dir = os.path.basename(os.path.normpath(save_dir))
    save_path = os.path.join(save_dir, f"episode_{episode_num}_unified_analysis.png")
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"🖼️ Unified Plot saved to: {save_path}")
    
    
