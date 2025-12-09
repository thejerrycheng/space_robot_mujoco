import argparse
import time
import os
import numpy as np
import mujoco
import mujoco.viewer
import sys
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# 1. HELPERS
# ==============================================================================
def get_metrics(model, data, body_id):  # <--- CHANGED: Added 'model' argument
    """
    Extracts and calculates readable metrics from MuJoCo data.
    """
    # Position (Dynamic state -> data)
    pos = data.xpos[body_id]
    
    # Velocity (Dynamic state -> data)
    vel = data.qvel[0:3]
    vel_mag = np.linalg.norm(vel)
    
    # Mass (Model Parameter -> model)
    # Even if we update this for fuel burn, it lives in 'model'
    mass = model.body_mass[body_id]     # <--- FIXED: data.body_mass -> model.body_mass
    
    # Tilt Calculation
    quat = data.qpos[3:7]
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    rocket_z_axis = r.apply([0, 0, 1])
    dot_product = np.clip(rocket_z_axis[2], -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(dot_product))
    
    return pos, vel, vel_mag, mass, tilt_deg

# ==============================================================================
# 2. INITIALIZATION LOGIC
# ==============================================================================
def initialize_high_velocity(data, seed=None):
    rng = np.random.default_rng(seed)

    # --- POSITION ---
    base_pos = np.array([250.0, 0.0, 2000.0])
    pos_noise = rng.uniform(low=[-50, -20, -50], high=[50, 20, 50])
    data.qpos[0:3] = base_pos + pos_noise

    # --- ORIENTATION (Pitch -90ish) ---
    base_euler = [0, -90, 0] 
    angle_noise = rng.uniform(-5, 5, size=3)
    final_euler = base_euler + angle_noise
    r_rot = R.from_euler('xyz', final_euler, degrees=True)
    x, y, z, w = r_rot.as_quat() 
    data.qpos[3:7] = [w, x, y, z]

    # --- LINEAR VELOCITY (350 m/s) ---
    target_speed = 350.0
    base_direction = np.array([-0.1, 0.0, -0.9]) 
    dir_noise = rng.uniform(-0.1, 0.1, size=3)
    # Normalize
    vec = base_direction + dir_noise
    unit_direction = vec / np.linalg.norm(vec)
    data.qvel[0:3] = unit_direction * target_speed

    # --- ANGULAR VELOCITY ---
    data.qvel[3:6] = rng.uniform(-0.1, 0.1, size=3)

# ==============================================================================
# 3. MAIN SIMULATION
# ==============================================================================
def launch_simulation(args):
    if not os.path.exists(args.xml):
        print(f"Error: Could not find XML file at {args.xml}")
        return

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    
    try:
        rocket_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    except:
        print("⚠️ Warning: Body 'ball' not found. Using Body 1.")
        rocket_bid = 1

    print(f"🚀 Simulation Initialized: {args.episodes} Episodes")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [1000, 0, 0] 
        viewer.cam.distance = 2000.0
        viewer.cam.azimuth = 80.0
        viewer.cam.elevation = -30.0
        
        for ep in range(1, args.episodes + 1):
            if not viewer.is_running(): break
            
            # --- EPISODE START ---
            mujoco.mj_resetData(model, data)
            seed = np.random.randint(0, 10000)
            initialize_high_velocity(data, seed)
            
            # Print Header
            print(f"\n{'='*80}")
            print(f"EPISODE {ep}/{args.episodes} | Seed: {seed}")
            print(f"{'='*80}")
            print(f"{'TIME (s)':<10} | {'POS [X, Y, Z] (m)':<25} | {'VEL (m/s)':<10} | {'TILT (°)':<10} | {'MASS (t)':<10}")
            print(f"{'-'*80}")

            start_time = time.time()
            frame_count = 0
            
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data)
                frame_count += 1
                viewer.sync()

                # --- LOGGING (Every 20 frames / ~0.1 seconds) ---
                if frame_count % 20 == 0:
                    # UPDATED CALL: Passing 'model' as the first argument
                    pos, vel, vel_mag, mass, tilt = get_metrics(model, data, rocket_bid)
                    sim_time = data.time
                    
                    pos_str = f"[{pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f}]"
                    print(f"{sim_time:8.2f}   | {pos_str:<25} | {vel_mag:9.1f}  | {tilt:8.1f}   | {mass/1000:8.1f}")

                # --- TERMINATION ---
                # Check for ground impact
                if data.qpos[2] < 50.0: 
                    print(f"{'-'*80}")
                    print("💥 IMPACT DETECTED")
                    time.sleep(1.0) 
                    break
                
                if frame_count > 2000: 
                    print("⌛ TIMEOUT")
                    break

                # Clock Sync
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rocket Data Logger")
    parser.add_argument('--xml', type=str, default="assets/mjcf/realistic_param.xml")
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    launch_simulation(args)