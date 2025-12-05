import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# Path to your XML file
xml_path = "assets/mjcf/realistic_param.xml"

# Check if file exists to prevent confusion
if not os.path.exists(xml_path):
    print(f"Error: Could not find file at {xml_path}")
    print("Please ensure the XML code is saved in the correct folder structure.")
    exit()

def launch_simulation():
    # 1. Load the model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 2. Set Initial Position (x, y, z)
    # We want 2km height (z=2000) and 250m distance (x=250)
    # The landing pad is at (0,0,0), so we offset by 250 on X.
    start_pos = [500.0, 0.0, 1000.0]

    # 3. Set Initial Orientation (Quaternion [w, x, y, z])
    # The rocket's default "up" is its local Z-axis.
    # To point it horizontally towards the pad (from x=250 towards x=0), 
    # we need to point the local Z-axis towards the World -X axis.
    # This requires a -90 degree rotation around the Y-axis.
    
    # Calculation:
    # angle = -90 degrees (-pi/2 radians)
    # axis = [0, 1, 0]
    # w = cos(angle/2) = cos(-pi/4) = 0.7071
    # y = sin(angle/2) = sin(-pi/4) = -0.7071
    start_quat = [0.7071068, 0.0, -0.7071068, 0.0]

    # Assign to qpos
    # The freejoint 'ball_free' is the first joint, so it occupies qpos[0] to qpos[6]
    # qpos layout for freejoint: [x, y, z, w, x, y, z]
    data.qpos[0:3] = start_pos
    data.qpos[3:7] = start_quat

    # -----------------------------------------------------------
    # 2. Set Initial Velocity (qvel)
    # -----------------------------------------------------------
    # qvel layout for freejoint: [vx, vy, vz, wx, wy, wz]
    
    # Linear Velocity (World Frame)
    # Moving towards the landing pad (from 250 to 0 on X axis) -> Negative X velocity
    # Falling downwards -> Negative Z velocity
    initial_linear_vel = [-50.0, 0.0, -300.0] 

    # Angular Velocity (Body Frame)
    # Let's add a tiny bit of spin just to test stability (optional)
    initial_angular_vel = [0.0, 0.0, 0.0]

    data.qvel[0:3] = initial_linear_vel
    data.qvel[3:6] = initial_angular_vel

    # Print initial state for verification
    print(f"Rocket initialized at Height: {start_pos[2]}m, Distance: {start_pos[0]}m")
    print(f"Initial Velocity: {initial_linear_vel} m/s")
    print("Launching viewer...")

    # 4. Launch the Passive Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
    

        # --- CAMERA CONFIGURATION ---
        # Look at the rocket's starting position
        viewer.cam.lookat[:] = [1000, 0, 0] 
        
        # Distance: 300m away (close enough to see the rocket, far enough to see motion)
        viewer.cam.distance = 2000.0
        
        # Azimuth: 90 degrees (View from the side/Y-axis to see the trajectory towards X)
        viewer.cam.azimuth = 80.0
        
        # Elevation: -10 degrees (Slightly above, looking down)
        viewer.cam.elevation = -30.0
        # -----------------------------

        # Close the viewer automatically if the simulation is stopped
        while viewer.is_running():
            step_start = time.time()

            # Step the physics
            mujoco.mj_step(model, data)

            # Sync changes to the viewer
            viewer.sync()

            # Sleep to maintain roughly real-time playback speed
            # (Remove this if you want the simulation to run as fast as possible)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    launch_simulation()