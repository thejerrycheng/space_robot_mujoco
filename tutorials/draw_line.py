import time
import numpy as np
import mujoco
import mujoco.viewer

# -----------------------------
# Simple projectile model
# -----------------------------
XML = """
<mujoco model="projectile">
  <compiler angle="degree"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <!-- ground plane -->
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <!-- ball with free joint -->
    <body name="ball" pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# -----------------------------
# Initial velocity (45 degrees)
# -----------------------------
speed = 10.0           # m/s
angle = np.deg2rad(45)
vx = speed * np.cos(angle)
vz = speed * np.sin(angle)

data.qvel[0] = vx      # x velocity
data.qvel[1] = 0.0     # y velocity
data.qvel[2] = vz      # z velocity

# -----------------------------
# Simulation with custom rendering
# -----------------------------
positions = []  # store (x, y, z) for drawing trail
t = 0.0
T_MAX = 3.0      # seconds to simulate

# Create visualization scene for custom rendering
scene = mujoco.MjvScene(model, maxgeom=10000)
opt = mujoco.MjvOption()

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 6.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    
    while viewer.is_running() and t < T_MAX:
        step_start = time.time()
        
        # record current ball position
        x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
        positions.append([x, y, z])
        
        # advance physics
        mujoco.mj_step(model, data)
        t += model.opt.timestep
        
        # Update the scene
        mujoco.mjv_updateScene(
            model, data, opt, None, viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        
        # Draw trail using capsules (cylinders) to connect points
        for i in range(len(positions) - 1):
            if scene.ngeom >= scene.maxgeom:
                break
                
            start = np.array(positions[i], dtype=np.float64)
            end = np.array(positions[i + 1], dtype=np.float64)
            
            # Calculate midpoint and half-length
            midpoint = (start + end) / 2
            half_vec = (end - start) / 2
            length = np.linalg.norm(end - start)
            
            if length < 1e-6:
                continue
            
            # Create rotation matrix to align cylinder with the line
            z_axis = (end - start) / length
            # Find a perpendicular vector
            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross(z_axis, [0, 0, 1])
            else:
                x_axis = np.cross(z_axis, [1, 0, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Rotation matrix
            mat = np.column_stack([x_axis, y_axis, z_axis]).flatten()
            
            # Initialize geometry for capsule segment
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([0.005, 0.005, length/2]),  # radius, radius, half-length
                pos=midpoint,
                mat=mat,
                rgba=np.array([0, 1, 0, 0.8])  # Green trail
            )
            
            scene.ngeom += 1
        
        # Sync viewer
        viewer.sync()
        
        # keep roughly real-time
        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)

print(f"Simulation complete. Trajectory had {len(positions)} points.")