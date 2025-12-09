import numpy as np
from scipy.spatial.transform import Rotation as R

def initialize_rocket(data, seed=None):
    """
    Sets the initial state (qpos, qvel) of the rocket with random noise for training.
    
    Args:
        data: The MuJoCo data object (modified in-place).
        seed: Random seed (int) for reproducibility.
    """
    # 1. Setup Random Generator
    rng = np.random.default_rng(seed)

    # ==========================================
    # POSITION (x, y, z)
    # ==========================================
    # Target: 2km Up (z=2000), 250m Horizontal Distance (x=250)
    # Noise: +/- 50m (x, z), +/- 20m (y)
    base_pos = np.array([250.0, 0.0, 2000.0])
    pos_noise = rng.uniform(low=[-50, -20, -50], high=[50, 20, 50])
    
    data.qpos[0:3] = base_pos + pos_noise

    # ==========================================
    # ORIENTATION (Quaternion: w, x, y, z)
    # ==========================================
    # Target: Pitch -90 degrees (Rocket nose points to horizon/-X)
    base_euler = [0, -90, 0] # Roll, Pitch, Yaw in degrees
    
    # Noise: +/- 5 degrees random rotation
    angle_noise = rng.uniform(-5, 5, size=3)
    
    # Combine and convert to Quaternion
    final_euler = base_euler + angle_noise
    r = R.from_euler('xyz', final_euler, degrees=True)
    x, y, z, w = r.as_quat() # Scipy returns [x, y, z, w]
    
    # MuJoCo expects [w, x, y, z]
    data.qpos[3:7] = [w, x, y, z]

    # ==========================================
    # LINEAR VELOCITY (vx, vy, vz)
    # ==========================================
    # REQUIREMENT: Magnitude must always be 350 m/s
    target_speed = 350.0
    
    # 1. Define a base direction vector 
    # [-1.0, 0.0, -0.2] means moving mostly Left (-X) and slightly Down (-Z)
    base_direction = np.array([-0.1, 0.0, -0.9])
    
    # 2. Add noise to the DIRECTION, not the speed
    # We perturb the vector slightly to vary the approach angle
    dir_noise = rng.uniform(-0.1, 0.1, size=3)
    noisy_direction = base_direction + dir_noise
    
    # 3. Normalize to Unit Vector (length = 1)
    unit_direction = noisy_direction / np.linalg.norm(noisy_direction)
    
    # 4. Scale to exactly 350 m/s
    final_vel = unit_direction * target_speed
    
    data.qvel[0:3] = final_vel

    # ==========================================
    # ANGULAR VELOCITY (wx, wy, wz)
    # ==========================================
    # Target: 0 spin
    # Noise: +/- 0.1 rad/s random spin
    data.qvel[3:6] = rng.uniform(-0.1, 0.1, size=3)