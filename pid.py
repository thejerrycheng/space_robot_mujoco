import time
import numpy as np
from rocket_env.rocket_landing_env import RocketLandingEnv


# ================================================================
#   SIMPLE PID CLASS
# ================================================================
class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(-np.inf, np.inf)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def __call__(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        low, high = self.output_limits
        output = np.clip(output, low, high)

        self.prev_error = error
        return output


# ====================================================================
#   OPTIONAL: Convert quaternion to Euler for easy terminal logging
# ====================================================================
def quat_to_euler(q):
    w, x, y, z = q

    # roll (x)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # pitch (y)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw (z)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


# ================================================================
#   MAIN PID RUNNER WITH TERMINAL LOGGING
# ================================================================
def run_pid_controller():
    env = RocketLandingEnv(render_mode="human")
    obs, _ = env.reset()

    dt = env.DT
    TARGET_Z = env.TARGET_Z

    print("🚀 Starting PID controller with terminal logging...\n")

    # PID loops
    pid_alt = PID(kp=3.0, ki=0.0, kd=6.0, setpoint=TARGET_Z, output_limits=(-1, 1))
    pid_pitch = PID(kp=1.2, ki=0, kd=0.5, setpoint=0, output_limits=(-1, 1))
    pid_yaw = PID(kp=1.2, ki=0, kd=0.5, setpoint=0, output_limits=(-1, 1))

    for step in range(5000):
        pos = obs[0:3]
        vel = obs[3:6]
        quat = obs[9:13]

        x, y, z = pos
        vx, vy, vz = vel

        # Compute PID control signals
        thrust_norm = pid_alt(z, dt)
        pitch_norm  = pid_pitch(vx, dt)
        yaw_norm    = pid_yaw(vy, dt)

        # Convert normalized controls → physical values
        thrust_N = (thrust_norm + 1) * 0.5 * env.MAX_THRUST
        pitch_rad = pitch_norm * env.MAX_GIMBAL
        yaw_rad   = yaw_norm * env.MAX_GIMBAL

        # Euler angles for logging (optional)
        roll_e, pitch_e, yaw_e = quat_to_euler(quat)

        # ============================
        # Terminal Logging
        # ============================
        print(f"\n=== Step {step} ===")
        print(f"Altitude (z):       {z:.3f} m")
        print(f"Fuel Mass:          {env.fuel_mass:.3f} kg")
        print(f"Pose (quat wxyz):   [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
        print(f"Pose (Euler rpy):   roll={roll_e:.3f} rad, pitch={pitch_e:.3f} rad, yaw={yaw_e:.3f} rad")
        print(f"Desired Altitude:   {TARGET_Z:.3f} m")

        print("\n--- Controls ---")
        print(f"Thrust (norm):      {thrust_norm:.3f}")
        print(f"Pitch (norm):       {pitch_norm:.3f}")
        print(f"Yaw (norm):         {yaw_norm:.3f}")

        print(f"Thrust (N):         {thrust_N:.1f} N")
        print(f"Pitch angle:        {pitch_rad:.3f} rad")
        print(f"Yaw angle:          {yaw_rad:.3f} rad")

        # Apply action
        action = np.array([thrust_norm, yaw_norm, pitch_norm], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        time.sleep(0.01)

        if terminated or truncated:
            print("\n🚨 Episode ended — resetting environment.\n")
            obs, _ = env.reset()
            pid_alt.reset()
            pid_pitch.reset()
            pid_yaw.reset()

    env.close()


if __name__ == "__main__":
    run_pid_controller()
