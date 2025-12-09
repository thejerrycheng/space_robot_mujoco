"""
PID Controller Demo for Rocket Landing Environment
Demonstrates how to use the refactored environment with a simple control system.

IMPORTANT: This rocket has a 30m thrust offset below the center of mass!
This means:
1. Tilting the rocket changes the thrust vector direction significantly
2. The gimbal provides TORQUE control, not direct thrust vectoring
3. We need to be very careful about tilt angles to maintain vertical thrust
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# Fix for macOS threading issue with matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from rocket_env.standard_gym import RocketLandingEnv


class PIDController:
    """Simple PID controller for a single variable."""
    
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Tuple of (min, max) output values
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        # State variables
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
    
    def compute(self, error, dt):
        """
        Compute PID control output.
        
        Args:
            error: Current error value
            dt: Time step
            
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        # Anti-windup: clamp integral
        max_integral = 10.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        i_term = self.ki * self.integral
        
        # Derivative term
        if self.last_time is not None:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        # Update state
        self.last_error = error
        self.last_time = dt
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output


class RocketPIDController:
    """
    PID-based controller for rocket landing with 30m thrust offset.
    
    The thrust offset means:
    - Gimbal angles create torques around the CoM
    - Small gimbal deflections can create large moments (30m lever arm!)
    - We need to keep the rocket nearly upright to maintain vertical thrust
    """
    
    def __init__(self, dt=0.02):
        """
        Initialize rocket controller.
        
        Args:
            dt: Control timestep (should match environment dt)
        """
        self.dt = dt
        
        # Physical parameters
        self.THRUST_OFFSET = 30.0  # meters below CoM
        self.MAX_GIMBAL_ANGLE = 30.0  # degrees
        
        # Thrust controller (controls vertical acceleration/velocity)
        # More aggressive gains since we have high TWR (~3)
        self.thrust_pid = PIDController(
            kp=0.15,      # Stronger proportional response
            ki=0.002,     # Small integral to eliminate drift
            kd=0.25,      # Strong damping
            output_limits=(-1.0, 1.0)
        )
        
        # Attitude controllers - MUST BE GENTLE due to long lever arm
        # Small gimbal angles create HUGE torques!
        self.pitch_pid = PIDController(
            kp=0.008,     # Very small proportional (30m lever makes this powerful)
            ki=0.00005,   # Minimal integral
            kd=0.03,      # Moderate damping
            output_limits=(-0.5, 0.5)  # Limit gimbal deflection
        )
        
        self.yaw_pid = PIDController(
            kp=0.008,
            ki=0.00005,
            kd=0.03,
            output_limits=(-0.5, 0.5)
        )
        
        # Desired descent rate (m/s, negative means down)
        self.target_vz = -10.0
        
        # Phase tracking
        self.phase = "descent"
    
    def reset(self):
        """Reset all controllers."""
        self.thrust_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.phase = "descent"
    
    def compute_action(self, obs):
        """
        Compute control action from observation.
        
        Args:
            obs: Environment observation [pos(3), vel(3), quat(4), ang_vel(3), fuel(1)]
            
        Returns:
            action: [thrust, yaw, pitch] in range [-1, 1]
            info: Dictionary with control information
        """
        # Parse observation
        pos = obs[0:3]
        vel = obs[3:6]
        quat = obs[6:10]  # [w, x, y, z] in MuJoCo format
        ang_vel = obs[10:13]
        fuel_ratio = obs[13]
        
        altitude = pos[2]
        vx, vy, vz = vel
        
        # Compute orientation
        # MuJoCo uses [w, x, y, z], scipy uses [x, y, z, w]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        
        # Get rocket's local Z-axis in world frame (thrust direction)
        z_body = r.apply([0, 0, 1])
        
        # Compute tilt from vertical
        tilt_angle = np.degrees(np.arccos(np.clip(z_body[2], -1.0, 1.0)))
        
        # Compute tilt components in world X and Y
        # These represent how much the rocket is leaning in each direction
        tilt_x = np.arctan2(z_body[0], z_body[2])  # Lean in X direction
        tilt_y = np.arctan2(z_body[1], z_body[2])  # Lean in Y direction
        
        # === PHASE LOGIC ===
        if altitude > 300:
            self.phase = "descent"
            self.target_vz = -20.0  # Fast descent
        elif altitude > 150:
            self.phase = "approach"
            self.target_vz = -10.0  # Medium descent
        elif altitude > 80:
            self.phase = "hover"
            self.target_vz = -5.0   # Slow descent
        else:
            self.phase = "landing"
            self.target_vz = -2.0   # Very slow final approach
        
        # === THRUST CONTROL ===
        # Account for current tilt - if tilted, we lose vertical thrust!
        thrust_efficiency = np.cos(np.radians(tilt_angle))
        
        # Error: difference between current and target vertical velocity
        vz_error = self.target_vz - vz
        thrust_cmd = self.thrust_pid.compute(vz_error, self.dt)
        
        # Compensate for tilt (need more thrust if tilted)
        if thrust_efficiency > 0.1:  # Avoid division by zero
            thrust_cmd = thrust_cmd / thrust_efficiency
        
        # Emergency thrust if falling too fast near ground
        if altitude < 100 and vz < -15:
            thrust_cmd = 1.0
        
        # Minimum thrust to maintain control
        if altitude > 50:
            thrust_cmd = max(thrust_cmd, 0.2)
        
        # === ATTITUDE CONTROL ===
        # Strategy: Keep rocket UPRIGHT (critical with thrust offset!)
        # Only use small corrections to brake horizontal velocity
        
        # Target tilt: very small lean to counteract horizontal velocity
        # But prioritize staying upright!
        horizontal_speed = np.linalg.norm([vx, vy])
        
        if horizontal_speed > 1.0:
            # Only lean if we have significant horizontal velocity
            lean_factor = 0.02  # Very conservative
            target_tilt_x = -np.clip(vx * lean_factor, -0.1, 0.1)
            target_tilt_y = -np.clip(vy * lean_factor, -0.1, 0.1)
        else:
            # Stay perfectly upright
            target_tilt_x = 0.0
            target_tilt_y = 0.0
        
        # Compute tilt errors (in radians)
        tilt_x_error = target_tilt_x - tilt_x
        tilt_y_error = target_tilt_y - tilt_y
        
        # Compute gimbal commands
        # NOTE: Gimbal creates torque, not direct thrust vectoring!
        pitch_cmd = self.pitch_pid.compute(tilt_x_error, self.dt)
        yaw_cmd = self.yaw_pid.compute(tilt_y_error, self.dt)
        
        # Add strong angular velocity damping (prevent oscillation)
        # With 30m lever arm, angular velocity creates huge centrifugal effects
        angular_damping = 0.3
        pitch_cmd -= ang_vel[1] * angular_damping
        yaw_cmd -= ang_vel[0] * angular_damping
        
        # Safety limits: never exceed safe gimbal angles
        # Map to actual gimbal angles for safety check
        pitch_angle_estimate = pitch_cmd * self.MAX_GIMBAL_ANGLE
        yaw_angle_estimate = yaw_cmd * self.MAX_GIMBAL_ANGLE
        
        # If tilt is already large, reduce gimbal authority
        tilt_safety_factor = np.clip(1.0 - tilt_angle / 45.0, 0.1, 1.0)
        pitch_cmd *= tilt_safety_factor
        yaw_cmd *= tilt_safety_factor
        
        # Final clipping
        pitch_cmd = np.clip(pitch_cmd, -1.0, 1.0)
        yaw_cmd = np.clip(yaw_cmd, -1.0, 1.0)
        
        action = np.array([thrust_cmd, yaw_cmd, pitch_cmd], dtype=np.float32)
        
        return action, {
            'phase': self.phase,
            'altitude': altitude,
            'vz': vz,
            'horizontal_speed': horizontal_speed,
            'tilt_angle': tilt_angle,
            'tilt_x': np.degrees(tilt_x),
            'tilt_y': np.degrees(tilt_y),
            'thrust_efficiency': thrust_efficiency,
            'vz_error': vz_error,
            'gimbal_pitch': pitch_cmd,
            'gimbal_yaw': yaw_cmd,
        }


def run_pid_demo(num_episodes=3, render=True, plot_results=True, max_steps=2000):
    """
    Run PID controller demo.
    
    Args:
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        plot_results: Whether to plot episode statistics
        max_steps: Maximum steps per episode
    """
    # Create environment
    render_mode = "human" if render else None
    env = RocketLandingEnv(reward_type="dense", render_mode=render_mode)
    
    # Create controller
    controller = RocketPIDController(dt=env.dt)
    
    # Storage for statistics
    episode_stats = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        # Reset
        obs, info = env.reset(seed=42 + episode)
        controller.reset()
        
        # Episode data
        episode_data = {
            'altitudes': [],
            'velocities': [],
            'horizontal_speeds': [],
            'tilts': [],
            'thrusts': [],
            'gimbal_pitch': [],
            'gimbal_yaw': [],
            'rewards': [],
            'phases': [],
            'thrust_efficiency': [],
        }
        
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < max_steps:
            # Compute action
            action, control_info = controller.compute_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Accumulate reward
            total_reward += reward
            step += 1
            
            # Store data
            episode_data['altitudes'].append(control_info['altitude'])
            episode_data['velocities'].append(control_info['vz'])
            episode_data['horizontal_speeds'].append(control_info['horizontal_speed'])
            episode_data['tilts'].append(control_info['tilt_angle'])
            episode_data['thrusts'].append(action[0])
            episode_data['gimbal_pitch'].append(control_info['gimbal_pitch'])
            episode_data['gimbal_yaw'].append(control_info['gimbal_yaw'])
            episode_data['rewards'].append(reward)
            episode_data['phases'].append(control_info['phase'])
            episode_data['thrust_efficiency'].append(control_info['thrust_efficiency'])
            
            # Print periodic updates
            if step % 100 == 0:
                print(f"Step {step:4d} | "
                      f"Phase: {control_info['phase']:8s} | "
                      f"Alt: {control_info['altitude']:7.1f}m | "
                      f"Vz: {control_info['vz']:6.2f}m/s | "
                      f"Tilt: {control_info['tilt_angle']:5.1f}° | "
                      f"H-spd: {control_info['horizontal_speed']:5.1f}m/s | "
                      f"Eff: {control_info['thrust_efficiency']:4.2f}")
            
            # Render
            if render:
                env.render()
        
        # Episode summary
        print(f"\n{'─'*70}")
        print(f"Episode finished after {step} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Success: {'✓' if info['is_success'] else '✗'} {info['is_success']}")
        print(f"Final altitude: {info['altitude']:.2f}m")
        print(f"Final lateral distance: {info['lateral_distance']:.2f}m")
        print(f"Final tilt: {info['tilt_angle']:.2f}°")
        print(f"Fuel remaining: {info['fuel_remaining']:.0f} kg "
              f"({100*info['fuel_remaining']/(env.START_FUEL):.1f}%)")
        print(f"{'─'*70}")
        
        episode_stats.append({
            'episode': episode + 1,
            'steps': step,
            'reward': total_reward,
            'success': info['is_success'],
            'data': episode_data,
        })
    
    env.close()
    
    # Plot results
    if plot_results and len(episode_stats) > 0:
        plot_episode_statistics(episode_stats)
    
    return episode_stats


def plot_episode_statistics(episode_stats):
    """Plot statistics from episodes."""
    num_episodes = len(episode_stats)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('PID Controller Performance (30m Thrust Offset)', 
                 fontsize=16, fontweight='bold')
    
    for i, stats in enumerate(episode_stats):
        data = stats['data']
        label = f"Ep {i+1} ({'✓' if stats['success'] else '✗'})"
        color = plt.cm.viridis(i / max(num_episodes - 1, 1))
        
        # Altitude over time
        axes[0, 0].plot(data['altitudes'], label=label, color=color, alpha=0.8, linewidth=2)
        
        # Vertical velocity over time
        axes[0, 1].plot(data['velocities'], label=label, color=color, alpha=0.8, linewidth=2)
        
        # Tilt angle over time
        axes[1, 0].plot(data['tilts'], label=label, color=color, alpha=0.8, linewidth=2)
        
        # Horizontal speed over time
        axes[1, 1].plot(data['horizontal_speeds'], label=label, color=color, alpha=0.8, linewidth=2)
        
        # Thrust command over time
        axes[2, 0].plot(data['thrusts'], label=label, color=color, alpha=0.8, linewidth=2)
        
        # Gimbal deflections
        axes[2, 1].plot(data['gimbal_pitch'], label=f"{label} Pitch", 
                       color=color, alpha=0.8, linewidth=1.5, linestyle='-')
        axes[2, 1].plot(data['gimbal_yaw'], 
                       color=color, alpha=0.6, linewidth=1.5, linestyle='--')
    
    # Configure subplots
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude Profile')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Vertical Velocity (m/s)')
    axes[0, 1].set_title('Descent Rate')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Tilt Angle (degrees)')
    axes[1, 0].set_title('Rocket Orientation (Critical with 30m Thrust Offset!)')
    axes[1, 0].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Upright')
    axes[1, 0].axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='±10° limit')
    axes[1, 0].axhline(y=-10, color='orange', linestyle=':', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Horizontal Speed (m/s)')
    axes[1, 1].set_title('Lateral Velocity')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Thrust Command [-1, 1]')
    axes[2, 0].set_title('Thrust Control')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('Gimbal Deflection [-1, 1]')
    axes[2, 1].set_title('Gimbal Control (Pitch=solid, Yaw=dashed)')
    axes[2, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pid_controller_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Plot saved as 'pid_controller_results.png'")
    # Don't call plt.show() to avoid GUI issues on macOS
    print("   Open the file to view the plots.")


def main():
    """Main function."""
    print("="*70)
    print("🚀 Rocket Landing Environment - PID Controller Demo")
    print("⚠️  NOTE: Rocket has 30m thrust offset below CoM!")
    print("="*70)
    
    # Run demo
    stats = run_pid_demo(
        num_episodes=3,
        render=False,  # Set to True to see visualization
        plot_results=True,
        max_steps=2000
    )
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("📊 Summary Statistics")
    print(f"{'='*70}")
    
    successes = sum(1 for s in stats if s['success'])
    avg_reward = np.mean([s['reward'] for s in stats])
    avg_steps = np.mean([s['steps'] for s in stats])
    
    print(f"Success rate: {successes}/{len(stats)} ({100*successes/len(stats):.1f}%)")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"{'='*70}")
    
    print("\n💡 Key Insights:")
    print("   • 30m thrust offset creates HUGE torque leverage")
    print("   • Small gimbal angles (few degrees) create large moments")
    print("   • Must keep rocket nearly upright for vertical thrust")
    print("   • Angular velocity damping is critical to prevent oscillation")


if __name__ == "__main__":
    main()