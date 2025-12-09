import numpy as np

class RocketReward:
    def __init__(self, config):
        self.cfg = config

    def compute(self, state, action, terminated, truncated, success):
        """
        Args:
            state (dict): Contains pos, vel, tilt, fuel, etc.
            action (np.array): The action taken.
            terminated (bool): Crashed or Landed.
            truncated (bool): Timeout or Flyaway.
            success (bool): Soft landing achieved.
        """
        rew = 0.0
        info = {}

        # 1. SHAPING: Distance to Target (Guide it to the pad)
        # Minimize horizontal and vertical distance
        dist_xy = state['lateral_dist']
        dist_z  = state['alt']
        
        # Penalize distance (normalized roughly)
        # We want it to be closer to 0.
        rew -= 0.005 * dist_xy
        rew -= 0.005 * abs(dist_z)

        # 2. SHAPING: Velocity (Minimize speed, especially downward)
        # We want vz to be small, but we allow some speed when high up.
        # This is a "Descent Guidance" reward.
        vel_pen = 0.01 * np.linalg.norm(state['vel'])
        rew -= vel_pen

        # 3. SHAPING: Tilt (Keep upright)
        # Tilt is in degrees. 0 is perfect.
        rew -= 0.02 * abs(state['tilt'])

        # 4. SHAPING: Action Regularization (Save Fuel / Smooth Control)
        # Penalize large control inputs to encourage efficiency
        rew -= 0.001 * np.sum(np.square(action))

        # 5. TERMINAL REWARDS
        if terminated:
            if success:
                # HUGE Bonus for landing
                rew += 100.0
                # Extra bonus for fuel remaining
                rew += 10.0 * (state['fuel_mass'] / self.cfg['start_fuel'])
                info['outcome'] = 'SUCCESS'
            else:
                # CRASH PENALTY
                # Scale penalty by how hard it hit
                crash_intensity = np.linalg.norm(state['vel']) + state['tilt']
                rew -= (20.0 + 0.5 * crash_intensity)
                info['outcome'] = 'CRASH'
        
        elif truncated:
            # Penalty for running out of time or flying away
            rew -= 10.0
            info['outcome'] = 'TIMEOUT/FLYAWAY'

        return rew, info