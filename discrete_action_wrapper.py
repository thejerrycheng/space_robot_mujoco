import gymnasium as gym
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts a continuous action space env into a discrete one.
    User controls:
      - number of bins for thrust, pitch, roll
      - min/max range for each action dimension

    Example:
        env = DiscreteActionWrapper(
            env,
            thrust_bins=5, thrust_range=(0, 1),
            pitch_bins=7, pitch_range=(-0.2, 0.2),
            roll_bins=7,  roll_range=(-0.2, 0.2),
        )
    """

    def __init__(
        self,
        env,
        thrust_bins=3,
        pitch_bins=3,
        roll_bins=3,
        thrust_range=(0.0, 1.0),
        pitch_range=(-0.1, 0.1),
        roll_range=(-0.1, 0.1),
    ):
        super().__init__(env)

        # Store config
        self.thrust_bins = thrust_bins
        self.pitch_bins = pitch_bins
        self.roll_bins = roll_bins

        self.thrust_range = thrust_range
        self.pitch_range = pitch_range
        self.roll_range = roll_range

        # --- Generate discrete values ---
        self.thrust_vals = np.linspace(thrust_range[0], thrust_range[1], thrust_bins)
        self.pitch_vals  = np.linspace(pitch_range[0],  pitch_range[1],  pitch_bins)
        self.roll_vals   = np.linspace(roll_range[0],   roll_range[1],   roll_bins)

        # --- Build discrete action table ---
        self.action_list = []
        for t in self.thrust_vals:
            for p in self.pitch_vals:
                for r in self.roll_vals:
                    self.action_list.append(np.array([t, p, r], dtype=np.float32))

        # NEW discrete action space size
        self.action_space = gym.spaces.Discrete(len(self.action_list))

        print(f"[DiscreteActionWrapper] Actions =", len(self.action_list))
        print(f"  thrust_bins={thrust_bins}, pitch_bins={pitch_bins}, roll_bins={roll_bins}")
        print(f"  sample thrust vals: {self.thrust_vals}")
        print(f"  sample pitch vals:  {self.pitch_vals}")
        print(f"  sample roll vals:   {self.roll_vals}")

    def action(self, act_idx):
        return self.action_list[act_idx]
