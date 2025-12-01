# rocket_env/rocket_landing_env.py

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "tintin_thrust.xml")


class RocketLandingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        # ------------------------
        # Load MuJoCo Model
        # ------------------------
        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)

        # ==========================================================
        #  IDs — MUST match XML exactly
        # ==========================================================
        self.rocket_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )

        # free joint (ball_free)
        self.free_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )
        self.qpos_adr = self.model.jnt_qposadr[self.free_joint_id]
        self.qvel_adr = self.model.jnt_dofadr[self.free_joint_id]

        # gimbal joints: thruster_yaw + thruster_pitch
        self.yaw_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "thruster_yaw"
        )
        self.pitch_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "thruster_pitch"
        )

        # actuators: yaw_motor, pitch_motor, thrust
        self.yaw_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_motor"
        )
        self.pitch_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor"
        )
        self.thrust_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust"
        )

        # ------------------------
        # Physics properties
        # ------------------------
        self.DRY_MASS = self.model.body_mass[self.rocket_bid]
        self.START_FUEL = 10.0
        self.fuel_mass = self.START_FUEL
        self.total_mass = self.DRY_MASS + self.START_FUEL
        self.initial_total_mass = self.total_mass
        self.orig_inertia = self.model.body_inertia[self.rocket_bid].copy()

        self.ISP = 250.0
        self.G0 = 9.81
        self.DT = self.model.opt.timestep

        # controls
        self.MAX_THRUST = 2200.0
        self.MAX_GIMBAL = np.deg2rad(20.0)

        # landing target
        self.TARGET_Z = 0.5

        # max episode steps
        self.max_steps = 2000
        self.step_count = 0

        # ------------------------
        # Gym spaces
        # ------------------------
        obs_high = np.array(
            [100, 100, 100,   # pos
             100, 100, 100,   # vel
             self.START_FUEL],  # fuel
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # action = [thrust_norm, yaw_norm, pitch_norm]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ------------------------
        # Rendering
        # ------------------------
        self.render_mode = render_mode
        self.viewer = None

        mujoco.mj_forward(self.model, self.data)

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def _get_pos(self):
        return self.data.xpos[self.rocket_bid].copy()

    def _get_vel(self):
        v_body = self.data.cvel[self.rocket_bid][3:]
        R = self.data.xmat[self.rocket_bid].reshape(3, 3)
        return R @ v_body

    def _get_obs(self):
        pos = self._get_pos()
        vel = self._get_vel()
        return np.array([
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            self.fuel_mass
        ], dtype=np.float32)

    # -----------------------------------------------------
    # RESET
    # -----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        mujoco.mj_resetData(self.model, self.data)

        # reset state
        self.data.qpos[self.qpos_adr + 0] = 0
        self.data.qpos[self.qpos_adr + 1] = 0
        self.data.qpos[self.qpos_adr + 2] = 20.0

        # orientation quaternion
        self.data.qpos[self.qpos_adr + 3 : self.qpos_adr + 7] = [1, 0, 0, 0]

        # velocities
        self.data.qvel[self.qvel_adr : self.qvel_adr + 6] = 0
        self.data.qvel[self.qvel_adr + 2] = -0.5

        # fuel/mass
        self.fuel_mass = self.START_FUEL
        self.total_mass = self.DRY_MASS + self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.total_mass
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia.copy()

        mujoco.mj_forward(self.model, self.data)

        # reset viewer
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        return self._get_obs(), {}

    # -----------------------------------------------------
    # STEP
    # -----------------------------------------------------
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1, 1)

        thrust = (action[0] + 1) * 0.5 * self.MAX_THRUST
        yaw    = action[1] * self.MAX_GIMBAL
        pitch  = action[2] * self.MAX_GIMBAL

        # fuel burn
        if self.fuel_mass > 0:
            mdot = -thrust / (self.ISP * self.G0)
            self.fuel_mass = max(self.fuel_mass + mdot * self.DT, 0)
        else:
            thrust = 0

        # mass
        self.total_mass = self.DRY_MASS + self.fuel_mass
        ratio = self.total_mass / self.initial_total_mass

        self.model.body_mass[self.rocket_bid] = self.total_mass
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia * ratio

        # apply actions
        self.data.ctrl[self.thrust_act] = thrust
        self.data.ctrl[self.yaw_act]    = yaw
        self.data.ctrl[self.pitch_act]  = pitch

        mujoco.mj_step(self.model, self.data)

        # state
        pos = self._get_pos()
        vel = self._get_vel()
        x, y, z = pos
        vx, vy, vz = vel

        obs = self._get_obs()

        # reward (simple shaping)
        pos_err = np.linalg.norm([x, y, z - self.TARGET_Z])
        vel_err = np.linalg.norm([vx, vy, vz])
        rew = -pos_err - 0.1 * vel_err

        terminated = False
        truncated = False

        # crash
        if z < 0:
            rew -= 100
            terminated = True

        # success landing
        if (0 < z < 1.0) and (pos_err < 0.5) and (vel_err < 0.5):
            rew += 200
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        return obs, rew, terminated, truncated, {}

    # -----------------------------------------------------
    # RENDER (for MuJoCo 3.1+)
    # -----------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            from mujoco_python_viewer import Viewer
            if self.viewer is None:
                self.viewer = Viewer(self.model, self.data)
            self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
