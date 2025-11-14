#!/usr/bin/env python3
import os, time, math, numpy as np
from pynput import keyboard
import mujoco, mujoco.viewer
from mujoco import mjtGridPos

from envs.rocket_goal_env import RocketGoalEnv, GIMBAL_MAX_DEG

THROTTLE_STEP  = 0.02
GIMBAL_STEP_DEG = 0.5

class Teleop:
    def __init__(self, gimbal_max_deg=GIMBAL_MAX_DEG):
        self.throttle = 0.0; self.pitch = 0.0; self.yaw = 0.0
        self.lim = math.radians(gimbal_max_deg)
    def clamp(self):
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        self.pitch = float(np.clip(self.pitch, -self.lim, self.lim))
        self.yaw   = float(np.clip(self.yaw,   -self.lim, self.lim))

def start_keyboard_listener(tele: Teleop, reset_cb, goal_cb):
    def on_press(key):
        try:
            if key.char in ('w','W'): tele.throttle += THROTTLE_STEP
            elif key.char in ('s','S'): tele.throttle -= THROTTLE_STEP
            elif key.char in ('x','X'): tele.throttle  = 0.0
            elif key.char in ('z','Z'): tele.pitch = tele.yaw = 0.0
            elif key.char in ('r','R'): reset_cb()
            elif key.char in ('g','G'): goal_cb()
        except AttributeError:
            if key == keyboard.Key.up:    tele.pitch += math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.down:  tele.pitch -= math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.left:  tele.yaw   += math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.right: tele.yaw   -= math.radians(GIMBAL_STEP_DEG)
        tele.clamp(); return True
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True; listener.start(); return listener

if __name__ == "__main__":
    xml = os.path.join(os.path.dirname(__file__), "..", "assets", "mjcf", "world.xml")
    env = RocketGoalEnv(xml_path=os.path.abspath(xml), frame_skip=5, goal_radius=2.0,
                        time_limit_s=180.0, init_z_range=(60,120), init_down_vz_range=(-12,-6))
    tele = Teleop()

    def do_reset():
        obs, info = env.reset()
        print(f"[reset] goal=({info['goal_xy'][0]:.1f},{info['goal_xy'][1]:.1f})  start z≈{info['z']:.1f} m")

    def randomize_goal():
        env.set_goal(None)
        print(f"[goal] new goal at ({env.goal_xy[0]:.1f},{env.goal_xy[1]:.1f})")

    do_reset()
    start_keyboard_listener(tele, do_reset, randomize_goal)

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        v.cam.lookat[:] = env.data.xipos[env.h['rocket_bid']]
        v.cam.distance   = 180.0; v.cam.azimuth = 130.0; v.cam.elevation = -35.0

        last_print = 0.0
        while v.is_running():
            step_start = time.time()
            a = np.array([tele.throttle, tele.pitch, tele.yaw], dtype=np.float32)
            obs, rew, term, trunc, info = env.step(a)

            # simple overlay HUD
            hud = (f"t={info['time']:6.2f}s   goal=({info['goal_xy'][0]:.1f},{info['goal_xy'][1]:.1f})   "
                   f"dist_xy={info['dist_to_goal_xy']:5.2f} m   z={info['z']:5.2f} m   "
                   f"vxy={info['vxy']:4.2f} m/s  vz={info['vz']:5.2f} m/s   "
                   f"T={info['thrust_N']/1000:6.1f} kN")
            try:
                v.add_overlay(mjtGridPos.mjGRID_TOPLEFT, "", hud)
            except Exception:
                pass  # older mujoco.viewer without add_overlay

            if time.time() - last_print > 0.2:
                print(hud); last_print = time.time()

            if term or trunc:
                print(f"[episode end] reason={info['reason']} success={info['success']}")
                do_reset()

            v.sync()
            dt = env.model.opt.timestep - (time.time() - step_start)
            if dt > 0: time.sleep(dt)
