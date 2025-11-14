# scripts/play_teleop.py
#!/usr/bin/env python3
import os, time, math, numpy as np
from pynput import keyboard
import mujoco
import mujoco.viewer

from envs.rocket_env import RocketEnv, GIMBAL_MAX_DEG

THROTTLE_STEP  = 0.02
GIMBAL_STEP_DEG = 0.5

class Teleop:
    def __init__(self, gimbal_max_deg=GIMBAL_MAX_DEG):
        self.throttle = 0.0
        self.pitch = 0.0
        self.yaw   = 0.0
        self.lim = math.radians(gimbal_max_deg)
    def clamp(self):
        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        self.pitch = float(np.clip(self.pitch, -self.lim, self.lim))
        self.yaw   = float(np.clip(self.yaw,   -self.lim, self.lim))

def start_keyboard_listener(tele: Teleop, reset_cb):
    def on_press(key):
        try:
            if key.char in ('w','W'): tele.throttle += THROTTLE_STEP
            elif key.char in ('s','S'): tele.throttle -= THROTTLE_STEP
            elif key.char in ('x','X'): tele.throttle  = 0.0
            elif key.char in ('z','Z'): tele.pitch = tele.yaw = 0.0
            elif key.char in ('r','R'): reset_cb()
        except AttributeError:
            if key == keyboard.Key.up:    tele.pitch += math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.down:  tele.pitch -= math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.left:  tele.yaw   += math.radians(GIMBAL_STEP_DEG)
            if key == keyboard.Key.right: tele.yaw   -= math.radians(GIMBAL_STEP_DEG)
        tele.clamp(); return True
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener

if __name__ == "__main__":
    xml = os.path.join(os.path.dirname(__file__), "..", "assets", "mjcf", "world.xml")
    env = RocketEnv(xml_path=os.path.abspath(xml), frame_skip=5, time_limit_s=120.0)
    tele = Teleop()

    def do_reset():
        obs, info = env.reset()
        print("[reset] fuel full, pose randomized.")

    do_reset()
    start_keyboard_listener(tele, do_reset)

    # Use the env's model/data inside the viewer loop
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        last_print = 0.0
        while v.is_running():
            step_start = time.time()

            action = np.array([tele.throttle, tele.pitch, tele.yaw], dtype=np.float32)
            obs, rew, term, trunc, info = env.step(action)

            # simple HUD
            if time.time() - last_print > 0.1:
                print(f"t={env.data.time:6.2f}s  thr={tele.throttle:0.2f}  "
                      f"pitch={math.degrees(tele.pitch):+5.1f}°  yaw={math.degrees(tele.yaw):+5.1f}°  "
                      f"T={info['thrust_N']/1000:6.1f} kN  mass={info['mass']:.1f} kg  AoA={info['aoa_deg']:5.1f}°")
                last_print = time.time()

            if term or trunc:
                do_reset()

            v.sync()
            dt = env.model.opt.timestep - (time.time() - step_start)
            if dt > 0: time.sleep(dt)
