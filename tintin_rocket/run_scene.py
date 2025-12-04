# run_scene_safe.py
# Fixes mac "trace trap" by keeping GLFW + mj_step on the main thread only.
# Requires: pip install pynput

import math, time, threading
from pathlib import Path
import mujoco, mujoco.viewer
from pynput import keyboard

# -------------------------- config --------------------------
class config:
    SCENE = str(Path(__file__).with_name("scene.xml"))
    SIMULATE_DT = 0.002
    PRINT_HZ    = 50

    MAX_FX_FY = 1.0
    MAX_FZ    = 1000000.0

    ANGLE_STEP_DEG  = 1.0
    ANGLE_LIMIT_DEG = 10.0
    FZ_STEP         = 5000.0

    # keyboard responsiveness
    SCAN_HZ   = 120.0
    REPEAT_HZ = 30.0
    INITIAL_BURST = True

# ------------------------- shared state ----------------------
lock = threading.Lock()
paused = False
quit_flag = False

pitch_cmd = 0.0  # +Y tilt (rad)
roll_cmd  = 0.0  # +X tilt (rad)
Fz_cmd    = 0.0  # N

# ------------------------ load mujoco ------------------------
m = mujoco.MjModel.from_xml_path(config.SCENE)
d = mujoco.MjData(m)
m.opt.timestep = config.SIMULATE_DT

def aid(name):
    idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx == -1:
        raise RuntimeError(f"Actuator '{name}' not found.")
    return idx
AIDX = {"Fx": aid("thrust_Fx"), "Fy": aid("thrust_Fy"), "Fz": aid("thrust_Fz")}

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

# ---------------------- MuJoCo in-window keys ----------------
def KeyCallback(keycode, _mod=None):
    global paused, quit_flag, pitch_cmd, roll_cmd, Fz_cmd
    K = mujoco.viewer
    if keycode == K.GLFW_KEY_ESCAPE: quit_flag = True; return
    if keycode == K.GLFW_KEY_P:      paused = not paused; print(f"[keys] paused={paused}"); return
    if keycode == K.GLFW_KEY_SPACE:  # zero tilt
        with lock: pitch_cmd = 0.0; roll_cmd = 0.0
        print("[keys] zero tilt"); return
    if keycode == K.GLFW_KEY_0:      # zero all
        with lock: Fz_cmd = 0.0; pitch_cmd = 0.0; roll_cmd = 0.0
        print("[keys] zero all"); return

# ---------------------- fast keyboard (pynput) ----------------
_pressed = {}            # key-id -> last_step_time
_klock   = threading.Lock()
stop_keys = threading.Event()

def key_id_from(key):
    if isinstance(key, keyboard.KeyCode) and key.char:
        ch = key.char.lower()
        if ch in ('w','a','s','d'): return ('char', ch)
    elif key in (keyboard.Key.up, keyboard.Key.down):
        return ('key', key)
    return None

def apply_single_step_for_key(kid):
    global pitch_cmd, roll_cmd, Fz_cmd
    step  = math.radians(config.ANGLE_STEP_DEG)
    limit = math.radians(config.ANGLE_LIMIT_DEG)
    with lock:
        if kid[0] == 'char':
            ch = kid[1]
            if ch == 'w': pitch_cmd = clamp(pitch_cmd + step, -limit, limit)
            if ch == 's': pitch_cmd = clamp(pitch_cmd - step, -limit, limit)
            if ch == 'a': roll_cmd  = clamp(roll_cmd  - step, -limit, limit)
            if ch == 'd': roll_cmd  = clamp(roll_cmd  + step, -limit, limit)
        else:
            if kid[1] == keyboard.Key.up:
                Fz_cmd = clamp(Fz_cmd + config.FZ_STEP, 0.0, config.MAX_FZ)
            if kid[1] == keyboard.Key.down:
                Fz_cmd = clamp(Fz_cmd - config.FZ_STEP, 0.0, config.MAX_FZ)

def on_press(key):
    kid = key_id_from(key)
    if kid is None:
        if key == keyboard.Key.esc:
            stop_keys.set(); return False
        return
    now = time.perf_counter()
    with _klock:
        if kid not in _pressed:
            _pressed[kid] = -1.0
            if config.INITIAL_BURST:
                apply_single_step_for_key(kid)
                _pressed[kid] = now

def on_release(key):
    kid = key_id_from(key)
    if kid is not None:
        with _klock:
            _pressed.pop(kid, None)

def KeyScanThread():
    scan_dt   = 1.0 / config.SCAN_HZ
    repeat_dt = 1.0 / config.REPEAT_HZ
    while not stop_keys.is_set() and not quit_flag:
        now = time.perf_counter()
        to_update = []
        with _klock:
            for kid, last_t in list(_pressed.items()):
                if last_t < 0 or (now - last_t) >= repeat_dt:
                    to_update.append(kid)
            for kid in to_update:
                _pressed[kid] = now
        for kid in to_update:
            apply_single_step_for_key(kid)
        time.sleep(scan_dt)

# ----------------------------- main --------------------------
if __name__ == "__main__":
    # Start global keyboard machinery (background threads)
    keyscan_thread = threading.Thread(target=KeyScanThread, daemon=True)
    keyscan_thread.start()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Create viewer on the MAIN thread and keep all GL + stepping here
    with mujoco.viewer.launch_passive(m, d, key_callback=KeyCallback) as viewer:
        print("[info] Viewer started. Hold WASD / ↑↓. P=pause, SPACE=zero tilt, 0=zero all, ESC=quit.")
        print_dt = 1.0 / config.PRINT_HZ
        next_print = 0.0

        while viewer.is_running() and not quit_flag:
            step_start = time.perf_counter()

            # read commands & write ctrl, then step — all on main thread
            with lock:
                Fx = clamp(Fz_cmd * math.tan(roll_cmd),  -config.MAX_FX_FY, config.MAX_FX_FY)
                Fy = clamp(Fz_cmd * math.tan(pitch_cmd), -config.MAX_FX_FY, config.MAX_FX_FY)
                Fz = clamp(Fz_cmd, 0.0, config.MAX_FZ)
                d.ctrl[AIDX["Fx"]] = Fx
                d.ctrl[AIDX["Fy"]] = Fy
                d.ctrl[AIDX["Fz"]] = Fz

            if not paused:
                mujoco.mj_step(m, d)

            # simple logging (avoid heavy prints every step)
            if d.time >= next_print:
                with lock:
                    print(f"t={d.time:6.3f}s  Fz={Fz:6.1f}  roll={math.degrees(roll_cmd):+5.1f}°  "
                          f"pitch={math.degrees(pitch_cmd):+5.1f}°  Fx={Fx:+7.1f} Fy={Fy:+7.1f}")
                next_print += print_dt

            # render; keep on main thread
            viewer.sync()

            # pacing toward real-time
            remaining = m.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    # clean shutdown
    quit_flag = True
    stop_keys.set()
    listener.stop()
    keyscan_thread.join(timeout=1.0)
    print("[info] Clean exit.")
