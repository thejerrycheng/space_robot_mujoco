# keyboard_hold_printer_fast.py
# Ultra-responsive hold printer for Arrow keys + WASD on macOS.
# Requires: pip install pynput
# Exit with ESC or Ctrl+C.

import time
import threading
import sys
from pynput import keyboard

SCAN_HZ     = 60.0    # how often we scan pressed keys (higher = more responsive)
REPEAT_HZ   = 25.0    # how often each held key prints (per-key rate)
INITIAL_BURST = True  # print immediately on press

_pressed = {}          # key_id -> last_print_time (perf_counter)
_lock = threading.Lock()
stop_event = threading.Event()

KEY_TEXT = {
    keyboard.Key.up: "UP",
    keyboard.Key.down: "DOWN",
    keyboard.Key.left: "LEFT",
    keyboard.Key.right: "RIGHT",
}
LETTER_TEXT = {'w': "W", 'a': "A", 's': "S", 'd': "D"}

def key_id_from(key):
    """Return a stable id + label for arrows and WASD; otherwise None."""
    if isinstance(key, keyboard.KeyCode) and key.char:
        ch = key.char.lower()
        if ch in LETTER_TEXT:
            return ('char', ch), LETTER_TEXT[ch]
    elif key in KEY_TEXT:
        return ('key', key), KEY_TEXT[key]
    return None, None

def on_press(key):
    kid, label = key_id_from(key)
    if kid is None:
        return
    now = time.perf_counter()
    with _lock:
        # If not already pressed, add and optionally print immediately.
        if kid not in _pressed:
            _pressed[kid] = -1.0  # force an immediate print below
            if INITIAL_BURST:
                _pressed[kid] = now
                sys.stdout.write(label + "\n")
                sys.stdout.flush()

def on_release(key):
    kid, _ = key_id_from(key)
    if kid is not None:
        with _lock:
            _pressed.pop(kid, None)
    if key == keyboard.Key.esc:
        stop_event.set()
        return False  # stop listener

def scan_loop():
    scan_dt = 1.0 / SCAN_HZ
    repeat_dt = 1.0 / REPEAT_HZ
    while not stop_event.is_set():
        now = time.perf_counter()
        to_print = []
        with _lock:
            for kid, last_t in list(_pressed.items()): 
                # Determine label again (safe & cheap)
                if kid[0] == 'char':
                    label = LETTER_TEXT[kid[1]]
                else:
                    label = KEY_TEXT[kid[1]]

                # First cycle after press or enough time passed -> print
                if last_t < 0 or (now - last_t) >= repeat_dt:
                    to_print.append((kid, label))
            # Update timestamps after deciding what to print
            for kid, _ in to_print:
                _pressed[kid] = now

        # Print outside the lock for minimal latency
        if to_print:
            # keep stable order for combined holds
            line = " + ".join(label for _, label in sorted(to_print, key=lambda x: x[1]))
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

        # tiny sleep to keep CPU in check but still very responsive
        time.sleep(scan_dt)

def main():
    print("Hold Arrow keys or W/A/S/D for continuous prints. ESC to quit.")
    t = threading.Thread(target=scan_loop, daemon=True)
    t.start()
    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        t.join(timeout=1.0)

if __name__ == "__main__":
    main()
