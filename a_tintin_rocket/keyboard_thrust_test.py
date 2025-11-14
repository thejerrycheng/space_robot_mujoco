#!/usr/bin/env python3
"""
Keyboard teleoperation for a thrust-controlled cylinder with gimbal actuators.
Controls:
  ↑ / ↓ : increase / decrease thrust
  W / S : pitch up / down (±60°)
  A / D : roll left / right (±60°)
  R     : reset pitch and roll to 0
  ESC   : quit
"""

import math
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard

# ---------------------- Config ----------------------
THRUST_MAX = 100.0
THRUST_MIN = 0.0
ANGLE_MAX_DEG = 60.0
ANGLE_STEP_DEG = 2.0
THRUST_STEP = 2.0

# ---------------------- XML Model ----------------------
XML = """
<mujoco model="thrust_cylinder_actuated">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <default>
    <joint damping="0.05" armature="0.001"/>
    <geom friction="1 0.1 0.01"/>
    <motor ctrllimited="true"/>
  </default>

  <worldbody>
    <!-- Ground plane -->
    <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Cylinder body -->
    <body name="cylinder" pos="0 0 1">
      <freejoint/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.026 0.026 0.013"/>
      <geom type="cylinder" size="0.1 0.5" rgba="0.2 0.6 0.8 1"/>

      <!-- Gimbaled nozzle assembly -->
      <body name="gimbal_pitch" pos="0 0 -0.5">
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.026 0.026 0.013"/>
        <joint name="pitch_joint" type="hinge" axis="1 0 0" range="-60 60"/>
        <body name="gimbal_roll" pos="0 0 0">
            <inertial pos="0 0 0" mass="1.0" diaginertia="0.026 0.026 0.013"/>
            <joint name="roll_joint" type="hinge" axis="0 1 0" range="-60 60"/>
            <site name="thrust_site" pos="0 0 0" size="0.02" rgba="1 0 0 0.8"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <general name="thruster" site="thrust_site"
             gainprm="0 0 1 0 0 0"
             biasprm="0 0 0 0 0 0"
             ctrlrange="0 100"/>
    <motor name="pitch_motor" joint="pitch_joint" ctrlrange="-60 60" gear="1"/>
    <motor name="roll_motor" joint="roll_joint" ctrlrange="-60 60" gear="1"/>
  </actuator>
</mujoco>
"""
# ---------------------- Initialize ----------------------
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

state = {"thrust": 0.0, "pitch": 0.0, "roll": 0.0, "running": True}

# ---------------------- Key Control ----------------------
def on_press(key):
    try:
        if key == keyboard.Key.up:
            state["thrust"] = min(state["thrust"] + THRUST_STEP, THRUST_MAX)
            print(f"\n[Key] Increase thrust to {state['thrust']:.1f}N")
        elif key == keyboard.Key.down:
            state["thrust"] = max(state["thrust"] - THRUST_STEP, THRUST_MIN)
            print(f"\n[Key] Decrease thrust to {state['thrust']:.1f}N")
        elif key.char.lower() == 'w':
            state["pitch"] = min(state["pitch"] + ANGLE_STEP_DEG, ANGLE_MAX_DEG)
            print(f"\n[Key] Increase pitch to {state['pitch']:.1f}°")
        elif key.char.lower() == 's':
            state["pitch"] = max(state["pitch"] - ANGLE_STEP_DEG, -ANGLE_MAX_DEG)
            print(f"\n[Key] Decrease pitch to {state['pitch']:.1f}°")
        elif key.char.lower() == 'a':
            state["roll"] = min(state["roll"] + ANGLE_STEP_DEG, ANGLE_MAX_DEG)
            print(f"\n[Key] Increase roll to {state['roll']:.1f}°")
        elif key.char.lower() == 'd':
            state["roll"] = max(state["roll"] - ANGLE_STEP_DEG, -ANGLE_MAX_DEG)
            print(f"\n[Key] Decrease roll to {state['roll']:.1f}°")
        elif key.char.lower() == 'r':
            state["pitch"], state["roll"] = 0.0, 0.0
            print(f"\n[Key] Reset pitch and roll to 0°")
    except AttributeError:
        if key == keyboard.Key.esc:
            state["running"] = False
            return False

listener = keyboard.Listener(on_press=on_press)
listener.start()

# ---------------------- Simulation Loop ----------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and state["running"]:
        # Send control signals to actuators
        data.ctrl[0] = state["thrust"]
        data.ctrl[1] = state["pitch"]
        data.ctrl[2] = state["roll"]

        mujoco.mj_step(model, data)
        viewer.sync()

        print(f"\rThrust={state['thrust']:.1f}N  Pitch={state['pitch']:.1f}°  Roll={state['roll']:.1f}°", end="")

print("\nSimulation ended.")
