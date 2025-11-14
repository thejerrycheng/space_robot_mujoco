import time
import mujoco, mujoco.viewer

xml = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 2">
      <freejoint/>
      <geom type="sphere" size="0.05" density="7800" rgba="0.85 0.25 0.25 1"/>
    </body>
    <geom type="plane" size="5 5 0.1" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as v:
    # interactive camera
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 3.0; v.cam.azimuth = 135; v.cam.elevation = -15
    t0 = time.perf_counter()
    next_t = t0
    last_log = t0
    while v.is_running():
        # step once in sim
        mujoco.mj_step(model, data)
        # schedule next step exactly dt later in wall-clock
        next_t += dt
        # sleep to keep real-time rate (with small clamp)
        nap = next_t - time.perf_counter()
        if nap > 0: time.sleep(nap)
        # log every ~0.5 s
        now = time.perf_counter()
        if now - last_log >= 0.5:
            wall_elapsed = now - t0
            sim_time = data.time
            drift = sim_time - wall_elapsed
            print(f"sim={sim_time:6.3f}s  wall={wall_elapsed:6.3f}s  drift={drift:+.4f}s")
            last_log = now
        v.sync()
    print("Sim closed.")