import mujoco
print("python package:", mujoco.__version__)           # e.g. '3.1.5'
print("engine (C lib):", mujoco.mj_versionString())     # e.g. '3.1.5'
print("engine int:", mujoco.mj_version())               # e.g. 315
