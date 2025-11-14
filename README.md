space_robot/
├─ pyproject.toml # deps, tooling (or requirements.txt)
├─ README.md
├─ configs/ # yaml / json experiment configs
├─ data/ # datasets / initial states / checkpoints
├─ logs/ # tensorboard / wandb / eval logs
├─ assets/
│ ├─ mjcf/ # MJCF XML scenes (entrypoint .xml files)
│ ├─ usd/ # USD assets (your rocket.usd lives here)
│ ├─ meshes/ # OBJ / STL meshes
│ └─ textures/ # PNG / JPG textures
├─ space_robot/ # Python package (importable)
│ ├─ **init**.py
│ ├─ envs/
│ │ ├─ **init**.py
│ │ └─ rocket_env.py # Gymnasium-style env
│ ├─ controllers/
│ │ ├─ **init**.py
│ │ └─ pid.py # e.g., simple PID / guidance law
│ ├─ models/
│ │ ├─ **init**.py
│ │ └─ sac.py # RL policy/algorithm code
│ ├─ utils/ # (rename your 'utilis' -> 'utils')
│ │ ├─ **init**.py
│ │ └─ paths.py # helpers for asset paths, etc.
│ └─ wrappers/ # optional Gym wrappers
├─ scripts/
│ ├─ run_viewer.py # quick visual run of the MJCF scene
│ ├─ train_rl.py # training loop that uses RocketEnv
│ └─ collect_rollouts.py
├─ tests/
│ └─ test_env.py
└─ notebooks/
└─ sandbox.ipynb
