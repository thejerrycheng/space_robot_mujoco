# Space Robot Project

A clean, modular template for simulating, controlling, and training a rocket-landing robot using MuJoCo/MJCF or USD assets. This repository provides:

- A fully importable Python package (`space_robot/`).
- Structured experiment configs.
- RL training pipelines.
- Simulation viewers and rollout collectors.
- Organized assets (MJCF, USD, meshes, textures).
- Testing and notebooks for iterative development.

---

## 📁 Repository Structure

Below is an explanation of each top-level directory and key file so you know exactly where everything lives.

```
space_robot/
├─ pyproject.toml
├─ README.md
├─ configs/
├─ data/
├─ logs/
├─ assets/
│  ├─ mjcf/
│  ├─ usd/
│  ├─ meshes/
│  └─ textures/
├─ space_robot/
│  ├─ __init__.py
│  ├─ envs/
│  ├─ controllers/
│  ├─ models/
│  ├─ utils/
│  └─ wrappers/
├─ scripts/
├─ tests/
└─ notebooks/
```

---

## 🔧 Top-Level Files

### **`pyproject.toml`**

Project configuration and dependencies. Contains:

- package metadata
- dependency list (instead of `requirements.txt`)
- tools configuration (Black, Ruff, Mypy, etc.)

### **`README.md`**

The file you're reading. Describes the project layout, usage, and workflows.

---

## ⚙️ `configs/`

Experiment configuration files in **YAML** or **JSON** used by training/eval scripts. Typical contents:

- environment parameters
- reward shaping settings
- RL algorithm hyperparameters
- asset paths or scenario definitions

Useful for keeping experiments reproducible.

---

## 📂 `data/`

Holds non-source data such as:

- datasets
- initial states
- saved checkpoints (policies, replay buffers)
- rollout/trajectory collections

This directory is usually **gitignored** except for small example datasets.

---

## 🧪 `logs/`

All runtime-generated logs:

- TensorBoard logs
- wandb logs
- evaluation results
- diagnostic CSVs

Automatically populated during training and rollouts.

---

## 🎨 `assets/`

Simulation and rendering assets.

### `assets/mjcf/`

MJCF XML scene definitions. The **main simulation entrypoint** (e.g., `rocket.xml`) lives here.

### `assets/usd/`

USD files for Omniverse or other USD-based simulators. Your main `rocket.usd` belongs here.

### `assets/meshes/`

Geometry files:

- `.obj`
- `.stl`
- other mesh formats

### `assets/textures/`

Texture maps used by MJCF/USD or rendering:

- `.png`
- `.jpg`

---

## 🐍 `space_robot/` (Python package)

This is the **core library**. Importable as:

```python
import space_robot
```

### `envs/`

Gymnasium-style simulation environments.

- `rocket_env.py` — primary environment implementing step(), reset(), reward logic, etc.

### `controllers/`

Control algorithms / guidance laws.

- `pid.py` — simple PID controller or baseline guidance module.

### `models/`

Machine learning or RL algorithms.

- `sac.py` — Soft Actor-Critic implementation or wrapper for training.

### `utils/`

Utility helpers.

- `paths.py` — robust helpers for finding asset directories, resolving paths, etc.

### `wrappers/`

Optional Gym wrappers for transforming observations/actions or adding logging.

---

## ▶️ `scripts/`

Entry-point scripts for running the project.

- **`run_viewer.py`** — launches a quick MJCF viewer to visualize the rocket scene.
- **`train_rl.py`** — full RL training loop using `RocketEnv` and models.
- **`collect_rollouts.py`** — captures trajectories for offline RL, debugging, or imitation learning.

These scripts typically load config files from `configs/`.

---

## 🧪 `tests/`

Unit tests to keep environments and logic correct.

- `test_env.py` — basic tests for the environment API (reset/step, shapes, determinism).

Run tests via:

```bash
pytest
```

---

## 📓 `notebooks/`

Jupyter notebooks for interactive exploration.

- `sandbox.ipynb` — quick experiment pad, plotting, debugging.

---

## 🚀 Quick Start

### Install the package

```bash
pip install -e .
```

### Run the viewer

```bash
python scripts/run_viewer.py
```

### Train an RL policy

```bash
python scripts/train_rl.py --config configs/sac_rocket.yaml
```

---

## 🛰️ Purpose

This project gives you a clean foundation for:

- building physically realistic rocket landing simulations,
- testing controllers and RL algorithms,
- running reproducible experiments,
- visualizing and collecting trajectories.

Feel free to extend the environments, swap assets, or integrate new algorithms.

---

<!-- ## 📬 Questions / Improvements

Open an issue or reach out if you want help structuring new features or reorganizing subsystems.

Happy launching! 🚀 -->
