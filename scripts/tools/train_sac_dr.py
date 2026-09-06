#!/usr/bin/env python3
"""
Train SAC on the domain-randomised 6-DoF MuJoCo lander
(rocket_env/rocket_dr_env.py) and checkpoint on a *staged* schedule so the
controller's progress can be shown, not just its final score.

    python scripts/tools/train_sac_dr.py --steps 3000000 --envs 12

Checkpoints are taken at a log-spaced set of environment steps
(1k, 5k, 20k, 50k, 100k, 250k, 500k, 1M, 2M, 3M ...), which is what the
"trained for 1 episode / 100 episodes / ..." slides show. Each one is written to
runs/<name>/stage_<steps>.zip and can be evaluated by eval_progress.py.

Domain randomisation is always on: every episode resamples dry mass, propellant
load, specific impulse, thrust authority, a persistent gimbal misalignment, a
constant lateral disturbance, and observation noise, on top of the initial-state
curriculum.
"""
import argparse, json, os, sys, time

# Tiny MLPs on this machine are ~14x faster single-threaded: with 8 OpenMP
# threads the per-op launch overhead dominates a 17 -> 256 -> 256 -> 6 network
# and SAC drops from ~500 to ~37 gradient steps per second.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
torch.set_num_threads(1)

sys.modules.setdefault("tensorflow", None)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from stable_baselines3 import SAC                                  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback        # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor  # noqa: E402
from rocket_env.rocket_dr_env import RocketLandingDREnv            # noqa: E402

# train_freq is counted per environment, so with n_envs = 10 a train_freq of 32
# collects 320 environment steps per training call.  gradient_steps must scale
# with that: an earlier run used 8, i.e. an update-to-data ratio of 0.025, and
# SAC had managed only ~6,000 gradient steps by 250k environment steps.
# Stages are in *control* steps (20 Hz), not physics steps.
STAGES = [1_000, 5_000, 20_000, 50_000, 100_000, 200_000, 350_000,
          500_000, 750_000, 1_000_000, 1_500_000]


def make_env(rank, dr=True, curriculum=True):
    def _f():
        return RocketLandingDREnv(seed=1000 + rank, domain_randomize=dr,
                                  curriculum=curriculum)
    return _f


class StageCheckpoint(BaseCallback):
    """Save at each stage in STAGES and log the rolling outcome mix."""

    def __init__(self, out_dir, stages, verbose=0):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.stages = sorted(stages)
        self.next_i = 0
        self.recent = []
        self.log_rows = []
        self.t0 = time.time()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "outcome" in info and info["outcome"]:
                self.recent.append((info["outcome"], bool(info.get("success"))))
        if len(self.recent) > 400:
            self.recent = self.recent[-400:]

        while self.next_i < len(self.stages) and self.num_timesteps >= self.stages[self.next_i]:
            n = self.stages[self.next_i]
            path = os.path.join(self.out_dir, f"stage_{n}")
            self.model.save(path)
            rate = np.mean([s for _, s in self.recent]) if self.recent else 0.0
            lvl = 0
            try:
                lvl = int(np.mean(self.training_env.get_attr("curriculum_level")))
            except Exception:
                pass
            row = dict(steps=n, success_rate=float(rate), curriculum=lvl,
                       wall_s=round(time.time() - self.t0, 1))
            self.log_rows.append(row)
            print(f"[stage] {n:>9,} steps  success {rate*100:5.1f}%  "
                  f"curriculum {lvl:>2}  {row['wall_s']:.0f}s", flush=True)
            with open(os.path.join(self.out_dir, "stages.json"), "w") as f:
                json.dump(self.log_rows, f, indent=2)
            self.next_i += 1
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sac_dr")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=600_000)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--train-freq", type=int, default=32)
    ap.add_argument("--gradient-steps", type=int, default=80)
    ap.add_argument("--net", type=int, nargs=2, default=[256, 256])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    out = os.path.join(ROOT, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    fns = [make_env(i) for i in range(a.envs)]
    venv = VecMonitor(SubprocVecEnv(fns) if a.envs > 1 else DummyVecEnv(fns))

    kw = dict(policy_kwargs=dict(net_arch=list(a.net)), learning_rate=a.lr,
              batch_size=a.batch, buffer_size=a.buffer, gamma=a.gamma, tau=a.tau,
              train_freq=a.train_freq, gradient_steps=a.gradient_steps,
              learning_starts=10_000, ent_coef="auto", verbose=0, device=a.device)
    model = (SAC.load(a.resume, env=venv, device=a.device) if a.resume
             else SAC("MlpPolicy", venv, **kw))

    cb = StageCheckpoint(out, [s for s in STAGES if s <= a.steps] + [a.steps])
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=cb, progress_bar=False)
    model.save(os.path.join(out, "final"))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
