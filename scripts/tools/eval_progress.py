#!/usr/bin/env python3
"""
Evaluate every training stage of a SAC run against the PD baseline, in the same
MuJoCo environment, and draw the progress figures.

    python scripts/tools/eval_progress.py --run sac_dr --episodes 120

Produces, in docs/figures/:
    tintin3d_progress.pdf     success rate and touchdown quality vs training
    tintin3d_bundle.pdf       trajectory bundles: PD vs the trained policy
    tintin3d_baseline.pdf     PD outcome breakdown and touchdown statistics
    tintin3d_results.json     every number, machine readable
and, in docs/, a policy export for the browser demo.
"""
import argparse, collections, glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.modules.setdefault("tensorflow", None)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from stable_baselines3 import SAC                              # noqa: E402
from rocket_env.rocket_dr_env import RocketLandingDREnv        # noqa: E402
from pd3d import PD3D, rollout as pd_rollout                   # noqa: E402

INK, HI, GOLD, GREEN, ASH = "#151820", "#E4442A", "#D9A13F", "#2E9E5B", "#7A7466"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.9,
                     "figure.dpi": 140, "savefig.bbox": "tight", "legend.frameon": False})

# The evaluation envelope: fixed, and the same for every controller.
EVAL = dict(altitude=200.0, lateral=70.0, vel_std=8.0, tilt_deg=30.0)


def policy_rollout(env, model, options, record=False, seed=None):
    obs, _ = env.reset(seed=seed, options=options)
    trace = [] if record else None
    info = {}
    while True:
        a, _ = model.predict(obs, deterministic=True)
        if record:
            m = env._metrics()
            trace.append(np.concatenate([[env.step_count * env.DT * env.FRAME_SKIP], m["pos"], m["vel"],
                                         m["quat"], m["omega"], a]))
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    m = env._metrics()
    r = dict(outcome=info.get("outcome"), success=bool(info.get("success")),
             t=env.step_count * env.DT * env.FRAME_SKIP, lateral=m["lateral"], speed=m["speed"],
             tilt_deg=float(np.degrees(2 * np.arccos(np.clip(abs(m["quat"][0]), 0, 1)))),
             vz=m["vz"], fuel_frac=env.fuel_mass / env.start_fuel)
    if record:
        r["trace"] = np.array(trace)
    return r


def evaluate(env, fn, n, record=0, seed0=7000):
    rows, traces = [], []
    for i in range(n):
        r = fn(env, options=EVAL, record=(i < record), seed=seed0 + i)
        if "trace" in r:
            traces.append(r.pop("trace"))
        rows.append(r)
    ok = [r for r in rows if r["success"]]
    c = collections.Counter(r["outcome"] for r in rows)
    med = lambda k, src: float(np.median([abs(r[k]) for r in src])) if src else float("nan")
    return dict(n=n, success=len(ok), rate=len(ok) / n,
                outcomes=dict(c.most_common()),
                lateral=med("lateral", ok), speed=med("speed", ok),
                tilt=med("tilt_deg", ok), t=med("t", ok),
                fuel_used=float(np.median([1 - r["fuel_frac"] for r in ok])) if ok else float("nan"),
                # quality over ALL episodes, so a policy that never lands still scores
                all_lateral=med("lateral", rows), all_speed=med("speed", rows),
                all_tilt=med("tilt_deg", rows)), traces


def export_policy(model, steps, path):
    """Dump the SAC actor's mean network as plain JSON for the browser."""
    import torch
    actor = model.policy.actor
    layers = []
    mods = list(actor.latent_pi) + [actor.mu]
    for mod in mods:
        if isinstance(mod, torch.nn.Linear):
            W = mod.weight.detach().cpu().numpy()
            b = mod.bias.detach().cpu().numpy()
            layers.append(dict(w=[round(float(x), 6) for x in W.ravel()],
                               b=[round(float(x), 6) for x in b.ravel()],
                               **{"in": int(W.shape[1])}, out=int(W.shape[0]),
                               act="relu"))
    layers[-1]["act"] = "linear"
    spec = dict(steps=int(steps), obs=int(layers[0]["in"]),
                act=int(layers[-1]["out"]), layers=layers)
    with open(path, "w") as f:
        json.dump(spec, f, separators=(",", ":"))
    return spec


def fig_progress(stages, pd_stats, out):
    xs = [s["steps"] for s in stages]
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0))
    ax[0].semilogx(xs, [s["eval"]["rate"] * 100 for s in stages], "o-", color=HI, lw=2, ms=5)
    ax[0].axhline(pd_stats["rate"] * 100, color=INK, ls="--", lw=1.4)
    ax[0].text(xs[0], pd_stats["rate"] * 100 + 3, "3-D PD baseline", fontsize=8, color=INK)
    ax[0].set_ylabel("success rate  [%]"); ax[0].set_ylim(-3, 103)
    ax[0].set_title("landings inside the box")

    ax[1].semilogx(xs, [s["eval"]["all_lateral"] for s in stages], "o-", color=GOLD, lw=2, ms=5)
    ax[1].axhline(pd_stats["all_lateral"], color=INK, ls="--", lw=1.4)
    ax[1].axhline(10, color=GREEN, ls=":", lw=1.2)
    ax[1].set_ylabel("median touchdown offset  [m]"); ax[1].set_yscale("log")
    ax[1].set_title("how close it gets")

    ax[2].semilogx(xs, [s["eval"]["all_speed"] for s in stages], "o-", color=GREEN, lw=2, ms=5)
    ax[2].axhline(pd_stats["all_speed"], color=INK, ls="--", lw=1.4)
    ax[2].axhline(3, color=GREEN, ls=":", lw=1.2)
    ax[2].set_ylabel("median arrival speed  [m/s]"); ax[2].set_yscale("log")
    ax[2].set_title("how hard it arrives")
    for a in ax:
        a.set_xlabel("environment steps"); a.grid(True, ls=":", alpha=.35)
    fig.tight_layout(); save(fig, out, "tintin3d_progress")


def fig_bundle(pd_traces, pol_traces, out, pol_label):
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.3))
    for k, (trs, name, col) in enumerate([(pd_traces, "3-D PD baseline", INK),
                                          (pol_traces, pol_label, HI)]):
        for tr in trs:
            ax[k].plot(np.hypot(tr[:, 1], tr[:, 2]), tr[:, 3] - 1.0, lw=0.8, alpha=0.55, color=col)
        ax[k].axvspan(0, 10, color=GOLD, alpha=0.2, lw=0)
        ax[k].set_xlabel("lateral distance from the pad  [m]")
        ax[k].set_ylabel("altitude  [m]"); ax[k].set_title(name)
        ax[k].set_xlim(0, 120); ax[k].set_ylim(0, 260); ax[k].grid(True, ls=":")
    for trs, col, name in [(pd_traces, INK, "PD"), (pol_traces, HI, "SAC")]:
        for tr in trs:
            ax[2].plot(tr[:, 1], tr[:, 2], lw=0.8, alpha=0.5, color=col)
    th = np.linspace(0, 2 * np.pi, 90)
    ax[2].plot(10 * np.cos(th), 10 * np.sin(th), color=GOLD, lw=2)
    ax[2].set_xlabel("x  [m]"); ax[2].set_ylabel("y  [m]")
    ax[2].set_title("ground track (ink = PD, red = SAC)")
    ax[2].set_aspect("equal"); ax[2].grid(True, ls=":")
    fig.tight_layout(); save(fig, out, "tintin3d_bundle")


def fig_baseline(pd_stats, out):
    c = pd_stats["outcomes"]
    keys = list(c); vals = [c[k] / pd_stats["n"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.barh(keys, vals, color=[GREEN if k == "success" else HI for k in keys],
            edgecolor=INK, lw=0.8)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, max(vals) * 1.3)
    ax.set_xlabel("share of episodes  [%]")
    ax.set_title(f"3-D PD baseline in MuJoCo, {pd_stats['n']} descents")
    ax.grid(True, ls=":", axis="x", alpha=.35)
    fig.tight_layout(); save(fig, out, "tintin3d_baseline")


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=170)
    plt.close(fig); print("  wrote", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sac_dr")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--pd-episodes", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    ap.add_argument("--export-dir", default=os.path.join(ROOT, "docs", "policies"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(a.export_dir, exist_ok=True)

    env = RocketLandingDREnv(seed=99, domain_randomize=True, curriculum=False)

    print(f"3-D PD baseline in MuJoCo, {a.pd_episodes} descents ...")
    pd = PD3D(env)
    pd_stats, pd_traces = evaluate(env, lambda e, **kw: pd_rollout(e, pd, **kw),
                                   a.pd_episodes, record=40)
    print("  ", json.dumps({k: v for k, v in pd_stats.items() if k != "outcomes"}))
    print("  ", pd_stats["outcomes"])

    run_dir = os.path.join(ROOT, "runs", a.run)
    files = sorted(glob.glob(os.path.join(run_dir, "stage_*.zip")),
                   key=lambda p: int(re.search(r"stage_(\d+)", p).group(1)))
    stages, last_traces, last_label = [], [], "SAC"
    for f in files:
        n = int(re.search(r"stage_(\d+)", f).group(1))
        model = SAC.load(f, device="cpu")
        rec = 40 if f == files[-1] else 0
        st, trs = evaluate(env, lambda e, **kw: policy_rollout(e, model, **kw),
                           a.episodes, record=rec)
        stages.append(dict(steps=n, eval=st))
        print(f"  stage {n:>9,}  success {st['rate']*100:5.1f}%  "
              f"offset {st['all_lateral']:7.1f} m  speed {st['all_speed']:6.2f} m/s")
        export_policy(model, n, os.path.join(a.export_dir, f"stage_{n}.json"))
        if rec:
            last_traces, last_label = trs, f"SAC, {n/1e6:.1f}M steps" if n >= 1e6 else f"SAC, {n//1000}k steps"

    if stages:
        fig_progress(stages, pd_stats, out)
        fig_bundle(pd_traces, last_traces, out, last_label)
    fig_baseline(pd_stats, out)

    with open(os.path.join(out, "tintin3d_results.json"), "w") as f:
        json.dump(dict(pd=pd_stats, stages=stages, eval_envelope=EVAL), f, indent=2)
    print("  wrote tintin3d_results.json")


if __name__ == "__main__":
    main()
