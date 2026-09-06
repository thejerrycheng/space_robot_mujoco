#!/usr/bin/env python3
"""
Why the cascaded PD stops working above a near-hover start.

Produces two figures:
    tintin3d_ablation.pdf   which randomised parameter breaks it, one at a time
    tintin3d_limitcycle.pdf the mechanism: gimbal saturation at low throttle
"""
import os, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.modules.setdefault("tensorflow", None)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from rocket_env.rocket_dr_env import RocketLandingDREnv as E   # noqa: E402
from pd3d import PD3D, rollout                                  # noqa: E402

INK, HI, GOLD, GREEN, ASH = "#151820", "#E4442A", "#D9A13F", "#2E9E5B", "#7A7466"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.9,
                     "figure.dpi": 140, "savefig.bbox": "tight", "legend.frameon": False})
OPT = dict(altitude=60, lateral=0, vel_std=0, tilt_deg=0)
OFF = dict(mass_scale=(1, 1), fuel_scale=(1, 1), isp_scale=(1, 1), thrust_scale=(1, 1),
           gimbal_bias_deg=0.0, gimbal_scale=(1, 1), rate0=0.0,
           obs_pos_noise=0.0, obs_vel_noise=0.0, obs_quat_noise=0.0)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=170)
    plt.close(fig); print("  wrote", name)


def ablate(out, n=10):
    base = dict(E.DR)
    cases = [("none randomised", {}),
             ("mass ±20 %", dict(mass_scale=base["mass_scale"])),
             ("thrust ±20 %", dict(thrust_scale=base["thrust_scale"])),
             ("gimbal gain ±15 %", dict(gimbal_scale=base["gimbal_scale"])),
             ("sensor noise", dict(obs_pos_noise=0.6, obs_vel_noise=0.2, obs_quat_noise=0.005)),
             ("initial body rate", dict(rate0=base["rate0"])),
             ("gimbal bias ±2.5°", dict(gimbal_bias_deg=base["gimbal_bias_deg"])),
             ("all of them", dict(base))]
    rows = []
    for name, over in cases:
        cfg = dict(OFF); cfg.update(over); E.DR = cfg
        env = E(seed=11, domain_randomize=True, curriculum=False)
        pd = PD3D(env)
        c = collections.Counter(); tilt = []
        for s in range(n):
            r = rollout(env, pd, options=OPT, seed=800 + s)
            c[r["outcome"]] += 1; tilt.append(r["tilt_deg"])
        rows.append((name, c["success"] / n * 100, float(np.median(tilt))))
        print(f"  {name:20s} {c['success']}/{n}   median touchdown tilt {np.median(tilt):5.1f}°")
        E.DR = base

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0))
    y = np.arange(len(rows))
    ax[0].barh(y, [r[1] for r in rows],
               color=[GREEN if r[1] > 50 else HI for r in rows], edgecolor=INK, lw=0.8)
    ax[0].set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlabel("landings inside the box  [%]"); ax[0].set_xlim(0, 108)
    ax[0].set_title("one parameter randomised at a time")
    ax[1].barh(y, [r[2] for r in rows],
               color=[GREEN if r[2] < 15 else HI for r in rows], edgecolor=INK, lw=0.8)
    ax[1].set_yticks(y, ["" for _ in rows])
    ax[1].axvline(15, color=INK, ls="--", lw=1.2)
    ax[1].set_xlabel("median touchdown tilt  [deg]")
    ax[1].set_title("15° is the landing limit")
    for a in ax:
        a.grid(True, ls=":", axis="x", alpha=.35)
    fig.suptitle("A 60 m descent, straight down, nothing else changed", fontsize=10, y=1.03)
    fig.tight_layout(); save(fig, out, "tintin3d_ablation")
    return rows


def limitcycle(out):
    base = dict(E.DR)
    cfg = dict(OFF); cfg["gimbal_bias_deg"] = 2.5
    E.DR = cfg
    env = E(seed=11, domain_randomize=True, curriculum=False)
    pd = PD3D(env)
    env.reset(seed=800, options=OPT)
    t, tilt, thr, gim, alt = [], [], [], [], []
    for i in range(700):
        a = pd(env)
        _, _, term, trunc, _ = env.step(a)
        m = env._metrics()
        t.append(env.step_count * env.DT * env.FRAME_SKIP)
        tilt.append(m["tilt_deg"]); alt.append(m["alt"])
        thr.append((a[0] + 1) / 2 * 100); gim.append(max(abs(a[1]), abs(a[2])) * 100)
        if term or trunc:
            break
    E.DR = base
    fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.4), sharex=True)
    ax[0].plot(t, tilt, color=HI, lw=1.8, label="tilt [deg]")
    ax[0].plot(t, alt, color=INK, lw=1.4, ls="--", label="altitude [m]")
    ax[0].axhline(15, color=GREEN, ls=":", lw=1.4)
    ax[0].text(t[-1] * 0.82, 17, "landing limit, 15°", fontsize=8, color=GREEN)
    ax[0].legend(fontsize=8); ax[0].grid(True, ls=":", alpha=.35)
    ax[1].plot(t, thr, color=GOLD, lw=1.6, label="throttle [%]")
    ax[1].plot(t, gim, color=INK, lw=1.4, label="|gimbal command| [% of ±30°]")
    ax[1].axhline(100, color=HI, ls=":", lw=1.4)
    ax[1].text(t[-1] * 0.72, 103, "gimbal saturated", fontsize=8, color=HI)
    ax[1].set_xlabel("time  [s]"); ax[1].legend(fontsize=8); ax[1].grid(True, ls=":", alpha=.35)
    ax[0].set_title("A 2.5° gimbal misalignment, and a cascade that cannot reject it")
    fig.tight_layout(); save(fig, out, "tintin3d_limitcycle")


if __name__ == "__main__":
    out = os.path.join(ROOT, "docs", "figures")
    print("ablation:"); ablate(out)
    print("limit cycle:"); limitcycle(out)
