#!/usr/bin/env python3
"""
Reproducible experiment harness for the TINTIN PID/PD baseline on
RocketGym-planar.  Regenerates every baseline figure used on
https://thejerrycheng.github.io/tintin.html from scratch.

    python scripts/tools/run_pid_baseline.py --trials 500 --out docs/figures

Figures (vector PDF + PNG):
    tintin_pid_bundle.pdf      Monte-Carlo trajectory bundle + touchdown scatter
    tintin_pid_failures.pdf    Failure-mode breakdown and touchdown histograms
    tintin_pid_gains.pdf       Success rate over the (kp_x, kd_x) gain grid
    tintin_pid_envelope.pdf    Success rate over the deployment envelope
    tintin_pid_episode.pdf     One success and one failure, state by state

All randomness is seeded; rerunning reproduces the numbers exactly.
"""
import argparse, collections, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rocketgym_planar as R

INK, HI, POP, GOLD, GREEN = "#151820", "#E4442A", "#D9A13F", "#B07C1E", "#2E9E5B"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": INK, "axes.linewidth": 0.9,
    "figure.dpi": 140, "savefig.bbox": "tight",
    "legend.frameon": False, "grid.alpha": 0.25,
})

MODE_LABEL = {"success": "success", "tilt": "tilt > 15°", "impact": "impact > 20 m/s",
              "drift": "off pad / drifted out", "overspeed": "speed > 200 m/s",
              "timeout": "2000-step timeout"}


def monte_carlo(trials, seed=0, gains=None, params=None, record=0):
    """Run `trials` PID episodes from the deployment ellipse."""
    env = R.RocketGymPlanar(seed=seed)
    if params:
        for k, v in params.items():
            setattr(R, k, v)
    pid = R.PIDBaseline(**(gains or {}))
    rows, traces = [], []
    for i in range(trials):
        rec = i < record
        r = R.rollout(env, pid, record=rec)
        if rec:
            traces.append(r.pop("trace"))
        rows.append(r)
    return rows, traces


def summarise(rows):
    c = collections.Counter(r["outcome"] for r in rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    def st(key, src):
        v = np.array([r[key] for r in src]) if src else np.array([0.0])
        return dict(mean=float(v.mean()), median=float(np.median(v)),
                    p95=float(np.percentile(v, 95)), max=float(v.max()))
    return dict(
        trials=len(rows), success=c["success"], rate=c["success"] / len(rows),
        modes={k: v for k, v in c.most_common()},
        radius=st("radius", ok), tilt=st("tilt", ok),
        vz=dict(mean=float(np.mean([abs(r["vz"]) for r in ok])) if ok else 0.0,
                median=float(np.median([abs(r["vz"]) for r in ok])) if ok else 0.0,
                p95=float(np.percentile([abs(r["vz"]) for r in ok], 95)) if ok else 0.0,
                max=float(np.max([abs(r["vz"]) for r in ok])) if ok else 0.0),
        prop=st("prop_frac", ok), flight=st("t", ok))


# --------------------------------------------------------------------------- #
def fig_bundle(rows, traces, out):
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6),
                           gridspec_kw={"width_ratios": [1.55, 1]})
    a = ax[0]
    for tr in traces:
        x, z = tr[:, 1], tr[:, 2]
        a.plot(x, z, lw=0.6, alpha=0.35, color=INK)
    ok = [r for r in rows if r["outcome"] == "success"]
    bad = [r for r in rows if r["outcome"] != "success"]
    a.axhline(R.Z_TOUCHDOWN, color=INK, lw=1.2)
    a.axvspan(-R.SUCCESS_RADIUS, R.SUCCESS_RADIUS, color=GOLD, alpha=0.18, lw=0)
    a.set_xlabel("downrange $x$  [m]"); a.set_ylabel("CoM altitude $z$  [m]")
    a.set_title(f"PD baseline, {len(traces)} recorded descents")
    a.set_xlim(-250, 620); a.set_ylim(0, 580); a.grid(True, ls=":")

    b = ax[1]
    # Only episodes that actually reached the pad have a meaningful touchdown
    # state; the ones that left the box are counted, not plotted.
    td_ok = [r for r in ok]
    td_bad = [r for r in bad if r["outcome"] in ("tilt", "impact")
              and abs(r["x"]) <= 400]
    gone = len(bad) - len(td_bad)
    b.scatter([r["x"] for r in td_ok], [abs(r["tilt"]) for r in td_ok], s=12,
              color=GREEN, label=f"landed ({len(td_ok)})", zorder=3)
    b.scatter([r["x"] for r in td_bad], [abs(r["tilt"]) for r in td_bad], s=14,
              color=HI, marker="x", label=f"hard touchdown ({len(td_bad)})", zorder=3)
    b.axvspan(-R.SUCCESS_RADIUS, R.SUCCESS_RADIUS, color=GOLD, alpha=0.18, lw=0)
    b.axhline(np.rad2deg(R.SUCCESS_TILT), color=INK, ls="--", lw=1)
    b.set_xlabel("touchdown offset  [m]"); b.set_ylabel("touchdown tilt  [deg]")
    b.set_title(f"touchdown box  ({gone} never reached it)")
    b.set_xlim(-120, 200); b.legend(loc="upper right", fontsize=8); b.grid(True, ls=":")
    fig.tight_layout(); save(fig, out, "tintin_pid_bundle")


def fig_failures(rows, out):
    c = collections.Counter(r["outcome"] for r in rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
    keys = [k for k, _ in c.most_common()]
    vals = [c[k] / len(rows) * 100 for k in keys]
    cols = [GREEN if k == "success" else HI for k in keys]
    ax[0].barh([MODE_LABEL.get(k, k) for k in keys], vals, color=cols,
               edgecolor=INK, lw=0.8)
    for i, v in enumerate(vals):
        ax[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlabel("share of episodes  [%]")
    ax[0].set_xlim(0, max(vals) * 1.28); ax[0].set_title("outcome breakdown")

    ax[1].hist([r["radius"] for r in ok], bins=20, color=GOLD, edgecolor=INK, lw=0.7)
    ax[1].axvline(R.SUCCESS_RADIUS, color=HI, ls="--", lw=1.2)
    ax[1].set_xlabel("touchdown radius  [m]"); ax[1].set_ylabel("successes")
    ax[1].set_title("landing accuracy")

    ax[2].hist([r["prop_frac"] * 100 for r in ok], bins=20, color=GOLD,
               edgecolor=INK, lw=0.7)
    ax[2].set_xlabel("propellant used  [% of load]"); ax[2].set_ylabel("successes")
    ax[2].set_title("propellant")
    for a in ax: a.grid(True, ls=":", axis="x")
    fig.tight_layout(); save(fig, out, "tintin_pid_failures")


def fig_gains(out, trials=60):
    kps = np.array([0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012])
    kds = np.array([0.04, 0.07, 0.10, 0.13, 0.18, 0.24])
    grid = np.zeros((len(kds), len(kps)))
    for i, kd in enumerate(kds):
        for j, kp in enumerate(kps):
            rows, _ = monte_carlo(trials, seed=11, gains=dict(kp_x=kp, kd_x=kd))
            grid[i, j] = sum(r["outcome"] == "success" for r in rows) / trials * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    im = ax.imshow(grid, origin="lower", cmap="YlOrBr", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(kps)), [f"{k:g}" for k in kps])
    ax.set_yticks(range(len(kds)), [f"{k:g}" for k in kds])
    ax.set_xlabel(r"$k_{p,x}$"); ax.set_ylabel(r"$k_{d,x}$")
    ax.set_title(f"PD success rate over the lateral gain grid ({trials} runs / cell)")
    for i in range(len(kds)):
        for j in range(len(kps)):
            ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center", fontsize=7,
                    color=INK if grid[i, j] < 60 else "white")
    fig.colorbar(im, label="success rate [%]")
    fig.tight_layout(); save(fig, out, "tintin_pid_gains")
    return dict(kps=kps.tolist(), kds=kds.tolist(), grid=grid.tolist())


def fig_envelope(out, trials=40):
    """Where does the fixed-gain PD's feasible set end?

    The PD burns only ~4 % of the propellant load, so mass depletion is *not*
    what limits it (a sweep over the propellant fraction from 30 % to 95 %
    moves the success rate by less than one run in forty).  What does limit it
    is the deployment envelope itself: the lateral loop has to slew the whole
    vehicle to translate, and past a certain downrange offset / entry speed it
    cannot null the offset before the glideslope runs out of altitude.
    """
    # capped at 650 m: the environment terminates past |x| > 700 m, so a
    # larger x0 is out of bounds at reset rather than a hard descent.
    x0s = np.array([100, 250, 400, 500, 600, 650], dtype=float)
    v0s = np.array([0, 10, 20, 30, 40, 50, 65, 80], dtype=float)
    grid = np.zeros((len(v0s), len(x0s)))
    env = R.RocketGymPlanar(seed=31)
    pid = R.PIDBaseline()
    for i, v0 in enumerate(v0s):
        for j, x0 in enumerate(x0s):
            ok = 0
            for k in range(trials):
                ang = 2 * np.pi * k / trials
                r = R.rollout(env, pid, x0=x0, z0=500.0,
                              vx0=v0 * np.cos(ang), vz0=-abs(v0 * np.sin(ang)),
                              theta0=0.0)
                ok += r["outcome"] == "success"
            grid[i, j] = ok / trials * 100

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    im = ax.imshow(grid, origin="lower", cmap="YlOrBr", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(x0s)), [f"{v:.0f}" for v in x0s])
    ax.set_yticks(range(len(v0s)), [f"{v:.0f}" for v in v0s])
    ax.set_xlabel("initial downrange offset $x_0$  [m]")
    ax.set_ylabel("initial speed $\\|v_0\\|$  [m/s]")
    ax.set_title(f"Where the fixed-gain PD still lands ({trials} entry angles / cell)")
    for i in range(len(v0s)):
        for j in range(len(x0s)):
            ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center", fontsize=7,
                    color=INK if grid[i, j] < 60 else "white")
    ax.axvline(3.0, color=HI, lw=1.8, ls="--")
    ax.text(3.08, len(v0s) - 1.35, "deployment\nellipse", fontsize=7.5, color=HI)
    fig.colorbar(im, label="success rate [%]")
    fig.tight_layout(); save(fig, out, "tintin_pid_envelope")
    return dict(x0=x0s.tolist(), v0=v0s.tolist(), grid=grid.tolist())


def sweep_massratio(trials=80):
    """Null result, reported rather than plotted: the propellant fraction
    barely moves the PD success rate, because the baseline never gets near
    the dry mass."""
    out = {}
    base = R.DRY_FRACTION
    for f in (0.30, 0.60, 0.90, 0.95):
        R.DRY_FRACTION = 1 - f
        R.M_DRY = R.DRY_FRACTION * R.M_WET
        R.M_PROP = R.M_WET - R.M_DRY
        rows, _ = monte_carlo(trials, seed=23)
        out[f"{f:.2f}"] = sum(r["outcome"] == "success" for r in rows) / trials * 100
    R.DRY_FRACTION = base
    R.M_DRY = R.DRY_FRACTION * R.M_WET
    R.M_PROP = R.M_WET - R.M_DRY
    return out


def fig_episode(out):
    env = R.RocketGymPlanar(seed=5)
    pid = R.PIDBaseline()
    good = bad = None
    for i in range(200):
        r = R.rollout(env, pid, record=True)
        if r["outcome"] == "success" and good is None:
            good = r
        elif r["outcome"] != "success" and bad is None:
            bad = r
        if good and bad:
            break
    fig, ax = plt.subplots(2, 3, figsize=(9.6, 4.6), sharex="col")
    for row, (r, name, col) in enumerate([(good, "success", GREEN), (bad, f"failure ({bad['outcome']})", HI)]):
        tr = r["trace"]
        t, x, z, vx, vz, th, om, m = (tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3],
                                      tr[:, 4], tr[:, 5], tr[:, 6], tr[:, 7])
        uT, uG = tr[:, 8], tr[:, 9]
        ax[row, 0].plot(t, z, color=col, lw=1.6, label="altitude $z$")
        ax[row, 0].plot(t, x, color=INK, lw=1.1, ls="--", label="downrange $x$")
        ax[row, 0].set_ylabel("m"); ax[row, 0].legend(fontsize=7)
        ax[row, 1].plot(t, np.hypot(vx, vz), color=col, lw=1.6, label="speed")
        ax[row, 1].plot(t, np.rad2deg(th), color=GOLD, lw=1.3, label="tilt [deg]")
        ax[row, 1].axhline(15, color=INK, ls=":", lw=0.9)
        ax[row, 1].legend(fontsize=7)
        ax[row, 2].plot(t, (uT + 1) / 2 * 100, color=col, lw=1.4, label="throttle [%]")
        ax[row, 2].plot(t, uG * 30, color=GOLD, lw=1.1, label="gimbal [deg]")
        ax[row, 2].plot(t, (m - R.M_DRY) / R.M_PROP * 100, color=INK, lw=1.1,
                        ls="--", label="propellant left [%]")
        ax[row, 2].legend(fontsize=7)
        ax[row, 0].set_title(name, loc="left", color=col, fontweight="bold")
        for a in ax[row]:
            a.grid(True, ls=":")
    for a in ax[1]:
        a.set_xlabel("t  [s]")
    fig.tight_layout(); save(fig, out, "tintin_pid_episode")
    return {k: dict(outcome=r["outcome"], t=r["t"], radius=r["radius"],
                    tilt=r["tilt"], vz=r["vz"], prop_frac=r["prop_frac"])
            for k, r in [("success", good), ("failure", bad)]}


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=170)
    plt.close(fig)
    print("  wrote", name + ".pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)

    print(f"RocketGym-planar  Tmax={R.T_MAX:.3e} N  m0={R.M_WET:.2e} kg  "
          f"Isp={R.ISP:g} s  g={R.G_MOON} m/s^2  dt={R.DT}s")
    print(f"Monte Carlo: {a.trials} PD episodes from the deployment ellipse ...")
    rows, traces = monte_carlo(a.trials, seed=0, record=120)
    s = summarise(rows)
    print(json.dumps(s, indent=2))

    fig_bundle(rows, traces, out)
    fig_failures(rows, out)
    ep = fig_episode(out)
    gains = fig_gains(out, trials=25 if a.quick else 60)
    env_grid = fig_envelope(out, trials=16 if a.quick else 40)
    mass = sweep_massratio(trials=40 if a.quick else 80)
    print("  propellant-fraction null result:", mass)

    res = dict(summary=s, gain_grid=gains, envelope=env_grid, mass_sweep=mass, episodes=ep,
               params=dict(T_MAX=R.T_MAX, M_WET=R.M_WET, ISP=R.ISP, DT=R.DT,
                           GAINS=R.DEFAULT_GAINS))
    with open(os.path.join(out, "tintin_pid_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("  wrote tintin_pid_results.json")


if __name__ == "__main__":
    main()
