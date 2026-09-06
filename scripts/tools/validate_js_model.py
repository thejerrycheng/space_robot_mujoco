#!/usr/bin/env python3
"""
Cross-check the browser lander against MuJoCo.

Runs the 3-D PD controller in the real MuJoCo environment, records the exact
action sequence and initial state, and writes them to JSON. The companion Node
script replays the same actions through assets/js/tintin3d.js on the website and
reports how far the two trajectories diverge — the number quoted on the page.
"""
import json, os, sys
import numpy as np

sys.modules.setdefault("tensorflow", None)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from rocket_env.rocket_dr_env import RocketLandingDREnv    # noqa: E402
from pd3d import PD3D                                       # noqa: E402


def main(out_path, n_eps=6):
    env = RocketLandingDREnv(seed=4242, domain_randomize=False, curriculum=False)
    pd = PD3D(env)
    eps = []
    for k in range(n_eps):
        env.reset(seed=500 + k, options=dict(altitude=120 + 20 * k, lateral=20 + 10 * k,
                                             vel_std=2.0, tilt_deg=6.0))
        m = env._metrics()
        ep = dict(p=[float(x) for x in m["pos"]], v=[float(x) for x in m["vel"]],
                  q=[float(x) for x in m["quat"]], w=[float(x) for x in m["omega"]],
                  actions=[], states=[])
        while True:
            a = pd(env)
            ep["actions"].append([float(x) for x in a])
            _, _, term, trunc, info = env.step(a)
            mm = env._metrics()
            ep["states"].append([float(x) for x in mm["pos"]] +
                                [float(x) for x in mm["vel"]] +
                                [float(x) for x in mm["quat"]])
            if term or trunc:
                break
        ep["outcome"] = info.get("outcome")
        eps.append(ep)
        print(f"  episode {k}: {len(ep['actions'])} steps, {ep['outcome']}")
    with open(out_path, "w") as f:
        json.dump(dict(dt=env.DT, episodes=eps), f)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mj_ref.json")
