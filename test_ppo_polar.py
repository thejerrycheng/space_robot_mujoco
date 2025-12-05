import os
import sys
import time
import argparse
import numpy as np
import importlib
import csv

# Plotting
import matplotlib
matplotlib.use("Agg")  # Prevent crash on macOS/Linux
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.colors as pc

# RL
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Environment
from rocket_env.polar_rocket_env import RocketLandingEnv


# ============================================================
# Utility: Load reward dynamically
# ============================================================
def load_reward_function(name):
    module_path = f"rocket_env.rewards.{name}"
    mod = importlib.import_module(module_path)
    return mod.compute_reward


# ============================================================
# Normalize obs using VecNormalize stats
# ============================================================
def normalize_obs(obs, obs_rms):
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
        -10, 10
    )



# ============================================================
# Save episode to CSV
# ============================================================
def save_episode(history, ep, save_dir):
    path = os.path.join(save_dir, f"episode_{ep}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time", "r", "h", "theta", "rdot", "hdot", "thetadot",
                    "thrust", "pitch", "roll", "reward"])
        for i in range(len(history["time"])):
            w.writerow([
                i,
                history["time"][i],
                *history["state"][i],
                *history["action"][i],
                history["reward"][i]
            ])
    print(f"💾 Saved CSV → {path}")


# ============================================================
# Plotly 3D Trajectory Plot
# ============================================================
def plot_trajectories(all_histories, save_dir):
    print("📈 Generating Plotly 3D trajectory visualization...")

    palette = pc.qualitative.Plotly
    fig = go.Figure()

    for idx, hist in enumerate(all_histories):
        states = np.array(hist["state"])
        r = states[:, 0]
        h = states[:, 1]
        theta = states[:, 2]

        # Convert polar to XY
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            name=f"Episode {idx+1}",
            line=dict(width=5, color=palette[idx % len(palette)]),
            opacity=0.85
        ))

        # Start and end points
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers",
            marker=dict(size=4, color="green"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode="markers",
            marker=dict(size=6, color="red"),
            showlegend=False
        ))

    # Landing pad visualization
    theta_pad = np.linspace(0, 2 * np.pi, 100)
    pad_r = 1.0
    fig.add_trace(go.Scatter3d(
        x=pad_r * np.cos(theta_pad),
        y=pad_r * np.sin(theta_pad),
        z=np.zeros_like(theta_pad),
        mode="lines",
        line=dict(color="black", width=4),
        name="Landing Pad"
    ))

    fig.update_layout(
        title="🚀 PPO Polar Rocket Trajectories",
        width=1200,
        height=800,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height",
            aspectmode="data"
        )
    )

    output = os.path.join(save_dir, "trajectory_plot.html")
    fig.write_html(output)

    print(f"🌎 Plot Saved → {output}")


# ============================================================
# MAIN TEST LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Test PPO Polar Rocket Agent")
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--reward", type=str, default="polar_vel_field")
    parser.add_argument("--model", type=str, default="final", choices=["final", "best", "latest"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    norm_path = os.path.join(run_dir, "vec_normalize.pkl")

    # Load reward
    reward_func = load_reward_function(args.reward)

    # Load normalization
    dummy_env = DummyVecEnv([lambda: RocketLandingEnv(reward_func=reward_func)])
    vecnorm = VecNormalize.load(norm_path, dummy_env)
    obs_rms = vecnorm.obs_rms
    print("Loaded observation normalization statistics.")

    # Locate model
    if args.model == "final":
        model_path = os.path.join(run_dir, "final_model.zip")
    elif args.model == "best":
        model_path = os.path.join(run_dir, "best_model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(run_dir, "best_model", "best_model.zip")
    else:  # latest
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        ckpts = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                        if f.endswith(".zip")], key=os.path.getmtime)
        model_path = ckpts[-1]

    print(f"🚀 Loading PPO Model → {model_path}")
    model = PPO.load(model_path)

    # Create env
    env = RocketLandingEnv(
        reward_func=reward_func,
        render_mode="human" if args.render else None
    )

    save_dir = os.path.join(run_dir, "test_results")
    os.makedirs(save_dir, exist_ok=True)

    all_histories = []

    # ============================================================
    # Run test episodes
    # ============================================================
    for ep in range(1, args.episodes + 1):
        print(f"\n==========================")
        print(f"▶ EPISODE {ep}")
        print("==========================")

        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0

        history = {"time": [], "state": [], "action": [], "reward": []}

        while not done:
            step += 1
            norm_obs = normalize_obs(obs, obs_rms)

            action, _ = model.predict(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            # Unpack obs (polar state)
            r, h, theta, rdot, hdot, thetadot = obs[:6]

            history["time"].append(step * env.DT)
            history["state"].append([r, h, theta, rdot, hdot, thetadot])
            history["action"].append(action)
            history["reward"].append(reward)

            if args.render:
                env.render()
                time.sleep(0.01)

            done = terminated or truncated

        all_histories.append(history)

        # Success evaluation
        is_success = info.get("success", False)
        is_semi = info.get("semi_success", False)
        final_r = history["state"][-1][0]
        final_h = history["state"][-1][1]

        if is_success:
            print(f"🏆 SUCCESS — r={final_r:.2f}, h={final_h:.2f}")
        elif is_semi:
            print(f"🟡 SEMI SUCCESS — r={final_r:.2f}")
        else:
            print(f"❌ FAILURE — r={final_r:.2f}")

        save_episode(history, ep, save_dir)

    env.close()

    # ============================================================
    # Plot trajectories using Plotly
    # ============================================================
    plot_trajectories(all_histories, save_dir)

    print("\n🎉 Testing Complete!")


if __name__ == "__main__":
    main()
