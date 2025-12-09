import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

import os
import glob
# ... rest of your imports
# ============================================================
# IEEE GLOBAL FIGURE SETTINGS
# ============================================================
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.titlesize": 9,
    "savefig.dpi": 300,
})

# ============================================================
# APPLE COLOR PALETTE
# ============================================================
APPLE_BLUE   = "#007AFF"  # iOS system blue
APPLE_ORANGE = "#FF9500"  # iOS system orange
APPLE_RED    = "#FF3B30"  # iOS system red


def get_valid_result_folder(algo, reward, base_path="./results"):
    if not os.path.exists(base_path):
        print(f"❌ Base path '{base_path}' does not exist.")
        return None, None

    all_folders = glob.glob(os.path.join(base_path, "*"))
    
    matching_folders = []
    target_algo = algo.lower()
    target_reward = reward.lower()

    for f in all_folders:
        folder_name = os.path.basename(f).lower()
        if target_algo in folder_name and target_reward in folder_name:
            matching_folders.append(f)
            
    if not matching_folders:
        print(f"❌ No folders found matching Algo='{algo}' and Reward='{reward}'")
        return None, None

    matching_folders.sort(key=os.path.getmtime, reverse=True)
    
    for folder in matching_folders:
        csv_path = os.path.join(folder, "training_log.csv")
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 10:
            continue

        try:
            df = pd.read_csv(csv_path)
            if len(df) > 2:
                print(f"✅ Using valid run: {os.path.basename(folder)}")
                return folder, df
        except Exception:
            continue

    print("❌ No valid data found.")
    return None, None


def plot_metric(df, x_col, y_col, ax, window=50, color="#007AFF", title=None, ylabel=None):
    """
    Plots smoothed mean + std band with Apple color palette.
    """
    rolling_mean = df[y_col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[y_col].rolling(window=window, min_periods=1).std()
    x_data = df[x_col]

    ax.plot(
        x_data,
        rolling_mean,
        color=color,
        linewidth=1.2,
        label=f"Mean (win={window})"
    )
    
    ax.fill_between(
        x_data,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color=color,
        alpha=0.20,
        label="±1 Std Dev"
    )
    
    ax.set_title(title if title else y_col)
    ax.set_ylabel(ylabel if ylabel else y_col)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')


def main():
    parser = argparse.ArgumentParser(description="Plot RL Training Results")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    folder_path, df = get_valid_result_folder(args.algo, args.reward)
    if folder_path is None:
        return

    folder_name = os.path.basename(folder_path)

    # ============================================================
    # IEEE FIGURE SIZE (3.5 in × ≤5 in)
    # ============================================================
    fig_width = 3.5
    fig_height = 5.0

    fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=True)

    # ------------------- PLOT 1: EPISODE REWARD -------------------
    if "episode_reward" in df.columns:
        plot_metric(
            df, "global_step", "episode_reward", axes[0],
            window=args.window,
            color=APPLE_BLUE,
            title="Episode Reward",
            ylabel="Reward"
        )
    else:
        axes[0].text(0.5, 0.5, "Missing episode_reward", ha='center', va='center')

    # ------------------- PLOT 2: EPISODE LENGTH -------------------
    if "episode_length" in df.columns:
        plot_metric(
            df, "global_step", "episode_length", axes[1],
            window=args.window,
            color=APPLE_ORANGE,
            title="Episode Length",
            ylabel="Steps"
        )
    else:
        axes[1].text(0.5, 0.5, "Missing episode_length", ha='center', va='center')

    # ------------------- PLOT 3: FUEL REMAINING -------------------
    if "fuel_remaining" in df.columns:
        plot_metric(
            df, "global_step", "fuel_remaining", axes[2],
            window=args.window,
            color=APPLE_RED,
            title="Fuel Remaining",
            ylabel="Fuel"
        )
    else:
        axes[2].text(0.5, 0.5, "Missing fuel_remaining", ha='center', va='center')

    axes[2].set_xlabel("Global Timesteps")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ============================================================
    # SAVE AS VECTORIZED PDF
    # ============================================================
    os.makedirs("./plots", exist_ok=True)
    save_path = f"./plots/{folder_name}.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"📄 Vector PDF saved: {save_path}")


if __name__ == "__main__":
    main()
