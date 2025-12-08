import os
import numpy as np
import pandas as pd

from plotfigure import plot_unified_analysis  # 改成你刚才那段代码所在的文件名

# 1. 指定 csv 路径
csv_path = "/Users/junchengzhou/Downloads/space_robot_mujoco/episode_1.csv"
save_dir = os.path.dirname(csv_path)

# 2. 读 csv
# === 读取 CSV ===
df = pd.read_csv(csv_path)

N = len(df)

# === 构造 history 字典（所有 reward 分量强制为 0）===
history = {
    "time": df["Time"].to_numpy(),
    "pos": df[["X", "Y", "Z"]].to_numpy(),
    "vel": df[["Vx", "Vy", "Vz"]].to_numpy(),
    "attitude": df[["Roll", "Pitch", "Yaw"]].to_numpy(),
    "thrust": df["Thrust"].to_numpy(),
    "gimbal": df[["GimbalYaw", "GimbalPitch"]].to_numpy(),
    "mass": df["Mass"].to_numpy(),

    # 强制 reward 列为 0，只要画图即可
    "reward": np.zeros(N),
    "r_upright": np.zeros(N),
    "r_vel": np.zeros(N),
    "r_dist": np.zeros(N),

    # 没有四元数 → 不画箭头
    "quat": np.zeros((N, 4)),
}

# === 调用统一画图函数 ===
plot_unified_analysis(
    history=history,
    episode_num=1,
    model_name="PID Landing",
    save_dir=save_dir
)

# 4. 设定保存目录（随便选一个）
save_dir = os.path.dirname(csv_path)  # 就存在 csv 同一个目录
episode_num = 0                      # 从文件名里自己填
model_name = "ppo_rocket2_velocity_field"  # 想叫啥都行

# 5. 调用你写好的画图函数
plot_unified_analysis(
    history=history,
    episode_num=episode_num,
    model_name=model_name,
    save_dir=save_dir,
)
