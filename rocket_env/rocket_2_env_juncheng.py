import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer
import importlib
from rocket_env.controllers.point_to_point import *

# Define path relative to this file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "realistic_param.xml")

class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, reward_func=None):
        super().__init__()

        self.T_TOTAL = 40.0          # 整个 A→B 轨迹时间（随便先定一个）
        self.Kp = 0.2                # 位置增益（先小一点）
        self.Kd = 0.6                # 速度增益
        self.LAT_GAIN = 0.5          # 横向加速度缩放
        self.PITCH_YAW_GAIN = 0.5    # 不用满行程，最多用 50% gimbal

        # 垂直方向控制参数
        self.VZ_TARGET_FAR   = -5.0     # 高空时希望的下降速度 (m/s)
        self.VZ_TARGET_NEAR  = -1.0     # 接近地面时希望的下降速度 (m/s)
        self.Z_SLOWDOWN_ALT  = 50.0     # 50m 以下减速
        self.Kv_z = 0.6                 # 垂直速度环增益

        # 限制最大纵向加速度（只给你 1g 的刹车能力，不允许 5g）
        self.AZ_MAX_UP   = 1.0 * 1.62  # 向上（减速）最大 a_z，大约 1g
        self.AZ_MAX_DOWN = 0.5 * 1.62  # 向下（加速）最大 a_z（可不要太大）

        # 水平 & 姿态控制
        self.Kp_xy = 0.2              # 水平位置 P
        self.Kd_xy = 0.2               # 水平速度 D
        self.AXY_MAX = 0.5 * 1.62  # 水平方向最多 0.3g

        # gimbal 使用比例，不用满行程
        self.GIMBAL_FRACTION = 0.2     # 最多用 40% 的 MAX_GIMBAL


        # ----------------------------------------------------------------
        # DYNAMIC REWARD LOADING
        # ----------------------------------------------------------------
        if reward_func is not None:
            self.reward_func = reward_func
        else:
            try:
                mod = importlib.import_module("rocket_env.rewards.flip_and_fuel")
                self.reward_func = mod.compute_reward
            except ImportError:
                print("⚠️  Warning: Could not import default reward 'flip_and_fuel'. Using placeholder.")
                self.reward_func = lambda env, m, t, term, succ: (0.0, {})

        # 1. LOAD MODEL & PHYSICS
        if not os.path.exists(MJCF_PATH):
            raise FileNotFoundError(f"Model file not found at: {MJCF_PATH}")

        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        
        # --- GRAVITY: MOON ---
        MOON_G = 1.62
        self.model.opt.gravity[:] = [0, 0, -MOON_G]

        # 2. IDENTIFIERS
        self.rocket_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.qpos_adr = self.model.jnt_qposadr[self.free_joint_id]
        self.qvel_adr = self.model.jnt_dofadr[self.free_joint_id]

        self.yaw_act   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "yaw_servo")
        self.pitch_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_servo")
        self.thrust_act= mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust")

        # --- PHYSICS CONSTANTS ---
        self.DRY_MASS = self.model.body_mass[self.rocket_bid]
        self.START_FUEL = 0.5 * self.DRY_MASS 
        TOTAL_MASS = self.DRY_MASS + self.START_FUEL
        
        self.ISP = 250.0
        self.G0 = 9.81  
        self.DT = self.model.opt.timestep

        # --- CONTROL LIMITS ---
        self.MAX_THRUST = TOTAL_MASS * MOON_G * 4.0
        self.MAX_GIMBAL = np.deg2rad(15.0)

        # --- TASK CONSTANTS (FIXED) ---
        self.TARGET_POS_WORLD = np.array([0.0, 0.0, 0.0])
        self.START_POS_FIXED  = np.array([100.0, 0.0, 500.0])
        self.INITIAL_SPEED    = 3.0 
        self.PITCH_DOWN_DEG   = 0
        self.LANDING_Z = 0.5 
        
        self.MAX_STEPS = 20000
        self.MAX_LATERAL_DIST = 10000.0 
        self.MAX_VELOCITY = 10000.0     

        # Observation Space
        obs_high = np.ones(23) * 500
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # 4. INIT STATE
        self.fuel_mass = self.START_FUEL
        self.total_mass = TOTAL_MASS
        self.orig_inertia = self.model.body_inertia[self.rocket_bid].copy()
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)
        # ---- 轨迹缓存：保存火箭 COM 轨迹 ----
        self.traj_points = []
        self.REF_TRAJ_POINTS = 100  # 虚拟轨迹采样点数，可自行调整

        print("Act IDs:",
            "yaw:", self.yaw_act,
            "pitch:", self.pitch_act,
            "thrust:", self.thrust_act)
        print("Total actuators:", self.model.nu)
        print("Actuator names:", [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                                for i in range(self.model.nu)])
        
    def quat_to_euler_rad(self, quat):
        """Convert [w, x, y, z] to [roll, pitch, yaw] in radians."""
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw], dtype=np.float64)


    def compute_landing_ctrl(self):
        """
        基于“最小化 gimbal + 最小化角速度 + 竖直软着陆”思路的控制器。
        - 不追踪水平位置，只要求最后竖直、速度接近 0。
        - 垂直方向：高度依赖的目标下降速度 -> PD 速度环 -> 期望加速度 -> 推力。
        - 姿态方向：以 pitch 为主的 PD（yaw 只做阻尼），大幅抑制角速度。
        - 在接近竖直时自动把 gimbal 命令压得很小（间接“最小化偏转角”）。
        返回:
            thrust_cmd [N]           ∈ [0, MAX_THRUST]
            yaw_cmd, pitch_cmd [rad] ∈ [-MAX_GIMBAL, +MAX_GIMBAL]
        """
        model, data = self.model, self.data
        g  = -model.opt.gravity[2]           # Moon g ≈ 1.62
        dt = model.opt.timestep

        # --------- 当前状态 ---------
        pos  = data.qpos[self.qpos_adr : self.qpos_adr+3].copy()      # [x, y, z]
        vel  = data.qvel[self.qvel_adr : self.qvel_adr+3].copy()      # [vx, vy, vz]
        quat = data.qpos[self.qpos_adr+3 : self.qpos_adr+7].copy()    # [w, x, y, z]
        roll, pitch, yaw = self.quat_to_euler_rad(quat)               # [rad]

        z  = float(pos[2])
        vz = float(vel[2])

        # 角速度（body frame / spatial frame，这里主要用来做阻尼）
        ang_vel = data.cvel[self.rocket_bid][:3].copy()  # [wx, wy, wz]

        # 度数形式，方便调参 / debug
        roll_deg  = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg   = np.degrees(yaw)
        tilt_deg  = np.sqrt(roll_deg**2 + pitch_deg**2)

        # 质量 = 干质量 + 剩余燃料
        dry_mass  = getattr(self, "DRY_MASS",
                             self.model.body_mass[self.rocket_bid])
        fuel_mass = getattr(self, "fuel_mass", 0.0)
        m = dry_mass + fuel_mass

        # ============================================================
        # 1. 垂直控制：高度依赖的目标下降速度 + 速度环
        #    思路：高空允许较快下落，越接近地面越慢，最后在 z≈LANDING_Z 减速到 0
        # ============================================================
        z_target = getattr(self, "LANDING_Z", 0.5)
        h = max(z - z_target, 0.0)       # 离着陆面的高度

        # 1.1 目标下降速度 vz_ref(h)
        # 高空：接近 -35 m/s，自由下落 + 一点控制
        # 中高空：逐渐减小
        # 近地面 (< 20m)：线性 taper 到 0，保证软着陆
        if h > 300.0:
            vz_ref = -35.0
        elif h > 200.0:
            vz_ref = -25.0
        elif h > 120.0:
            vz_ref = -15.0
        elif h > 60.0:
            vz_ref = -8.0
        elif h > 20.0:
            # 20m 以内开始明显刹车
            # h=20 -> -4 m/s, h=0 -> 0 m/s
            vz_ref = -4.0 * (h / 20.0)
        else:
            # 最后 20m 内再收一档，接近地面要非常慢
            vz_ref = -1.5 * (h / 20.0)   # h=20 -> -1.5, h=0 -> 0

        # 1.2 速度环：根据 (vz_ref - vz) 给一个期望向上的额外加速度
        #       a_z_des > 0 表示需要额外向上加速（减小下落速度）
        K_v_far  = 0.6   # 高空增益
        K_v_near = 1.0   # 近地面更激进一点，保证拉住
        alpha = np.clip(h / 100.0, 0.0, 1.0)
        K_v = K_v_near + (K_v_far - K_v_near) * alpha

        a_z_des = K_v * (vz_ref - vz)

        # 限制最大上下加速度，防止过激
        a_z_up_max   = 3.0 * g    # 最大向上加速度（约 3g）
        a_z_down_max = 0.5 * g    # 最大额外向下加速度（加速下落用得不多）
        a_z_des = np.clip(a_z_des, -a_z_down_max, a_z_up_max)

        # 1.3 推力计算：T = m * (g + a_z_des)
        F_des = m * (g + a_z_des)
        F_des = float(np.clip(F_des, 0.0, self.MAX_THRUST))

        # 额外保险：在高空且下落速度太大时，强制不低于一定推力
        if h > 250.0 and vz < -10.0:
            F_des = max(F_des, 0.30 * self.MAX_THRUST)

        # ============================================================
        # 2. 姿态控制：目标 roll=pitch=0（竖直），并强烈抑制角速度
        #    思路：
        #      - 大倾角时允许多用一点 gimbal 以加快翻转；
        #      - 进入竖直附近时，快速把 gimbal 命令收小，避免抖动；
        #      - yaw 只做阻尼，避免绕 z 自转。
        # ============================================================

        # 2.1 根据倾角设置增益（gain scheduling）
        # 倾角越大，允许的修正力度越大；接近竖直时减小增益 -> gimbal 自动变小
        tilt_for_gain = np.clip(abs(pitch_deg) / 80.0, 0.0, 1.0)  # 约 0~1
        tilt_for_gain = max(0.2, tilt_for_gain)                   # 至少保留一点控制

        # 基础增益（可再调）
        Kp_pitch_base = 0.08
        Kd_pitch_base = 0.6

        Kp_pitch = Kp_pitch_base * tilt_for_gain
        Kd_pitch = Kd_pitch_base * tilt_for_gain

        # pitch 角速度（这里用 ang_vel[1]，按你原来习惯）
        pitch_rate = ang_vel[1]

        # 标准 PD：目标 pitch = 0
        pitch_cmd_local = -Kp_pitch * pitch - Kd_pitch * pitch_rate

        # 竖直附近，如果角度和角速度都很小，直接把命令归零，避免小抖动
        if abs(pitch_deg) < 3.0 and abs(np.degrees(pitch_rate)) < 3.0:
            pitch_cmd_local = 0.0

        # 2.2 yaw：只做阻尼，不追任何目标方位
        K_yaw_damp = 0.6
        yaw_rate = ang_vel[2]
        yaw_cmd_local = -K_yaw_damp * yaw_rate

        # 2.3 gimbal 物理限制 + “少用偏转角”策略
        # 最大只用 80% 的 MAX_GIMBAL
        gimbal_limit = self.MAX_GIMBAL * 0.8

        pitch_cmd = float(np.clip(pitch_cmd_local, -gimbal_limit, gimbal_limit))
        yaw_cmd   = float(np.clip(yaw_cmd_local,   -gimbal_limit, gimbal_limit))

        # ============================================================
        # （可选）DEBUG 打印
        # ============================================================
        step_idx = int(data.time / dt)
        step_idx = int(data.time / dt)
        if step_idx % 50 == 0:
            print(
                f"[ATT] step={step_idx:5d} "
                f"tilt={tilt_deg:6.2f}deg "
                f"pitch={pitch_deg:7.2f}deg "
                f"pitch_cmd(rad)={pitch_cmd: .4f} "
            )

        # 这里保持你原来的符号约定：外面用 F_des, -yaw_cmd, -pitch_cmd
        return F_des, -yaw_cmd, -pitch_cmd











    # =========================================================================
    # CORE: STEP
    # =========================================================================
    def step(self, action):
        self.step_count += 1

        # ===== 1) 调用内置着陆控制器（忽略外部 action）=====
        thrust_cmd, yaw_cmd, pitch_cmd = self.compute_landing_ctrl()

        # ===== 2) 燃料消耗 =====
        if self.fuel_mass > 0.0:
            mdot = -thrust_cmd / (self.ISP * self.G0)   # T = mdot * g0 * Isp
            self.fuel_mass = max(self.fuel_mass + mdot * self.DT, 0.0)
        else:
            thrust_cmd = 0.0

        # ===== 3) 写入 MuJoCo 控制量（注意：这里是物理量，不是 [-1,1]）=====
        self.data.ctrl[self.thrust_act] = thrust_cmd
        self.data.ctrl[self.yaw_act]    = yaw_cmd
        self.data.ctrl[self.pitch_act]  = pitch_cmd

        mujoco.mj_step(self.model, self.data)

        # （可选：打印一点 debug）
        if self.step_count % 50 == 0:
            print("ctrl thrust/yaw/pitch =",
                  self.data.ctrl[self.thrust_act],
                  self.data.ctrl[self.yaw_act],
                  self.data.ctrl[self.pitch_act])

        # ===== 4) 观测 & 终止条件 & 奖励 =====
        obs = self._get_obs()
        state_metrics = self._get_state_metrics()
        terminated, truncated, success = self._check_termination(state_metrics)

        # 半成功标记（进圈但条件不完全满足）
        dist_xy = state_metrics["dist_xy"]
        semi_success = (dist_xy < 5.0) and not success

        reward, reward_info = self.reward_func(
            self, state_metrics, thrust_cmd, terminated, success
        )

        info = {
            "success": success,
            "semi_success": semi_success,
            "fuel": self.fuel_mass,
            "dist": state_metrics["target_dist_3d"],
            **reward_info,
        }

        # 记录真实轨迹用于绘图
        if hasattr(self, "traj_points"):
            self.traj_points.append(self._get_pos())

        return obs, reward, terminated, truncated, info


    # =========================================================================
    # LOGIC: TERMINATION
    # =========================================================================
    def _check_termination(self, m):
        terminated = False
        truncated = False
        success = False

        if m["z"] < 0.4: terminated = True
        if m["dist_xy"] > self.MAX_LATERAL_DIST: terminated = True
        if m["vel_err"] > self.MAX_VELOCITY: terminated = True

        # Success: Low altitude, close to 0,0 XY, slow, upright
        if (0.0 < m["z"] < 1.0 and 
            m["dist_xy"] < 0.5 and
            m["vel_err"] < 0.5 and 
            m["tilt"] < 0.05):
            success = True
            terminated = True

        if self.step_count >= self.MAX_STEPS: truncated = True

        return terminated, truncated, success

    # =========================================================================
    # RESET & UTILS
    # =========================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. SET POSITION
        self.data.qpos[self.qpos_adr : self.qpos_adr+3] = self.START_POS_FIXED

        # 2. CALCULATE HEADING (YAW)
        dx = self.TARGET_POS_WORLD[0] - self.START_POS_FIXED[0]
        dy = self.TARGET_POS_WORLD[1] - self.START_POS_FIXED[1]
        yaw_angle = np.arctan2(dy, dx)

        # 3. CALCULATE PITCH (90 + 10 deg down)
        pitch_angle_rad = np.deg2rad(90.0 + self.PITCH_DOWN_DEG)

        # 4. CONSTRUCT QUATERNION
        hp = pitch_angle_rad / 2
        hy = yaw_angle / 2
        q_pitch = np.array([np.cos(hp), 0, np.sin(hp), 0])
        q_yaw = np.array([np.cos(hy), 0, 0, np.sin(hy)])
        
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_pitch
        q_total = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7] = q_total

        # 5. SET VELOCITY
        nz = np.cos(pitch_angle_rad)
        nh = np.sin(pitch_angle_rad)
        nx = nh * np.cos(yaw_angle)
        ny = nh * np.sin(yaw_angle)
        
        self.data.qvel[self.qvel_adr : self.qvel_adr+3] = [
            nx * self.INITIAL_SPEED, 
            ny * self.INITIAL_SPEED, 
            nz * self.INITIAL_SPEED
        ]
        self.data.qvel[self.qvel_adr+3 : self.qvel_adr+6] = [0, 0, 0]

        self.fuel_mass = self.START_FUEL
        self.model.body_mass[self.rocket_bid] = self.DRY_MASS + self.START_FUEL
        self.model.body_inertia[self.rocket_bid] = self.orig_inertia.copy()
        
        
        mujoco.mj_forward(self.model, self.data)
        
        
        # ==== 1) 真实轨迹：从当前 COM 起点开始 ====
        self.traj_points = [self._get_pos()]

        # ==== 2) 虚拟轨迹：起点 → 目标点 的一条参考路径 ====
        start = self._get_pos()                # 起点：火箭当前 COM
        target = self.TARGET_POS_WORLD.copy()  # 目标：你定义的 [0,0,0]
        N = self.REF_TRAJ_POINTS

        # 简单做法：三维直线插值（之后想改成抛物线 / 其他轨迹也很容易）
        self.ref_traj = [
            (1.0 - s) * start + s * target
            for s in np.linspace(0.0, 1.0, N)
        ]
        
        
        
        if self.viewer is not None: self.viewer.sync()
        return self._get_obs(), {}

    def _get_state_metrics(self):
        pos = self._get_pos()
        vel = self._get_vel()
        quat = self._get_quat()
        ang_vel = self._get_ang_vel()
        
        dist_xy = np.linalg.norm(pos[:2])
        dist_3d = np.linalg.norm(pos - self.TARGET_POS_WORLD)

        return {
            "pos": pos, "vel": vel, "z": pos[2], "vz": vel[2], "quat_w": quat[0],
            "dist_xy": dist_xy,
            "target_dist_3d": dist_3d,
            "pos_err": dist_3d, "vel_err": np.linalg.norm(vel),
            "ang_err": np.linalg.norm(ang_vel), "tilt": 1.0 - quat[0]
        }

    def _get_obs(self):
        pos = self._get_pos()
        rel_pos = -1.0 * pos
        return np.array([*pos, *rel_pos, *self._get_vel(), *self._get_acc(), *self._get_quat(), *self._get_ang_vel(), *self._get_ang_acc(), self.fuel_mass], dtype=np.float32)

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            # 启动被动 viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # 使用 lock 保护 user_scn 的修改
        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0  # 清空之前的自定义几何

            max_geom = scn.maxgeom
            geom_count = 0

            # =========================
            # A. 起点：画一个小圆点（球）
            # =========================
            if hasattr(self, "ref_traj") and len(self.ref_traj) > 0 and geom_count < max_geom:
                start_pos = np.asarray(self.ref_traj[0], dtype=np.float32)
                rgba_start = np.array([0.0, 0.0, 1.0, 0.9], dtype=np.float32)  # 蓝色

                geom = scn.geoms[geom_count]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.4, 0, 0], dtype=np.float32),  # 半径 0.4m，可调
                    pos=start_pos,
                    mat=np.eye(3, dtype=np.float32).ravel(),
                    rgba=rgba_start,
                )
                geom_count += 1

            # =========================
            # B. 目标点：画一个叉叉（X）
            # =========================
            if geom_count < max_geom:
                target = np.asarray(self.TARGET_POS_WORLD, dtype=np.float64)
                rgba_target = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)  # 红色

                # 在 XY 平面上画一个 X，边长 d
                d = 1.0  # X 的半长度，可调
                # 对角线 1：(+d,+d) 到 (-d,-d)
                p1 = target + np.array([ d,  d, 0.0])
                p2 = target + np.array([-d, -d, 0.0])

                geom = scn.geoms[geom_count]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.array([0, 0, 0], dtype=np.float32),
                    pos=np.zeros(3, dtype=np.float32),
                    mat=np.eye(3, dtype=np.float32).ravel(),
                    rgba=rgba_target,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    2.0,   # 线宽
                    p1,
                    p2,
                )
                geom_count += 1

            if geom_count < max_geom:
                # 对角线 2：(+d,-d) 到 (-d,+d)
                p3 = target + np.array([ d, -d, 0.0])
                p4 = target + np.array([-d,  d, 0.0])

                geom = scn.geoms[geom_count]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.array([0, 0, 0], dtype=np.float32),
                    pos=np.zeros(3, dtype=np.float32),
                    mat=np.eye(3, dtype=np.float32).ravel(),
                    rgba=rgba_target,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    2.0,   # 线宽
                    p3,
                    p4,
                )
                geom_count += 1

            # =========================
            # C. 虚拟轨迹：起点 → 目标点 的“虚线”
            # =========================
            if hasattr(self, "ref_traj") and len(self.ref_traj) > 1:
                rgba_ref = np.array([0.0, 0.8, 0.8, 0.9], dtype=np.float32)  # 青色，区分真实轨迹
                pts = self.ref_traj

                # 用隔点画线段的方式实现“虚线”
                for i in range(0, len(pts) - 1, 2):
                    if geom_count >= max_geom:
                        break

                    p0 = np.asarray(pts[i],   dtype=np.float64)
                    p1 = np.asarray(pts[i+1], dtype=np.float64)

                    geom = scn.geoms[geom_count]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        size=np.array([0, 0, 0], dtype=np.float32),
                        pos=np.zeros(3, dtype=np.float32),
                        mat=np.eye(3, dtype=np.float32).ravel(),
                        rgba=rgba_ref,
                    )
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        1.5,  # 线宽
                        p0,
                        p1,
                    )
                    geom_count += 1

            # =========================
            # D. 真实轨迹：COM 的绿色虚线（你之前实现的）
            # =========================
            if hasattr(self, "traj_points") and len(self.traj_points) > 1:
                rgba_real = np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32)  # 绿色
                pts = self.traj_points

                for i in range(0, len(pts) - 1, 2):  # 同样隔点画虚线
                    if geom_count >= max_geom:
                        break

                    p0 = np.asarray(pts[i],   dtype=np.float64)
                    p1 = np.asarray(pts[i+1], dtype=np.float64)

                    geom = scn.geoms[geom_count]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        size=np.array([0, 0, 0], dtype=np.float32),
                        pos=np.zeros(3, dtype=np.float32),
                        mat=np.eye(3, dtype=np.float32).ravel(),
                        rgba=rgba_real,
                    )
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        1.5,
                        p0,
                        p1,
                    )
                    geom_count += 1
            # =========================
            # E. 飞船当前朝向箭头
            # =========================
            if geom_count < max_geom:
                # 1) 取当前 COM 位置
                pos = self._get_pos()  # 等价于 data.xpos[rocket_bid]

                # 2) 取姿态矩阵，得到“正方向”向量
                R = self.data.xmat[self.rocket_bid].reshape(3, 3)
                heading_dir = R[:,2]   # 假设 +x 是飞船朝向
                # 如果方向反了，可以改成 -R[:,0] 或 R[:,2]

                # 3) 箭头长度
                arrow_len = 200
                start = pos
                end = pos + arrow_len * heading_dir

                rgba_arrow = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)  # 黄色

                geom = scn.geoms[geom_count]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,  # 箭头
                    size=np.array([0, 0, 0], dtype=np.float32),
                    pos=np.zeros(3, dtype=np.float32),
                    mat=np.eye(3, dtype=np.float32).ravel(),
                    rgba=rgba_arrow,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.8,       # 箭头粗细
                    start,
                    end,
                )
                geom_count += 1

            scn.ngeom = geom_count

            try:
                self.viewer.sync()
            except:
                pass


    def close(self):
        if self.viewer: self.viewer.close(); self.viewer = None
    # Accessors
    def _get_pos(self): return self.data.xpos[self.rocket_bid].copy()
    def _get_vel(self): return self.data.qvel[self.qvel_adr:self.qvel_adr+3].copy()
    def _get_acc(self): return self.data.cacc[self.rocket_bid][3:].copy()
    def _get_quat(self): return self.data.qpos[self.qpos_adr+3 : self.qpos_adr+7].copy()
    def _get_ang_vel(self): return self.data.cvel[self.rocket_bid][:3].copy()
    def _get_ang_acc(self): return self.data.cacc[self.rocket_bid][:3].copy()