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
        self.MAX_THRUST = TOTAL_MASS * MOON_G * 2.0
        self.MAX_GIMBAL = np.deg2rad(100.0)

        # --- TASK CONSTANTS (FIXED) ---
        self.TARGET_POS_WORLD = np.array([0.0, 0.0, 0.0])
        self.START_POS_FIXED  = np.array([500.0, 0.0, 500.0])
        self.INITIAL_SPEED    = 3.0 
        self.PITCH_DOWN_DEG   = 0
        self.LANDING_Z = 0.5 
        
        self.MAX_STEPS = 3000
        
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
        - 垂直方向：高度依赖的目标下降速度 -> 速度环 -> 期望加速度 -> 推力。
        - 姿态方向：分三段（大角度翻转 / 刹车 / 直立微调），强烈抑制角速度。
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

        # 角速度
        ang_vel = data.cvel[self.rocket_bid][:3].copy()  # [wx, wy, wz]
        # ============ 把角速度变成机体系，然后取 pitch 方向的角速度 ============
        # data.xmat: body->world 的旋转矩阵，按行摊平成 9 个元素
        R_bw = data.xmat[self.rocket_bid].reshape(3, 3)   # body -> world
        R_wb = R_bw.T                                     # world -> body

        # 世界坐标角速度 -> 机体坐标角速度
        ang_vel_body = R_wb @ ang_vel              # [wx_body, wy_body, wz_body]

        # 机体系的 pitch 是绕 y 轴旋转，所以：
        pitch_rate = ang_vel_body[1]                     # [rad/s]
        pitch_rate_deg = np.degrees(pitch_rate)          # [deg/s]
        
        # 度数形式
        roll_deg  = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg   = np.degrees(yaw)
        tilt_deg  = np.sqrt(roll_deg**2 + pitch_deg**2)

        # 质量
        dry_mass  = getattr(self, "DRY_MASS",
                             self.model.body_mass[self.rocket_bid])
        fuel_mass = getattr(self, "fuel_mass", 0.0)
        m = dry_mass + fuel_mass

        # ============================================================
        # 1. 垂直控制：高度依赖的目标下降速度 + 速度环
        # ============================================================
        z_target = getattr(self, "LANDING_Z", 0.5)
        h = max(z - z_target, 0.0)       # 离着陆面的高度

        # 1.1 目标下降速度 vz_ref(h)
        if h > 300.0:
            vz_ref = -35.0
        elif h > 200.0:
            vz_ref = -25.0
        elif h > 120.0:
            vz_ref = -20.0
        elif h > 60.0:
            vz_ref = -8.0
        elif h > 20.0:
            vz_ref = -4.0 * (h / 20.0)      # 20m -> -4, 0m -> 0
        else:
            vz_ref = -1.5 * (h / 20.0)      # 20m -> -1.5, 0m -> 0

        # 1.2 速度环
        K_v_far  = 0.6
        K_v_near = 1.0
        alpha = np.clip(h / 100.0, 0.0, 1.0)
        K_v = K_v_near + (K_v_far - K_v_near) * alpha

        a_z_des = K_v * (vz_ref - vz)

        # 限制最大上下加速度
        a_z_up_max   = 3.0 * g
        a_z_down_max = 0.5 * g
        a_z_des = np.clip(a_z_des, -a_z_down_max, a_z_up_max)

        # 1.3 推力
        F_des = m * (g + a_z_des)
        F_des = float(np.clip(F_des, 0.0, self.MAX_THRUST))

        if h > 250.0 and vz < -10.0:
            F_des = max(F_des, 0.30 * self.MAX_THRUST)

        # ============================================================
        # 2. 姿态控制（重点修改的部分）
        #    分三段：
        #      A) |pitch| > 20° : 翻转模式（PD）
        #      B) 5° < |pitch| ≤ 20° : 刹车模式（只阻尼角速度）
        #      C) |pitch| ≤ 5° : 直立微调（弱阻尼或直接 0）
        # ============================================================

        # pitch 是弧度，先转成“带符号”的角度
        pitch_deg = np.degrees(pitch)

        gimbal_limit = self.MAX_GIMBAL*1.3

        FLIP_HIGH = 90.0
        FLIP_LOW  = 80.0   # 90~80: 最大翻转
        SLOW_LOW  = 65.0   # 80~55: 力度减小
        BRAKE_LOW = 20.0   # 55~20: 刹车到 0
        # DEAD_BAND = 2.0  # 这版你要求里其实没用到
        VEL_SMALL_DEG = 1.0   # 角速度阈值：小于这个就认为“快停了”

        pitch_cmd_local = 0.0

        if pitch_deg >= FLIP_LOW:
            # 90 ~ 80: pitch_cmd = +gimbal_limit
            pitch_cmd_local = gimbal_limit

        elif pitch_deg >= SLOW_LOW:
            # 80 ~ 55: 从 +gimbal_limit 线性减到 0
            #  SLOW_LOW -> 0, FLIP_LOW -> 1
            t = (pitch_deg - SLOW_LOW) / (FLIP_LOW - SLOW_LOW)
            t = max(0.0, min(1.0, t))
            pitch_cmd_local = 3*gimbal_limit * t

        elif pitch_deg >= BRAKE_LOW:
            # 55 ~ 20: 从 0 线性减到 -0.3*gimbal_limit
            #  BRAKE_LOW -> 0, SLOW_LOW -> 1
            t = (pitch_deg - BRAKE_LOW) / (SLOW_LOW - BRAKE_LOW)
            t = max(0.0, min(1.0, t))
            pitch_cmd_local = -0.2 * gimbal_limit * t


        elif pitch_deg >= 3.0:
            # 20 ~ 0
            t = pitch_deg / BRAKE_LOW
            t = max(0.0, min(1.0, t))
            base_cmd = -0.13 * gimbal_limit * t

            # 这里用 pitch_rate_deg（刚算出来的）
            if abs(pitch_rate_deg) < VEL_SMALL_DEG:
                micro = 0.003 * gimbal_limit
                if abs(base_cmd) > micro:
                    pitch_cmd_local = np.sign(base_cmd) * micro
                else:
                    pitch_cmd_local = base_cmd
            else:
                pitch_cmd_local = base_cmd

        elif pitch_deg >= -20.0:
            # 0 ~ -20
            t = (-pitch_deg) / 20.0
            t = max(0.0, min(1.0, t))
            base_cmd = -0.18 * gimbal_limit * t

            if abs(pitch_rate_deg) < VEL_SMALL_DEG:
                micro = 0.005 * gimbal_limit
                if abs(base_cmd) > micro:
                    pitch_cmd_local = np.sign(base_cmd) * micro
                else:
                    pitch_cmd_local = base_cmd
            else:
                pitch_cmd_local = base_cmd

        else:
            # pitch_deg < -20: 先简单饱和在 +0.3*gimbal_limit
            pitch_cmd_local = 0
            
        if abs(z) < 5:
            pitch_cmd_local = 0.0
        if abs(vz) < 2:
            pitch_cmd_local = 0.0


        # 限幅一次（防止浮点误差）
        pitch_cmd = float(np.clip(pitch_cmd_local, -gimbal_limit, gimbal_limit))

        # yaw 你之前用的是简单阻尼，这里不乱改：
        K_yaw_damp = 0.6
        yaw_cmd_local = 0
        yaw_cmd = float(np.clip(yaw_cmd_local, -gimbal_limit, gimbal_limit))

        # DEBUG（可选）
        step_idx = int(data.time / dt)
        if step_idx % 50 == 0:
            print(
                f"[ATT] step={step_idx:5d} "
                f"pitch={pitch_deg:7.2f}deg "
                f"pitch_cmd_local={pitch_cmd_local: .4f}"
            )

        # 注意：这里我**不再对 pitch_cmd 做任何额外加负号**。
        # 外面的接口你要不要 -pitch_cmd，由你自己决定：
        return F_des, yaw_cmd, pitch_cmd  # 如果你之前就是这样，就保持这样













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