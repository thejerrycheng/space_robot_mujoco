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
        self.GIMBAL_FRACTION = 0.4     # 最多用 40% 的 MAX_GIMBAL


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
        self.MAX_GIMBAL = np.deg2rad(3.0)

        # --- TASK CONSTANTS (FIXED) ---
        self.TARGET_POS_WORLD = np.array([0.0, 0.0, 0.0])
        self.START_POS_FIXED  = np.array([500.0, 0.0, 500.0])
        self.INITIAL_SPEED    = 3.0 
        self.PITCH_DOWN_DEG   = 0
        self.LANDING_Z = 0.5 
        
        self.MAX_STEPS = 2000
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
        
    def compute_landing_ctrl(self):
        """
        极简着陆控制器（写在 env 里面）：
        - 垂直：目标下降速度 v_z^*(z)，用一阶速度环调 thrust，不再总是拉满。
        - 水平：PD 控制期望水平加速度 a_xy^*，一起合成“期望推力方向向量”，
                再一次性解出 yaw / pitch，避免两个舵机互相打架。
        返回:
            thrust_cmd  [N]
            yaw_cmd     [rad]
            pitch_cmd   [rad]
        """

        # ---- 当前状态 ----
        pos = self._get_pos()
        vel = self._get_vel()
        z   = pos[2]
        vz  = vel[2]

        # 当前总质量（干质量 + 剩余燃料）
        m = self.DRY_MASS + self.fuel_mass
        g = -float(self.model.opt.gravity[2])   # ≈ 1.62

        # =====================================================================
        # 1) 垂直方向：目标下降速度曲线 v_z^*(z) + 速度环 → a_z_extra
        # =====================================================================
        if z > self.Z_SLOWDOWN_ALT:
            # 高空：下降快一点
            vz_target = self.VZ_TARGET_FAR      # 例如 -5 m/s
        else:
            # 低空：在 [VZ_TARGET_FAR, VZ_TARGET_NEAR] 之间线性插值
            alpha = np.clip(z / self.Z_SLOWDOWN_ALT, 0.0, 1.0)
            vz_target = alpha * self.VZ_TARGET_FAR + (1.0 - alpha) * self.VZ_TARGET_NEAR

        vz_err = vz_target - vz                 # 目标 - 当前
        a_z_extra = self.Kv_z * vz_err          # “额外”竖直加速度（相对 free-fall）

        # 限幅：额外加速度不能太大
        a_z_extra = np.clip(a_z_extra,
                            -self.AZ_MAX_DOWN,   # 负数：加速下落
                            +self.AZ_MAX_UP)     # 正数：减速 / 上升

        # 总竖直加速度（世界系，向上为正） = g + 额外项
        a_z_total = g + a_z_extra

        # =====================================================================
        # 2) 推力大小：T = m * a_z_total   （不能为负）
        # =====================================================================
        if a_z_total <= 0.0:
            # 不会反向推火箭，a_z_total<=0 说明“推力不如自由落体有用”，干脆关掉推力
            thrust_cmd = 0.0
        else:
            thrust_cmd = m * a_z_total

        thrust_cmd = np.clip(thrust_cmd, 0.0, self.MAX_THRUST)

        # =====================================================================
        # 3) 水平方向：PD 得到期望水平加速度 a_xy_des
        # =====================================================================
        xy   = pos[:2]
        v_xy = vel[:2]

        xy_err  = -xy        # 目标是 (0,0)
        vxy_err = -v_xy

        a_xy_des = self.Kp_xy * xy_err + self.Kd_xy * vxy_err

        # 限制水平加速度模长，避免倾角太大（例如最多 0.5 g）
        a_xy_norm = np.linalg.norm(a_xy_des)
        if a_xy_norm > 1e-6:
            scale = min(1.0, self.AXY_MAX / a_xy_norm)
            a_xy_des = a_xy_des * scale

        # =====================================================================
        # 4) 合成“期望推力方向向量” → thrust direction t_W
        # =====================================================================
        # 注意：这里用的是 [a_x, a_y, a_z_total]，方向和真正 thrust 一致
        a_vec = np.array([a_xy_des[0], a_xy_des[1], a_z_total], dtype=np.float64)

        if np.linalg.norm(a_vec) < 1e-6:
            tW = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # 默认向上
        else:
            tW = a_vec / np.linalg.norm(a_vec)

        tx, ty, tz = tW

        # =====================================================================
        # 5) 由期望推力方向 tW 反解 universal joint 的 yaw / pitch
        #    假设：先绕 yaw，再绕 pitch（MuJoCo XML 的顺序），
        #          推力大致沿 body +Z 轴。
        # =====================================================================
        # yaw 使得(0,y,z) → 对齐 tW 的 y-z 投影
        yaw = np.arctan2(ty, tz)

        # 为了计算 pitch，要补偿 yaw 的影响，防止 cos(yaw) 接近 0
        cos_y = np.cos(yaw)
        if abs(cos_y) < 1e-4:
            cos_y = np.sign(cos_y) * 1e-4 if cos_y != 0 else 1e-4

        # pitch 负责在 x-z 平面内倾斜
        pitch = np.arctan2(-tx, tz / cos_y)

        # =====================================================================
        # 6) 限制到 gimbal 行程的一部分，避免来回“打架”
        # =====================================================================
        max_gimbal = self.GIMBAL_FRACTION * self.MAX_GIMBAL   # 例如 0.4 * 10° ≈ 4°
        yaw_cmd   = np.clip(yaw,   -max_gimbal, max_gimbal)
        pitch_cmd = np.clip(pitch, -max_gimbal, max_gimbal)

        return thrust_cmd, yaw_cmd, pitch_cmd




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