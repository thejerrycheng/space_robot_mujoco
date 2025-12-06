def get_env_class(env_name):
    env_map = {
        "default": "rocket_env.rocket_landing_env",
        "env2":    "rocket_env.rocket_landing_env_2",
        "env3":    "rocket_env.rocket_landing_env_3",
        "simple":  "rocket_env.rocket_landing_env_simple",
        "new":     "rocket_env.rocket_landing_env_new",
        "real":    "rocket_env.rocket_realistic_env",
        "default2": "rocket_env.rocket_2_env",
        "polar":   "rocket_env.polar_rocket_env",
        "default3": "rocket_env.rocket_3_env",
        "current": "RocketLandingEnv"
    }

it should be synced to the environment if u test it using ```mjpython test_env --env default2```
or the policy: ```mjpython test_ppo2.py runs/ppo_rocket2_velocity_field_20251203_134613/```
初始位置在哪里写死的？
self.START_POS_FIXED  = np.array([15.0, 0.0, 150.0])
初始朝向（姿态）如何确定的？（附完整解释）
火箭初始朝向：头朝向地面，yaw 朝向目标点 (0,0,0)
初始速度又是从哪里算的？
初始速度 = 3 m/s，方向沿着火箭指向的方向
判断成功和失败的逻辑在哪里？
_check_termination




RocketLandingEnv：/Users/junchengzhou/Downloads/space_robot_mujoco/rocket_env/polar_rocket_env.py
然后能找到：
MJCF_PATH = os.path.join(ROOT_DIR, "assets", "mjcf", "realistic_param.xml")

重力、时间步长、可视化
<option timestep="0.01" gravity="0 0 -1.62" integrator="RK4"/>
. 火箭刚体（ball）

<body name="ball" pos="0 0 150">
    <freejoint name="ball_free"/>

    <inertial pos="0 0 0" mass="5000000"
              diaginertia="1.2e9 1.2e9 3.0e7"/>
    ...
</body>
初始位置（XML 默认）：pos="0 0 150"，也就是世界坐标里 (x, y, z) = (0, 0, 150)。

质量：mass = 5,000,000 kg（5000 吨）

转动惯量对角项：I = [1.2e9, 1.2e9, 3.0e7] kg·m²
→ 这是一个“又长又重的大管子”的量级。

推力施加点和偏心距
<body name="thruster_mount" pos="0 0 -30">
</body>

4. 推力和舵机（actuator）
<actuator>
  <position name="yaw_servo" joint="thruster_yaw"
            ctrllimited="true" ctrlrange="-1 1"/>

  <position name="pitch_servo" joint="thruster_pitch"
            ctrllimited="true" ctrlrange="-1 1"/>

  <general name="thrust" site="thrust_site"
           gear="0 0 1 0 0 0"
           biastype="none"
           ctrllimited="true" ctrlrange="0 25000000"/>
</actuator>

真真的初始位置：由 Python 的 RocketLandingEnv 决定
self.TARGET_POS_WORLD = np.array([0.0, 0.0, 0.0])

self.INIT_RADIUS  = 15.0
self.INIT_HEIGHT  = 10.0
self.INITIAL_SPEED = 5.0
self.INITIAL_ROLL_DEG = 0.0


r = self.INIT_RADIUS  # 15m
theta_pos = 0.0       # 沿 +X
start_x = r * cos(theta_pos) = 15
start_y = r * sin(theta_pos) = 0
start_z = self.INIT_HEIGHT    = 10

self.data.qpos[self.qpos_adr : self.qpos_adr+3] = [start_x, start_y, start_z]




从 environment 中得到什么？我需要返回哪些 action？
def test_env(env_name, episodes=5):
    # 1. Load the correct environment class
    EnvClass = get_env_class(env_name)
    env = EnvClass(render_mode="human")

pos  = env.data.qpos[env.qpos_adr : env.qpos_adr+3].copy()   # [x,y,z]
vel  = env.data.qvel[env.qvel_adr : env.qvel_adr+3].copy()   # [vx,vy,vz]
quat = env.data.qpos[env.qpos_adr+3 : env.qpos_adr+7].copy() # [w,x,y,z]
env.model, env.data：标准 MuJoCo MjModel / MjData

env.model.opt.gravity：重力向量

env.model.opt.timestep：仿真步长

env.qpos_adr, env.qvel_adr：这个火箭在 qpos/qvel 里的索引起点

env.DRY_MASS、env.fuel_mass：干质量、当前燃料质量

env.thrust_act、env.yaw_act、env.pitch_act：在 data.ctrl 里的三个 actuator 索引

env.action_space：动作空间范围（例如 Box([-1,-1,-1], [1,1,1]) 之类）

info 里：至少有 info["success"]，有时还有 info["reason"]。


2 你需要返回什么样的 action？
# action = np.array([-1.0, 0.0, 0.0]) # Free fall (0 thrust)
action = env.action_space.sample() * 0.1 + np.array([-1, 0, 0])
第一个元素：thrust（推力大小 / 油门归一化）
后两个元素：gimbal yaw & pitch（喷口偏转）


def my_controller(obs, env):
    # 1. 从 env / obs 解析出 x,y,z, vx,vy,vz, 姿态等
    # 2. 根据期望着陆点 / 期望姿态，算出需要的 thrust + gimbal
    # 3. 把它归一化成动作空间 [-1,1] 内
    return np.array([throttle_cmd, yaw_cmd, pitch_cmd], dtype=np.float32)
然后把测试脚本里这行换掉：
# 原来：
# action = env.action_space.sample() * 0.1 + np.array([-1, 0, 0])

# 改成：
action = my_controller(obs, env)
