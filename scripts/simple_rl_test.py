import time
import os
import mujoco
import numpy as np
import style3dsim as sim
import mujoco_style3d.s3d_mj as s3d_mj
from dataclasses import dataclass
import mujoco_style3d.s3d_mjx as s3d_mjx
import jax

# -----------------------------------------------------------------------------
# 1. 布料参数类 (来自 s3d_mj.py，在这里定义以确保可用)
# -----------------------------------------------------------------------------
cloth_param = s3d_mj.ClothParams(
    stretch_stiff=sim.Vec3f(1.0e5, 2.0e5, 2.0e4),
    bend_stiff=sim.Vec3f(1.0e3, 2.0e3, 1.5e3),
    density=150,
    static_friction=0.6,
    dynamic_friction=0.6
)


# -----------------------------------------------------------------------------
# 2. log in and load xml
# -----------------------------------------------------------------------------
def log_in(usr, pw):
    sim.login(usr, pw, True, None)

log_in('simsdk001', 'xSXiaCMd') # 使用您的凭据

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()

print("PROJECT_ROOT:", PROJECT_ROOT)
xml_path = os.path.join(PROJECT_ROOT, 'assets', 'mujoco_model', 'dual_piper_with_cloth_simple.xml')
xml_path ='/home/hwk/program/deformale_mjx/thirdparty/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/mjx_panda.xml'

# -----------------------------------------------------------------------------
# 2. 初始化
# -----------------------------------------------------------------------------
BATCH_SIZE = 64  # 尝试一个大的批次！
print(f"Initializing {BATCH_SIZE} parallel environments with s3d_mjx...")

# 初始化新的数据管理器
data_manager = s3d_mjx.mjx_data_manager(xml_path, BATCH_SIZE)

# 获取 JAX 模型 (用于获取执行器数量)
mjx_model, _ = data_manager._get_mjx_data()
num_actuators = mjx_model.nu
print(f"Initialization complete. Num actuators per env: {num_actuators}")

# -----------------------------------------------------------------------------
# 3. RL 测试循环
# -----------------------------------------------------------------------------
print("\n--- 开始RL效率测试 (SPS) ---")

# (在真实RL中，这是一个JAX策略)
# 这里我们只创建一个固定的随机动作批次
key = jax.random.PRNGKey(0)
action_batch = jax.random.uniform(key, shape=(BATCH_SIZE, num_actuators), minval=-1.0, maxval=1.0)

start_time = time.time()
total_steps = 0
num_frames_to_run = 1000

for i in range(num_frames_to_run):
    # 1. (获取观测)
    # mjx_model, mjx_data = data_manager._get_mjx_data()
    # obs_batch = get_my_observations(mjx_data)
    
    # 2. (运行策略)
    # action_batch = policy(obs_batch)
    
    # 3. 将动作设置回 mjx_data
    # 我们需要先获取，然后替换，然后设置回去
    mjx_model, mjx_data = data_manager._get_mjx_data()
    mjx_data_with_ctrl = mjx_data.replace(ctrl=action_batch)
    data_manager.mjx_data.set_mjx_data(mjx_data_with_ctrl) #
    
    # 4. 步进
    data_manager.step() #
    
    # 5. 更新计数器
    total_steps += BATCH_SIZE
    
    if (i + 1) % 100 == 0:
        elapsed_time = time.time() - start_time
        # SPS = 总共模拟的步数 / 花费的时间
        sps = total_steps / elapsed_time
        print(f"Frame: {i+1} | Total Steps: {total_steps} | SPS: {sps:,.2f}")

print("--- 测试完成 ---")