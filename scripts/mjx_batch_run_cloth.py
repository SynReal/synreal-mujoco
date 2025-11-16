import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import style3dsim as sim
import mujoco_style3d.s3d_mj as s3d_mj  # 假设 s3d_mj 位于您的 python 路径中
from pathlib import Path
# -----------------------------------------------------------------------------
# 1. cloth param
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
# -----------------------------------------------------------------------------
# 3. batch environment setup
# -----------------------------------------------------------------------------
batch_size = 10  # 设置您想要的批量大小
viz_index = 0   # 我们将可视化第 0 个环境

# 为不同的环境创建不同的参数
params_list = [cloth_param] * batch_size

print(f"Total {batch_size} environments created.")

m = mujoco.MjModel.from_xml_path(xml_path)

# 为每个环境创建单独的数据和模拟器实例
d_list = []
world_list = []
sim_pieces_list = []
piece_names_list = []
rigid_bodies_list = []

for i in range(batch_size):
    # 1. 为 MuJoCo 创建 MjData
    d = mujoco.MjData(m)
    
    # (可选) 在此处随机化初始状态，例如：
    # random_qpos = m.qpos0 + np.random.randn(m.nq) * 0.1
    # d.qpos[:] = random_qpos 
    
    mujoco.mj_forward(m, d)  # 填充初始数据
    
    # 2. 为 Style3D Sim 创建 World
    world = s3d_mj.get_a_sim_world()
    
    # 3. 从 MjData 将对象添加到 Style3D Sim，并传入物理参数
    sim_pieces, piece_names = s3d_mj.add_piece_to_sim(m, d, world, params_list[i])
    rigid_bodies = s3d_mj.add_rigid_body_to_sim(m, d, world)
    
    # 4. 存储所有实例
    d_list.append(d)
    world_list.append(world)
    sim_pieces_list.append(sim_pieces)
    piece_names_list.append(piece_names)
    rigid_bodies_list.append(rigid_bodies)

print("环境创建完毕。")

# -----------------------------------------------------------------------------
# 4. 批量模拟循环
# -----------------------------------------------------------------------------
sync_rate = 1

# 仅为我们要可视化的环境启动查看器
with mujoco.viewer.launch_passive(m, d_list[viz_index]) as viewer:
    fi = 0
    while viewer.is_running():
        begin0_t = time.time()
        
        # --- 在 Python 中循环批量模拟 ---
        for i in range(batch_size):
            # 获取此环境的实例
            d = d_list[i]
            world = world_list[i]
            rigid_bodies = rigid_bodies_list[i]
            sim_pieces = sim_pieces_list[i]
            piece_names = piece_names_list[i]
            
            # 1. 步进 MuJoCo (用于刚体)
            mujoco.mj_step(m, d)
            
            # 2. 将 MuJoCo 刚体姿态同步到 Style3D
            s3d_mj.set_rigid_body_pos_to_sim(m, d, rigid_bodies)
            
            # 3. 步进 Style3D Sim (用于布料)
            world.step_sim()
            
            # 4. 从 Style3D 获取结果
            world.fetch_sim(0)
            
            # 5. 将 Style3D 布料顶点位置同步回 MuJoCo (用于可视化)
            s3d_mj.set_piece_pos_to_mujoco(m, d, sim_pieces, piece_names)
        # --- 批量循环结束 ---

        # 仅同步被可视化的环境
        if fi % sync_rate == 0:
            viewer.sync()
            
        fi += 1
        
        end0_t = time.time()
        duration0 = end0_t - begin0_t
        if duration0 > 0:
            print(f"FPS (Batch of {batch_size}) = {1. / duration0:.2f}")