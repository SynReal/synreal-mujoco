import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import time
import os

def print_model_statistics(model: mujoco.MjModel, xml_path: str):
    """
    打印影响JAX/MJX性能的关键模型统计数据。
    """
    print("-" * 60)
    print(f"📊 模型统计数据: {xml_path}")
    print(f"  刚体数量 (nbody):                 {model.nbody}")
    print(f"  关节数量 (nq):                    {model.nq}")
    print(f"  自由度 (nv):                      {model.nv}")
    print(f"  执行器 (nu):                      {model.nu}")
    print("")
    print(f"  --- 碰撞 (最关键) ---")
    print(f"  几何体总数 (ngeom):             {model.ngeom}  <-- (非常重要)")
    print(f"  网格 (nmesh):                 {model.nmesh}")
    print(f"  网格顶点 (nmeshvert):      {model.nmeshvert}  <-- (显存杀手)")
    print(f"  网格面 (nmeshface):        {model.nmeshface}  <-- (显存杀手)")
    print("")
    print(f"  --- 求解器 ---")
    print(f"  最大约束数 (njmax):        {model.njmax}")
    print(f"  最大接触数 (nconmax):        {model.nconmax}")
    print("-" * 60)

# -----------------------------------------------------------------
# ⚙️ 1. 在这里配置您的测试
# -----------------------------------------------------------------

# *** 第一步: 测试 Panda (这个高精度版) ***
# XML_PATH = "/home/hwk/program/deformale_mjx/thirdparty/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/mjx_panda.xml"
# BATCH_SIZE = 16  # (!!!) 您的日志显示32失败了，请从16开始

# *** 第二步: 测试您的模型 ***
# (取消注释下面两行来进行您的测试)
# XML_PATH = "/home/hwk/program/deformale_mjx/thirdparty/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/mjx_panda.xml"
# XML_PATH = "/home/hwk/program/deformale_mjx/assets/mujoco_model/dual_piper_with_cloth_simple.xml"
XML_PATH = "/home/hwk/program/deformale_mjx/assets/mujoco_model/test.xml"

BATCH_SIZE = 32                          # (!!!) 同样从一个小值开始

# 运行多少步来计算平均SPS
NUM_STEPS_TO_RUN = 1000
# -----------------------------------------------------------------

def run_benchmark():
    if not os.path.exists(XML_PATH):
        print(f"错误: XML 文件未找到 {XML_PATH}")
        return

    # 1. 加载模型并打印统计数据
    try:
        if 'flexcomp' in open(XML_PATH).read():
             print("警告: 正在加载包含 <flexcomp> 的XML。MJx 不支持 <flexcomp>。")
             print("     MJx 将自动忽略布料，仅测试刚体性能。")

        mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        print_model_statistics(mj_model, XML_PATH)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 将模型和数据 "put" 到 JAX/GPU
    mj_data = mujoco.MjData(mj_model)
    try:
        print("正在将模型传输到 JAX/MJX...")
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.put_data(mj_model, mj_data)
        print("传输完成。")
    except Exception as e:
        print(f"错误: 模型传输到 MJX 失败 (模型可能包含 MJX 不支持的特性): {e}")
        return

    # 3. 创建 JAX 批处理
    print(f"正在创建 {BATCH_SIZE} 个环境的 JAX 批处理...")
    try:
        batch_array = jnp.arange(BATCH_SIZE)
        mjx_data_batch = jax.vmap(lambda _: mjx_data)(batch_array)
    except Exception as e:
        print(f"错误: 创建批处理失败 (很可能是显存溢出 OOM): {e}")
        return

    # 4. 创建 JIT 编译的并行 step 函数
    def single_step(data, action):
        data_with_ctrl = data.replace(ctrl=action)
        return mjx.step(mjx_model, data_with_ctrl)

    vmapped_step = jax.vmap(single_step, in_axes=(0, 0), out_axes=0)
    parallel_step = jax.jit(vmapped_step)

    # 创建一个虚拟的动作批次 (全为0)
    dummy_actions = jnp.zeros((BATCH_SIZE, mjx_model.nu))

    # 5. "预热" JIT 编译器 (第一次运行会很慢)
    print("正在预热/编译 JIT (这可能需要几分钟)...")
    try:
        start_compile = time.time()
        
        # --- [BUG 修复 1] ---
        # 调用 JIT 函数
        mjx_data_batch = parallel_step(mjx_data_batch, dummy_actions)
        # 在一个 *数组* 上调用 .block_until_ready() 来等待编译完成
        mjx_data_batch.qpos.block_until_ready() 
        # --- [修复结束] ---
        
        compile_time = time.time() - start_compile
        print(f"JIT 编译完成，耗时 {compile_time:.2f} 秒。")
        
    except jax.errors.JaxRuntimeError as e:
        # 捕获真正的 JAX 运行时错误 (这才是 OOM)
        print(f"错误: JIT 编译失败 (很可能是显存溢出 OOM): {e}")
        print(f"  显存不足以编译 {BATCH_SIZE} 个并行的 '{XML_PATH}' 模型。")
        return
    except Exception as e:
        # 捕获其他 Python 错误 (比如我的bug)
        print(f"错误: JIT 预热期间发生未知错误: {e}")
        return

    # 6. 运行基准测试
    print(f"开始基准测试... (运行 {NUM_STEPS_TO_RUN} 步)")
    start_benchmark = time.time()
    
    for _ in range(NUM_STEPS_TO_RUN):
        mjx_data_batch = parallel_step(mjx_data_batch, dummy_actions)
    
    # --- [BUG 修复 2] ---
    # 确保所有JAX操作都已完成
    mjx_data_batch.qpos.block_until_ready()
    # --- [修复结束] ---
    
    end_benchmark = time.time()
    
    total_time = end_benchmark - start_benchmark
    total_steps = BATCH_SIZE * NUM_STEPS_TO_RUN
    sps = total_steps / total_time

    print("\n" + "=" * 60)
    print("--- 🏁 基准测试完成 ---")
    print(f"  XML 文件:         {XML_PATH}")
    print(f"  批处理大小:       {BATCH_SIZE}")
    print(f"  总运行步数:       {total_steps:,}")
    print(f"  总耗时:           {total_time:.4f} 秒")
    print(f"  SPS (步/秒):      {sps:,.0f}  <-- (这是您的性能指标)")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()