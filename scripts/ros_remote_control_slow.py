import sys
import os
import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from typing import Dict, Tuple, Optional, List, Any, Literal, TypedDict, cast
from pynput import keyboard

try:
    import rospy
    from sensor_msgs.msg import JointState
except ImportError:
    print("Error: 'rospy' or 'sensor_msgs' not found.")
    exit(1)

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()

class JointStateData(TypedDict):
    time: float
    positions: List[float]


class MujocoRosController:
    
    def __init__(self, xml_path):
        self.lock = threading.Lock()
        
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            print(f"Error loading XML: {e}")
            exit(1)
        
        # 硬编码关节名称顺序
        self.left_arm_joints = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
            'joint7', 'joint8'  # gripper
        ]
        self.right_arm_joints = [
            'joint1_arm2', 'joint2_arm2', 'joint3_arm2', 
            'joint4_arm2', 'joint5_arm2', 'joint6_arm2',
            'joint7_arm2', 'joint8_arm2'  # gripper
        ]
        

        #  状态变量
        self.run_mode = "realtime"  # 'realtime'  或 'physics_priority' (慢速)
        
        self.gripper_factor = 0.35/0.688  # gripper 修正因子

        self.dof = 6 # 每个手臂的自由度 (不包括gripper)
        self.latest_left_positions: List[float] = [0.0] * (self.dof + 1)  # (j1-j6, gripper)
        self.latest_right_positions: List[float] = [0.0] * (self.dof + 1) # (j1-j6, gripper)

        # 2. 物理模式下的插值数据缓冲区
        self.left_data_buffer: List[JointStateData] = []
        self.right_data_buffer: List[JointStateData] = []

        # 3. 物理模式下的插值索引跟踪器
        self.time_trackers_idx: Dict[str, int] = {'left': 0, 'right': 0}
        
        self.master_time_offset: float = 0.0 # 仿真时间与墙上时间的时间戳偏移
        self.latest_msg_time: float = 0.0 # 最新消息时间戳



        # 初始化为当前位置
        self._init_positions()
        
        print(f"MuJoCo-ROS控制器初始化成功")
        print(f"Left arm: {self.left_arm_joints[:6]} + gripper")
        print(f"Right arm: {self.right_arm_joints[:6]} + gripper")
        print("-" * 30)
        print(f"  当前模式: {self.run_mode} (瞬时更新)")
        print(f"  [按键]：按 '空格键' 切换 '插值模式' (物理真实, 慢)")
        print("-" * 30)

    def _init_positions(self):
        """从模型初始位置初始化"""
        for i in range(self.dof):
            q_addr = self.model.joint(self.left_arm_joints[i]).qposadr
            self.latest_left_positions[i] = self.data.qpos[q_addr][0]
            q_addr_r = self.model.joint(self.right_arm_joints[i]).qposadr
            self.latest_right_positions[i] = self.data.qpos[q_addr_r][0]
        
        self.latest_left_positions[self.dof] = 0.0
        self.latest_right_positions[self.dof] = 0.0


    def left_arm_callback(self, msg):
        """左臂话题回调"""
        positions = list(msg.position)[:self.dof + 1]
        if len(positions) < self.dof + 1: return
        msg_time = msg.header.stamp.to_sec()
        self.latest_msg_time = msg_time
        with self.lock:
            self.latest_left_positions = positions
            self.left_data_buffer.append({
                'time': msg_time,
                'positions': positions
            })

    def right_arm_callback(self, msg):
        """右臂话题回调"""
        positions = list(msg.position)[:self.dof + 1]
        if len(positions) < self.dof + 1: return
        msg_time = msg.header.stamp.to_sec()
        self.latest_msg_time = msg_time
        with self.lock:
            self.latest_right_positions = positions
            self.right_data_buffer.append({
                'time': msg_time,
                'positions': positions
            })


    def start_ros_listener(self):
        """启动ROS监听器"""
        rospy.init_node('mujoco_ros_listener', anonymous=True)
        
        # 订阅两个话题
        rospy.Subscriber('/left_arm/joint_states', JointState, self.left_arm_callback)
        rospy.Subscriber('/right_arm/joint_states', JointState, self.right_arm_callback)
        
        print("\n✓ ROS监听器已启动")
        print("  订阅话题:")
        print("    - /left_arm/joint_states")
        print("    - /right_arm/joint_states")
        print("\n  消息格式 (JointState):")
        print("    - position: [j1, j2, j3, j4, j5, j6, gripper]")
        print("    - gripper: 0(闭合) ~ 0.035(张开)\n")
        
        ros_thread = threading.Thread(target=rospy.spin, daemon=True)
        ros_thread.start()
    
    def toggle_mode(self):
            with self.lock:
                current_ros_time = rospy.get_time()
                if self.run_mode == "realtime":
                    self.run_mode = "physical"
                    self.master_time_offset = self.latest_msg_time - self.data.time
                    self._reset_trackers_to_time(current_ros_time)
                    print(f"\n--- 模式: 切换到 [物理模式] (物理优先) ---")
                else:
                    self.run_mode = "realtime"
                    print(f"\n--- 模式: 切换到 [实时模式] (瞬时更新) ---")

    def start_pynput_listener(self):
        def on_press(key):
            try:
                if key.char == 'q':
                    print("[Keyboard] 'q' pressed. Signaling quit...")
                    self.simulation_running = False # 设置标志
                    if self.viewer_handle:
                         # 尝试从 pynput 线程安全地关闭 viewer
                         self.viewer_handle.close()
                    return False # 停止监听器
            except AttributeError:
                # 监听 'SPACE'
                if key == keyboard.Key.space:
                    print("[Keyboard] SPACE pressed. Toggling mode...")
                    self.toggle_mode() # 直接调用

        # 创建并启动监听器
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("\n✓ Pynput 全局键盘监听器已启动")

    def _reset_trackers_to_time(self, current_wall_time: float):
        idx_left = 0
        while idx_left < len(self.left_data_buffer) - 2 and \
              self.left_data_buffer[idx_left + 1]['time'] < current_wall_time:
            idx_left += 1
        self.time_trackers_idx['left'] = idx_left
        
        idx_right = 0
        while idx_right < len(self.right_data_buffer) - 2 and \
              self.right_data_buffer[idx_right + 1]['time'] < current_wall_time:
            idx_right += 1
        self.time_trackers_idx['right'] = idx_right

    def _get_interpolated_positions(
        self, 
        target_time: float, 
        arm_data: List[JointStateData], 
        current_idx: int
    ) -> Tuple[Optional[List[float]], int]:
        if not arm_data: return None, current_idx
        new_idx = current_idx
        while new_idx < len(arm_data) - 2 and arm_data[new_idx + 1]['time'] < target_time:
            new_idx += 1
        if target_time <= arm_data[0]['time']: return arm_data[0]['positions'], new_idx
        if target_time >= arm_data[-1]['time']: return arm_data[-1]['positions'], new_idx
        
        p0, p1 = arm_data[new_idx], arm_data[new_idx + 1]
        t0, pos0_list = p0['time'], p0['positions']
        t1, pos1_list = p1['time'], p1['positions']
        interval = t1 - t0
        if interval <= 0: return pos0_list, new_idx
        
        alpha = (target_time - t0) / interval
        interpolated_pos_arr = np.array(pos0_list) + alpha * (np.array(pos1_list) - np.array(pos0_list))
        return interpolated_pos_arr.tolist(), new_idx

    def _apply_arm_state(self, joint_names: List[str], positions: List[float]) -> None:
        """
        统一的函数，用于将关节值应用到qpos
        """
        for i in range(self.dof): # 臂关节
            try:
                self.data.qpos[self.model.joint(joint_names[i]).qposadr] = positions[i]
            except: pass
        gripper_value = positions[self.dof] # Gripper

        # TODO: 数据好可以干掉这个部分
        gripper_value = gripper_value * self.gripper_factor

        try:
            self.data.qpos[self.model.joint(joint_names[self.dof]).qposadr] = gripper_value
            self.data.qpos[self.model.joint(joint_names[self.dof + 1]).qposadr] = -gripper_value
        except: pass

    def run_simulation(self):
        """运行仿真循环"""
        print("启动MuJoCo可视化器...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer :
            
            self.viewer_handle = viewer 
            
            while viewer.is_running() and not rospy.is_shutdown():
                step_start_wall_time = time.time()
                
               # --- [模式 1: 实时模式] ---
                if self.run_mode == "realtime":
                    with self.lock:
                        self._apply_arm_state(self.left_arm_joints, self.latest_left_positions)
                        self._apply_arm_state(self.right_arm_joints, self.latest_right_positions)
                    
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start_wall_time)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                
              # --- [模式 2: 插值模式] ---
                else: 
                    target_wall_time = self.data.time + self.master_time_offset
                    
                    with self.lock:
                        # 1. 获取左臂插值
                        left_pos, new_left_idx = self._get_interpolated_positions(
                            target_wall_time,
                            self.left_data_buffer,
                            self.time_trackers_idx['left']
                        )
                        self.time_trackers_idx['left'] = new_left_idx # 更新索引
                        
                        # 2. 获取右臂插值
                        right_pos, new_right_idx = self._get_interpolated_positions(
                            target_wall_time,
                            self.right_data_buffer,
                            self.time_trackers_idx['right']
                        )
                        self.time_trackers_idx['right'] = new_right_idx # 更新索引
                        
                        # 3. 应用位置 (使用和实时模式一样的函数)
                        if left_pos:
                            self._apply_arm_state(self.left_arm_joints, left_pos)
                        if right_pos:
                            self._apply_arm_state(self.right_arm_joints, right_pos)

                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()


if __name__ == "__main__":
    # xml_file_path = os.path.join(PROJECT_ROOT, "assets/mujoco_model/dual_piper_with_cloth_simple.xml")
    xml_file_path  = os.path.join(PROJECT_ROOT, "assets/mujoco_model/Style3dCloth_NewPiper_Camera.xml")
    controller = MujocoRosController(xml_file_path)
    controller.start_ros_listener()
    controller.start_pynput_listener()
    controller.run_simulation()