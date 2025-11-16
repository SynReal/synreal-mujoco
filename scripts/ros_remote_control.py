import sys

# sys.path.append('/home/hwk/anaconda3/envs/mujo_ros_style3d/lib/python3.8/site-packages')
# sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time

try:
    import rospy
    from sensor_msgs.msg import JointState
except ImportError:
    print("Error: 'rospy' or 'sensor_msgs' not found.")
    exit(1)


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
        
        # 存储最新的关节位置
        self.latest_positions = {}
        
        # 初始化为当前位置
        self._init_positions()
        
        print(f"MuJoCo-ROS控制器初始化成功")
        print(f"Left arm: {self.left_arm_joints[:6]} + gripper")
        print(f"Right arm: {self.right_arm_joints[:6]} + gripper")

    def _init_positions(self):
        """从模型初始位置初始化"""
        for joint_name in self.left_arm_joints + self.right_arm_joints:
            try:
                joint_id = self.model.joint(joint_name).id
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.latest_positions[joint_name] = self.data.qpos[qpos_addr]
            except:
                self.latest_positions[joint_name] = 0.0

    def left_arm_callback(self, msg):
        """左臂话题回调"""
        with self.lock:
            self._update_arm(self.left_arm_joints, msg.position)

    def right_arm_callback(self, msg):
        """右臂话题回调"""
        with self.lock:
            self._update_arm(self.right_arm_joints, msg.position)

    def _update_arm(self, joint_names, positions):
        """
        更新一个手臂的关节位置
        期望输入: [j1, j2, j3, j4, j5, j6, gripper]
        - 前6个是手臂关节
        - 第7个是gripper值 (0~0.035)
        """
        positions = list(positions)
        
        if len(positions) < 7:
            print(f"Warning: Expected 7 values, got {len(positions)}")
            return
        
        # 前6个关节
        for i in range(6):
            self.latest_positions[joint_names[i]] = positions[i]
        
        # Gripper: joint7 和 joint8 (反向)
        gripper_value = positions[6]
        self.latest_positions[joint_names[6]] = gripper_value
        self.latest_positions[joint_names[7]] = -gripper_value

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

    def run_simulation(self):
        """运行仿真循环"""
        print("启动MuJoCo可视化器...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer :
            while viewer.is_running() and not rospy.is_shutdown():
                step_start = time.time()
                
                # 应用关节位置到仿真
                with self.lock:
                    for joint_name, position in self.latest_positions.items():
                        try:
                            joint_id = self.model.joint(joint_name).id
                            qpos_addr = self.model.jnt_qposadr[joint_id]
                            self.data.qpos[qpos_addr] = position
                        except:
                            pass
                
                # 仿真步进
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 保持实时速率
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    xml_file_path = "/home/hwk/program/deformale_mjx/assets/mujoco_model/dual_piper_with_cloth_simple.xml"
    
    controller = MujocoRosController(xml_file_path)
    controller.start_ros_listener()
    controller.run_simulation()