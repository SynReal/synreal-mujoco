import roboticstoolbox as rtb
import numpy as np
import os
from spatialmath.base import r2q


class PiperRobot:
    def __init__(self, urdf_path, ee_link_name='link7'):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

        try:
            self.robot = rtb.ERobot.URDF(
                file_path=urdf_path,
                gripper=ee_link_name
            )
            print("Robot load Successfully.")

        except Exception as e:
            print(f"Error loading robot from URDF: {e}")
            print(f"Please ensure the URDF file is correctly formatted and the end effector link '{ee_link_name}' exists.")

            raise

    def get_fk_solution(self, joint_angles):
        """
        Calculate the forward kinematics solution for given joint angles.

        Args:
            joint_angles (list or np.ndarray): Joint angles matching the robot's degrees of freedom.

        Returns:
            np.ndarray: end effector position [x, y, z], Orientation in quaternion [w, x, y, z].
        """
        if len(joint_angles) != self.robot.n:
            raise ValueError(f"提供的关节角度数量 ({len(joint_angles)}) 与机器人的自由度 ({self.robot.n}) 不匹配。")

        all_poses = self.robot.fkine_all(joint_angles)
        fk_matrix = all_poses[-1]

        rotation_matrix = fk_matrix.R
        orientation_wxyz = r2q(rotation_matrix)

        return fk_matrix.t, orientation_wxyz


class K1DualArmRobot:
    def __init__(self, urdf_path):
        """
        初始化K1双臂机器人，并维护一个内部的完整关节状态。
        参数:
            urdf_path (str): URDF文件路径。
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF文件未找到: {urdf_path}")

        self.robot = rtb.ERobot.URDF(file_path=urdf_path)
        print(f"K1 Robot model loaded from: {urdf_path}, found {self.robot.n} DoFs.")
        # print(self.robot)

        # --- 关节和连杆的硬定义 ---
        self.right_arm_indices = [0, 1, 2, 3, 4, 5, 6]
        self.right_gripper_indices = [7, 8]
        self.left_arm_indices = [9, 10, 11, 12, 13, 14, 15]
        self.left_gripper_indices = [16, 17]

        self.left_ee_link = "lt"
        self.right_ee_link = "rt"
        self.left_ee_link = "left_gripper_adapter"
        self.right_ee_link = "right_gripper_adapter"

        # --- 内部状态管理 ---
        self.q_home = np.zeros(self.robot.n)
        self.q_current = self.q_home.copy()

    def update_full_state(self, q_full):
        if len(q_full) != self.robot.n:
            raise ValueError(f"提供的完整状态向量长度 ({len(q_full)}) 与机器人自由度 ({self.robot.n}) 不匹配。")
        self.q_current = np.array(q_full)
        print("Robot full state updated.")

    def _format_pos_and_orientation(self, fk_matrix):
        """【辅助函数】将SE3矩阵转换为 (位置, 四元数) 元组。"""
        position = fk_matrix.t
        orientation_wxyz = r2q(fk_matrix.R)
        return position, orientation_wxyz

    def get_fk_solution(self, joint_angles, arm='both'):
        """
        【最简化FK接口】
        Args:
            joint_angles: 关节角度。
                - arm='both': 完整的18个关节角度的扁平列表。
                - arm='left'/'right': 对应手臂的7个或9个关节角度的列表。
            arm (str): 'left', 'right', 或 'both'。
        """
        # 统一转为numpy数组
        joint_angles = np.array(joint_angles)

        if arm == 'both':
            # 直接使用传入的完整关节向量
            q_calc = joint_angles

            # 计算两次FK
            fk_L = self.robot.fkine(q_calc, end=self.left_ee_link)
            fk_R = self.robot.fkine(q_calc, end=self.right_ee_link)

            # 返回结果
            pos_L, ori_L = fk_L.t, r2q(fk_L.R)
            pos_R, ori_R = fk_R.t, r2q(fk_R.R)
            return {'left': (pos_L, ori_L), 'right': (pos_R, ori_R)}

        # 对于单臂计算，从当前状态开始
        q_calc = self.q_current.copy()

        if arm == 'left':
            # 直接更新左臂部分的关节值
            q_calc[self.left_arm_indices] = joint_angles[:7]
            if len(joint_angles) == 9:
                q_calc[self.left_gripper_indices] = joint_angles[7:]

            # 计算FK并返回
            fk_L = self.robot.fkine(q_calc, end=self.left_ee_link)
            return fk_L.t, r2q(fk_L.R)

        elif arm == 'right':
            # 直接更新右臂部分的关节值
            q_calc[self.right_arm_indices] = joint_angles[:7]
            if len(joint_angles) == 9:
                q_calc[self.right_gripper_indices] = joint_angles[7:]

            # 计算FK并返回
            fk_R = self.robot.fkine(q_calc, end=self.right_ee_link)
            return fk_R.t, r2q(fk_R.R)
        else:
            raise ValueError(f"参数 'arm' 必须是 'left', 'right', 或 'both'")

if __name__ == "__main__":
    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    JOINT_ANGLES_TO_TEST = [-0.45048693, 1.22996843, -0.28687977, -0.11801916, 0.53476888, 0.021485]
    URDF_FILE_PATH = os.path.join(PROJECT_ROOT,"assets","piper_description","piper_with_gripper.urdf")
    piper_robot = PiperRobot(URDF_FILE_PATH, ee_link_name="link7")
    t,q = piper_robot.get_fk_solution(JOINT_ANGLES_TO_TEST+ [0.0])
    print(f"\n[PiperRobot] Forward Kinematics for end effector:\n  -> Position: {t}, Orientation: {q}")

    # Position: [ 0.14182882 -0.08372661  0.00450508]

    q_full_robot = np.array([-1.768, -58.3,  -54.936,  -118.927,  -127.422,      42.904,  157.592,
                             0.0, 0.0,
                            1.792, -58.533,   54.750,  -119.007,  127.363,     -43.143, -157.583,
                            0.0, 0.0])*np.pi/180

    K1_urdf_path = os.path.join(PROJECT_ROOT, "assets", "k1_description", "k1_pgc_fix_v1.urdf")
    k1_robot = K1DualArmRobot(K1_urdf_path)
    both_poses = k1_robot.get_fk_solution(q_full_robot)  # arm='both' 是默认值
    print("--- 同时计算双臂 ---")
    print(f"左臂位置: {both_poses['left'][0]}")
    print(f"右臂位置: {both_poses['right'][0]}")
    print("-" * 20)
    left_q = q_full_robot[k1_robot.left_arm_indices + k1_robot.left_gripper_indices]
    print("left_q", left_q)
    pos_L, ori_L = k1_robot.get_fk_solution(left_q, arm='left')
    print("--- 只计算左臂 ---")
    print(f"左臂位置: {pos_L}")
    print(f"左臂姿态: {ori_L}")
    print("-" * 20)

    right_q = q_full_robot[k1_robot.right_arm_indices + k1_robot.right_gripper_indices]
    print("right_q",right_q)
    pos_R, ori_R = k1_robot.get_fk_solution(right_q, arm='right')
    print("--- 只计算右臂 ---")
    print(f"右臂位置: {pos_R}")
    print(f"右臂姿态: {ori_R}")
    print("-" * 20)