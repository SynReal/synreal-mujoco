#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import JointState
import math
import time

def talker():
    """
    一个简单的 ROS 节点，用于发布模拟的关节数据。
    它会使手臂的几个关节和夹爪做平滑的正弦波运动。
    """
    
    # 1. 初始化发布者
    #    (话题名称必须与 MuJoCo 脚本中订阅的完全一致)
    left_pub = rospy.Publisher('/left_arm/joint_states', JointState, queue_size=10)
    right_pub = rospy.Publisher('/right_arm/joint_states', JointState, queue_size=10)
    
    # 2. 初始化 ROS 节点
    rospy.init_node('mujoco_joint_tester', anonymous=True)
    
    # 3. 设置发布频率 (例如 50Hz)
    rate = rospy.Rate(50) 
    
    print("✓ ROS 关节测试发布器已启动")
    print("正在向 /left_arm/joint_states 和 /right_arm/joint_states 发布数据...")
    print("按 Ctrl+C 停止")
    
    start_time = time.time()

    while not rospy.is_shutdown():
        # 4. 创建消息
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        
        # 5. 生成随时间变化的关节角度 (使用正弦波)
        current_time = time.time() - start_time
        
        # 摆动幅度为 0.5 弧度
        pos_wave = 0.5 * math.sin(current_time * 1.5) # 1.5 是摆动速度
        
        # 夹爪在 0 到 0.035 之间摆动
        gripper_wave = 0.0175 * (1 + math.sin(current_time * 0.8)) # 0.8 是较慢的夹爪速度

        # 6. 填充 position 数组 (必须是 7 个值)
        # [j1, j2, j3, j4, j5, j6, gripper]
        # 我们让第2、第4关节和夹爪动起来
        joint_positions = [
            0.0,         # j1
            pos_wave,    # j2
            0.0,         # j3
            pos_wave,    # j4
            0.0,         # j5
            0.0,         # j6
            gripper_wave # gripper
        ]
        
        msg.position = joint_positions
        
        # 7. 发布消息
        #    (让两只手臂做一样的动作)
        left_pub.publish(msg)
        right_pub.publish(msg)
        
        # 8. 按频率休眠
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass