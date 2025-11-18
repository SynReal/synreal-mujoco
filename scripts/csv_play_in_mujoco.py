import os.path as osp
import os
import sys
from pathlib import Path
import mujoco
import ast
import time
import pandas as pd
from mujoco import viewer
from typing import Dict, Tuple, Optional, Union, List, Any, Literal, TypedDict
from typing import cast
import numpy as np
import imageio
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from common.csv_data_utils import load_processed_data, JointState

from loguru import logger
from omegaconf import DictConfig, OmegaConf

ArmType = Literal["left", "right"]

class ArmConfigData(TypedDict):
    """Stores configuration and data for one arm."""
    name: ArmType
    mujoco_joint_names: List[str]
    data: List[JointState]

class DualArmController:
    """
    CSV-based dual-arm controller.
    Loads pre-processed CSV data and replays it in MuJoCo with linear interpolation.
    """
    def __init__(
            self,
            model_path: str,
            left_csv: str,
            right_csv: str,
            robot_type: Literal["piper", "k1"] = "piper",
            data_play_start_time: float = 0.0,
    ) -> None:
        """Initialize the controller with separate left/right arms."""
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Robot configuration
        self.robot_type = robot_type

        # --- Left Arm Data ---
        self.left_arm_joint_names: List[str] = []
        self.left_arm_data: List[JointState] = []
        self.left_arm_tracker_idx: int = 0

        # --- Right Arm Data ---
        self.right_arm_joint_names: List[str] = []
        self.right_arm_data: List[JointState] = []
        self.right_arm_tracker_idx: int = 0

        # Playback timeline
        self.playback_start_time_abs: Optional[float] = None
        self.playback_end_time_abs: Optional[float] = None

        # Load configuration and data
        self._init_arms_config(left_csv, right_csv, data_play_start_time)

        np.set_printoptions(precision=4, suppress=True)
        logger.success("Controller initialized successfully.")

    def _init_arms_config(
        self, 
        left_csv: str, 
        right_csv: str, 
        start_time_offset: float
    ) -> None:
        """Load and configure arm data from CSV files."""
        
        # Define joint naming based on robot type
        if self.robot_type == "piper":
            self.left_arm_joint_names = [f'joint{i}' for i in range(1, 7)] + ['joint7', 'joint8']
            self.right_arm_joint_names = [f'joint{i}_arm2' for i in range(1, 7)] + ['joint7_arm2', 'joint8_arm2']
        elif self.robot_type == "k1":
            self.left_arm_joint_names = [f'l-j{i+1}' for i in range(7)] + ['left_finger1_joint', 'left_finger2_joint']
            self.right_arm_joint_names = [f'r-j{i+1}' for i in range(7)] + ['right_finger1_joint', 'right_finger2_joint']
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")
        
        # Load left arm data
        if left_csv and osp.exists(left_csv):
            try:
                self.left_arm_data = load_processed_data(
                    csv_path=left_csv,
                    start_time_offset=start_time_offset
                )
                logger.info(f"Loaded {len(self.left_arm_data)} data points for left arm")
            except Exception as e:
                logger.error(f"Failed to load left arm data: {e}")
        
        # Load right arm data
        if right_csv and osp.exists(right_csv):
            try:
                self.right_arm_data = load_processed_data(
                    csv_path=right_csv,
                    start_time_offset=start_time_offset
                )
                logger.info(f"Loaded {len(self.right_arm_data)} data points for right arm")
            except Exception as e:
                logger.error(f"Failed to load right arm data: {e}")
        
        # Determine global playback timeline
        all_times = []
        if self.left_arm_data:
            all_times.extend([self.left_arm_data[0]['time'], self.left_arm_data[-1]['time']])
        if self.right_arm_data:
            all_times.extend([self.right_arm_data[0]['time'], self.right_arm_data[-1]['time']])
        
        if all_times:
            self.playback_start_time_abs = min(all_times[::2])  # Start times
            self.playback_end_time_abs = max(all_times[1::2])   # End times
            duration = self.playback_end_time_abs - self.playback_start_time_abs
            
            logger.info("-" * 50)
            logger.info("Playback Timeline:")
            logger.info(f"  Start: {self.playback_start_time_abs:.4f}s")
            logger.info(f"  End:   {self.playback_end_time_abs:.4f}s")
            logger.info(f"  Duration: {duration:.2f}s")
            logger.info("-" * 50)
        else:
            logger.error("No valid arm data loaded!")
    
    
    
    
    def step(self) -> None:
        """
        Steps the simulation forward by one time step, updating the arms' joint positions
        """
        current_absolute_time = self.get_master_start_time() + self.data.time
        # Update interpolated joint positions for each arm
        for arm_name, arm_config in self.arms.items():
            if not arm_config['data']:
                continue

            interpolated_positions = self._get_interpolated_positions(
                target_time=current_absolute_time,
                arm_data=arm_config['data'],
                tracker=self.time_trackers[arm_name]
            )

            if interpolated_positions:
                self._apply_arm_state(arm_config, interpolated_positions)

        # Perform one physics simulation step
        mujoco.mj_step(self.model, self.data)

    # ===================================================================
    # Dual Arm Controller Methods
    # ===================================================================
    def _apply_arm_state(self, joint_names: List[str], positions: List[float]) -> None:
        """
        Applies the interpolated list of joint positions to the MuJoCo model,
        using the stored joint names for mapping by index.
        """
        num_arm_joints = len(joint_names) - 2  # Exclude 2 gripper joints
        
        if len(positions) != num_arm_joints + 1:
            return
        
        # Apply arm joints
        for i, name in enumerate(joint_names[:num_arm_joints]):
            try:
                self.data.qpos[self.model.joint(name).qposadr] = positions[i]
            except KeyError:
                pass
        
        # Apply gripper (mirrored)
        gripper_value = positions[-1]
        try:
            self.data.qpos[self.model.joint(joint_names[-2]).qposadr] = gripper_value
            self.data.qpos[self.model.joint(joint_names[-1]).qposadr] = -gripper_value
        except KeyError:
            pass

    def _get_interpolated_positions(
        self, 
        target_time: float, 
        arm_data: List[JointState], 
        current_idx: int
    ) -> tuple[Optional[List[float]], int]:
        """Get interpolated joint positions at target_time."""
        if not arm_data:
            return None, current_idx
        
        # Advance index
        new_idx = current_idx
        while new_idx < len(arm_data) - 2 and arm_data[new_idx + 1]['time'] < target_time:
            new_idx += 1
        
        # Boundary cases
        if target_time <= arm_data[0]['time']:
            return arm_data[0]['positions'], new_idx
        if target_time >= arm_data[-1]['time']:
            return arm_data[-1]['positions'], new_idx
        
        # Linear interpolation
        p0, p1 = arm_data[new_idx], arm_data[new_idx + 1]
        t0, pos0 = p0['time'], np.array(p0['positions'])
        t1, pos1 = p1['time'], np.array(p1['positions'])
        
        alpha = (target_time - t0) / (t1 - t0) if t1 > t0 else 0.0
        interpolated = pos0 + alpha * (pos1 - pos0)
        
        return interpolated.tolist(), new_idx

    def step(self) -> None:
        """Execute one simulation step with interpolated positions."""
        target_time = self.playback_start_time_abs + self.data.time
        
        # Update left arm
        if self.left_arm_data:
            left_pos, new_left_idx = self._get_interpolated_positions(
                target_time,
                self.left_arm_data,
                self.left_arm_tracker_idx
            )
            if left_pos:
                self._apply_arm_state(self.left_arm_joint_names, left_pos)
                self.left_arm_tracker_idx = new_left_idx
        
        # Update right arm
        if self.right_arm_data:
            right_pos, new_right_idx = self._get_interpolated_positions(
                target_time,
                self.right_arm_data,
                self.right_arm_tracker_idx
            )
            if right_pos:
                self._apply_arm_state(self.right_arm_joint_names, right_pos)
                self.right_arm_tracker_idx = new_right_idx
        
        # Step physics
        mujoco.mj_step(self.model, self.data)

    # ===================================================================
    # Online and offline run simulation in mujoco
    # ===================================================================
    def run(self) -> None:
        """Run interactive simulation with viewer."""
        if self.playback_start_time_abs is None:
            logger.error("No playback data loaded. Cannot run simulation.")
            return
        
        logger.info("Starting interactive simulation (press ESC to exit)...")
        
        with viewer.launch_passive(self.model, self.data) as v:
            sim_dt = self.model.opt.timestep
            
            while v.is_running():
                step_start = time.time()
                
                self.step()
                v.sync()
                
                # Real-time pacing
                elapsed = time.time() - step_start
                if elapsed < sim_dt:
                    time.sleep(sim_dt - elapsed)
        
        logger.success("Simulation ended.")


    def record_video(
        self,
        output_path: str,
        speed: float = 1.0,
        fps: int = 60,
        width: int = 1920,
        height: int = 1080,
        camera_name: Optional[str] = None
    ) -> None:
        """Record simulation to video file."""
        if self.playback_start_time_abs is None:
            logger.error("No playback data loaded. Cannot record.")
            return
        
        if speed <= 0:
            raise ValueError("Speed must be positive")
        
        # Calculate video parameters
        data_duration = self.playback_end_time_abs - self.playback_start_time_abs
        sim_dt = self.model.opt.timestep
        total_sim_steps = int(data_duration / sim_dt)
        total_video_frames = int((data_duration / speed) * fps)
        render_interval = data_duration / total_video_frames if total_video_frames > 0 else float('inf')
        
        logger.info(f"Recording video: {total_video_frames} frames over {data_duration:.2f}s")
        logger.info(f"  Speed: {speed}x, FPS: {fps}, Render every {render_interval:.4f}s")
        
        # Setup renderer
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
        except ValueError:
            logger.warning("Resolution too high, falling back to 640x480")
            renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        self.left_arm_tracker_idx = 0
        self.right_arm_tracker_idx = 0
        
        # Record frames
        frames = []
        next_render_time = 0.0
        
        for _ in tqdm(range(total_sim_steps), desc="Simulating"):
            if self.data.time >= next_render_time:
                if camera_name:
                    renderer.update_scene(self.data, camera=camera_name)
                else:
                    renderer.update_scene(self.data)
                frames.append(renderer.render())
                next_render_time += render_interval
            
            self.step()
        
        # Save video
        logger.info(f"Saving {len(frames)} frames to {output_path}...")
        try:
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in tqdm(frames, desc="Writing"):
                    writer.append_data(frame)
            logger.success(f"Video saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
        finally:
            renderer.close()

if __name__ == "__main__":


    try:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        PROJECT_ROOT = os.getcwd()

    # blue dress
    # data_path = os.path.join(PROJECT_ROOT, "data", "blue_dress_2025-11-18-18-13-48")
    # model_file=os.path.join(PROJECT_ROOT, "assets", "mujoco_model", "test_blue_dress.xml")

    # green tshirt
    data_path = os.path.join(PROJECT_ROOT, "data", "green_tshirt_2025-11-18-17-55-38")
    model_file=os.path.join(PROJECT_ROOT, "assets", "mujoco_model", "test_green_tshirt.xml")

    target_camera = 'video_third_view'  # 你可以根据需要修改这个相机名称


    try:
        player = DualArmController(
            model_path= model_file,
            right_csv=f"{data_path}/joints/right_arm_joint_states_and_end_pose.csv",
            left_csv=f"{data_path}/joints/left_arm_joint_states_and_end_pose.csv",
            robot_type="piper",
            data_play_start_time= 3.0 # 例如，从CSV记录时间的2秒后开始播放
        )

        # 选项1: 交互式预览
        print("\nStarting interactive simulation...")
        player.run()


        # 示例 2: 录制一个与CSV数据时长一致的视频 (1倍速)
        print("\n--- Recording video at 1.0x speed (real-time match) ---")
        output_filename = f"{data_path}/simulation_{target_camera}_step_{player.model.opt.timestep}.mp4"
        # player.record_video(
        #     output_path= output_filename + ".mp4",
        #     speed=1.0,  # 倍速
        #     camera_name=target_camera,  # 使用指定的相机
        #     fps=30,
        # )



    except FileNotFoundError as e:
        print(f"Error: File not found. Please check paths. Details: {e}")
    except ValueError as e:
        print(f"Error: Value error during initialization or data loading. Details: {e}")
    except Exception as e: # 其他未知错误
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()