"""  
视频录制模块  
============  
  
提供训练和评估过程的视频录制功能  
"""  
  
import os  
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.animation import FuncAnimation, FFMpegWriter  
from mpl_toolkits.mplot3d import Axes3D  
from typing import List, Optional, Dict, Callable  
import warnings  
  
  
class VideoRecorder:  
    """  
    视频录制器  
      
    录制AUV运动的视频  
    """  
      
    def __init__(self,   
                 save_dir: str = "visualization/videos",  
                 fps: int = 30,  
                 dpi: int = 100):  
        """  
        初始化  
          
        Args:  
            save_dir: 视频保存目录  
            fps: 帧率  
            dpi: 图像分辨率  
        """  
        self.save_dir = save_dir  
        self.fps = fps  
        self.dpi = dpi  
          
        os.makedirs(save_dir, exist_ok=True)  
          
        # 帧缓冲区  
        self.frames = []  
        self.is_recording = False  
          
    def start_recording(self):  
        """开始录制"""  
        self.frames = []  
        self.is_recording = True  
          
    def add_frame(self, frame_data: Dict):  
        """  
        添加一帧  
          
        Args:  
            frame_data: 帧数据字典，包含位置、姿态等  
        """  
        if self.is_recording:  
            self.frames.append(frame_data.copy())  
              
    def stop_recording(self):  
        """停止录制"""  
        self.is_recording = False  
          
    def save_video_2d(self,   
                      filename: str = "trajectory_2d.mp4",  
                      trail_length: int = 50) -> str:  
        """  
        保存2D轨迹视频  
          
        Args:  
            filename: 文件名  
            trail_length: 轨迹尾巴长度  
              
        Returns:  
            视频文件路径  
        """  
        if not self.frames:  
            print("No frames to save!")  
            return ""  
              
        # 提取数据  
        positions = np.array([f['position'] for f in self.frames])  
        orientations = np.array([f.get('orientation', [0, 0, 0]) for f in self.frames])  
          
        # 创建图形  
        fig, ax = plt.subplots(figsize=(10, 10))  
          
        # 确定边界  
        margin = 2.0  
        x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin  
        y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin  
          
        # 使边界对称  
        x_range = max(abs(x_min), abs(x_max))  
        y_range = max(abs(y_min), abs(y_max))  
        max_range = max(x_range, y_range)  
          
        ax.set_xlim(-max_range, max_range)  
        ax.set_ylim(-max_range, max_range)  
        ax.set_aspect('equal')  
        ax.grid(True)  
        ax.set_xlabel('X (m)')  
        ax.set_ylabel('Y (m)')  
        ax.set_title('AUV Trajectory (Top View)')  
          
        # 初始化绘图元素  
        trail, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5)  
        robot, = ax.plot([], [], 'bo', markersize=15)  
        direction, = ax.plot([], [], 'r-', linewidth=3)  
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,   
                           fontsize=12, verticalalignment='top')  
          
        # 目标点（如果有）  
        if 'target' in self.frames[0]:  
            target = self.frames[0]['target']  
            ax.plot(target[0], target[1], 'r*', markersize=20, label='Target')  
            ax.legend()  
              
        def init():  
            trail.set_data([], [])  
            robot.set_data([], [])  
            direction.set_data([], [])  
            time_text.set_text('')  
            return trail, robot, direction, time_text  
          
        def update(frame_idx):  
            # 轨迹尾巴  
            start_idx = max(0, frame_idx - trail_length)  
            trail.set_data(positions[start_idx:frame_idx+1, 0],   
                          positions[start_idx:frame_idx+1, 1])  
              
            # 机器人位置  
            robot.set_data([positions[frame_idx, 0]], [positions[frame_idx, 1]])  
              
            # 朝向  
            yaw = orientations[frame_idx, 2] if len(orientations[frame_idx]) > 2 else 0  
            arrow_len = 1.0  
            dx = arrow_len * np.cos(yaw)  
            dy = arrow_len * np.sin(yaw)  
            direction.set_data([positions[frame_idx, 0], positions[frame_idx, 0] + dx],  
                              [positions[frame_idx, 1], positions[frame_idx, 1] + dy])  
              
            # 时间  
            time_text.set_text(f'Time: {frame_idx * 0.02:.2f}s')  
              
            return trail, robot, direction, time_text  
          
        # 创建动画  
        anim = FuncAnimation(fig, update, init_func=init,  
                            frames=len(self.frames), interval=1000/self.fps,  
                            blit=True)  
          
        # 保存  
        save_path = os.path.join(self.save_dir, filename)  
        try:  
            writer = FFMpegWriter(fps=self.fps, bitrate=1800)  
            anim.save(save_path, writer=writer, dpi=self.dpi)  
            print(f"Video saved to {save_path}")  
        except Exception as e:  
            print(f"Error saving video with FFmpeg: {e}")  
            # 尝试使用其他writer  
            try:  
                anim.save(save_path.replace('.mp4', '.gif'),   
                         writer='pillow', fps=self.fps)  
                print(f"GIF saved instead (FFmpeg not available)")  
            except Exception as e2:  
                print(f"Could not save animation: {e2}")  
                  
        plt.close(fig)  
        return save_path  
      
    def save_video_3d(self,   
                      filename: str = "trajectory_3d.mp4",  
                      trail_length: int = 50,  
                      rotate_view: bool = True) -> str:  
        """  
        保存3D轨迹视频  
          
        Args:  
            filename: 文件名  
            trail_length: 轨迹尾巴长度  
            rotate_view: 是否旋转视角  
              
        Returns:  
            视频文件路径  
        """  
        if not self.frames:  
            print("No frames to save!")  
            return ""  
              
        # 提取数据  
        positions = np.array([f['position'] for f in self.frames])  
          
        # 创建图形  
        fig = plt.figure(figsize=(12, 10))  
        ax = fig.add_subplot(111, projection='3d')  
          
        # 确定边界  
        margin = 2.0  
        x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin  
        y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin  
        z_min, z_max = positions[:, 2].min() - margin, positions[:, 2].max() + margin  
          
        ax.set_xlim(x_min, x_max)  
        ax.set_ylim(y_min, y_max)  
        ax.set_zlim(z_min, z_max)  
        ax.set_xlabel('X (m)')  
        ax.set_ylabel('Y (m)')  
        ax.set_zlabel('Z (m)')  
        ax.set_title('AUV 3D Trajectory')  
          
        # 初始化绘图元素  
        trail, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.5)  
        robot = ax.scatter([], [], [], c='blue', s=100)  
          
        def init():  
            trail.set_data([], [])  
            trail.set_3d_properties([])  
            return trail, robot  
          
        def update(frame_idx):  
            # 轨迹  
            start_idx = max(0, frame_idx - trail_length)  
            trail.set_data(positions[start_idx:frame_idx+1, 0],  
                          positions[start_idx:frame_idx+1, 1])  
            trail.set_3d_properties(positions[start_idx:frame_idx+1, 2])  
              
            # 机器人位置  
            robot._offsets3d = ([positions[frame_idx, 0]],   
                               [positions[frame_idx, 1]],   
                               [positions[frame_idx, 2]])  
              
            # 旋转视角  
            if rotate_view:  
                ax.view_init(elev=20, azim=frame_idx * 0.5)  
                  
            return trail, robot  
          
        # 创建动画  
        anim = FuncAnimation(fig, update, init_func=init,  
                            frames=len(self.frames), interval=1000/self.fps,  
                            blit=False)  
          
        # 保存  
        save_path = os.path.join(self.save_dir, filename)  
        try:  
            writer = FFMpegWriter(fps=self.fps, bitrate=1800)  
            anim.save(save_path, writer=writer, dpi=self.dpi)  
            print(f"Video saved to {save_path}")  
        except Exception as e:  
            print(f"Error saving video: {e}")  
              
        plt.close(fig)  
        return save_path  
      
    def save_pursuit_video(self,  
                           pursuer_frames: List[List[Dict]],  
                           evader_frames: List[Dict],  
                           filename: str = "pursuit_scenario.mp4") -> str:  
        """  
        保存追逃场景视频  
          
        Args:  
            pursuer_frames: 追击者帧数据列表  
            evader_frames: 逃逸者帧数据  
            filename: 文件名  
              
        Returns:  
            视频文件路径  
        """  
        if not evader_frames:  
            print("No frames to save!")  
            return ""  
              
        n_frames = len(evader_frames)  
        n_pursuers = len(pursuer_frames)  
          
        # 提取数据  
        evader_pos = np.array([f['position'] for f in evader_frames])  
        pursuer_pos = [np.array([f['position'] for f in pf]) for pf in pursuer_frames]  
          
        # 创建图形  
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))  
          
        # 确定边界  
        all_positions = [evader_pos] + pursuer_pos  
        all_x = np.concatenate([p[:, 0] for p in all_positions])  
        all_y = np.concatenate([p[:, 1] for p in all_positions])  
          
        margin = 5.0  
        x_min, x_max = all_x.min() - margin, all_x.max() + margin  
        y_min, y_max = all_y.min() - margin, all_y.max() + margin  
          
        # 2D视图  
        ax1 = axes[0]  
        ax1.set_xlim(x_min, x_max)  
        ax1.set_ylim(y_min, y_max)  
        ax1.set_aspect('equal')  
        ax1.grid(True)  
        ax1.set_xlabel('X (m)')  
        ax1.set_ylabel('Y (m)')  
        ax1.set_title('Pursuit-Evasion (Top View)')  
          
        # 距离图  
        ax2 = axes[1]  
        ax2.set_xlim(0, n_frames * 0.02)  
        ax2.set_ylabel('Distance (m)')  
        ax2.set_xlabel('Time (s)')  
        ax2.set_title('Distance to Evader')  
        ax2.grid(True)  
          
        # 计算所有距离  
        all_distances = []  
        for ppos in pursuer_pos:  
            min_len = min(len(ppos), len(evader_pos))  
            dist = np.linalg.norm(ppos[:min_len] - evader_pos[:min_len], axis=1)  
            all_distances.append(dist)  
              
        max_dist = max([d.max() for d in all_distances]) * 1.1  
        ax2.set_ylim(0, max_dist)  
          
        # 颜色  
        pursuer_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_pursuers))  
        evader_color = 'red'  
          
        # 初始化绘图元素  
        evader_trail, = ax1.plot([], [], '-', color=evader_color, linewidth=2, alpha=0.5)  
        evader_dot, = ax1.plot([], [], 'o', color=evader_color, markersize=15)  
          
        pursuer_trails = []  
        pursuer_dots = []  
        for i in range(n_pursuers):  
            trail, = ax1.plot([], [], '-', color=pursuer_colors[i], linewidth=2, alpha=0.5)  
            dot, = ax1.plot([], [], 'o', color=pursuer_colors[i], markersize=12)  
            pursuer_trails.append(trail)  
            pursuer_dots.append(dot)  
              
        distance_lines = []  
        for i in range(n_pursuers):  
            line, = ax2.plot([], [], '-', color=pursuer_colors[i], linewidth=2,  
                           label=f'Pursuer {i+1}')  
            distance_lines.append(line)  
        ax2.legend()  
          
        time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,  
                            fontsize=12, verticalalignment='top')  
          
        trail_length = 100  
          
        def update(frame_idx):  
            # 更新逃逸者  
            start_idx = max(0, frame_idx - trail_length)  
            evader_trail.set_data(evader_pos[start_idx:frame_idx+1, 0],  
                                 evader_pos[start_idx:frame_idx+1, 1])  
            evader_dot.set_data([evader_pos[frame_idx, 0]], [evader_pos[frame_idx, 1]])  
              
            # 更新追击者  
            for i in range(n_pursuers):  
                if frame_idx < len(pursuer_pos[i]):  
                    start_idx = max(0, frame_idx - trail_length)  
                    pursuer_trails[i].set_data(  
                        pursuer_pos[i][start_idx:frame_idx+1, 0],  
                        pursuer_pos[i][start_idx:frame_idx+1, 1]  
                    )  
                    pursuer_dots[i].set_data(  
                        [pursuer_pos[i][frame_idx, 0]],  
                        [pursuer_pos[i][frame_idx, 1]]  
                    )  
                      
            # 更新距离图  
            time_array = np.arange(frame_idx + 1) * 0.02  
            for i in range(n_pursuers):  
                if frame_idx < len(all_distances[i]):  
                    distance_lines[i].set_data(time_array, all_distances[i][:frame_idx+1])  
                      
            time_text.set_text(f'Time: {frame_idx * 0.02:.2f}s')  
              
            return ([evader_trail, evader_dot] + pursuer_trails +   
                    pursuer_dots + distance_lines + [time_text])  
          
        # 创建动画  
        anim = FuncAnimation(fig, update, frames=n_frames,  
                            interval=1000/self.fps, blit=True)  
          
        # 保存  
        save_path = os.path.join(self.save_dir, filename)  
        try:  
            writer = FFMpegWriter(fps=self.fps, bitrate=2400)  
            anim.save(save_path, writer=writer, dpi=self.dpi)  
            print(f"Video saved to {save_path}")  
        except Exception as e:  
            print(f"Error saving video: {e}")  
              
        plt.close(fig)  
        return save_path  
      
    def clear(self):  
        """清除帧缓冲区"""  
        self.frames = []  
        self.is_recording = False  
  
  
class RealtimeVisualizer:  
    """  
    实时可视化器  
      
    用于训练过程中的实时监控  
    """  
      
    def __init__(self, update_freq: int = 10):  
        """  
        初始化  
          
        Args:  
            update_freq: 更新频率（每多少步更新一次）  
        """  
        self.update_freq = update_freq  
        self.step_count = 0  
          
        # 启用交互模式  
        plt.ion()  
          
        # 创建图形  
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))  
          
        # 数据缓冲  
        self.reward_buffer = []  
        self.loss_buffer = []  
        self.position_buffer = []  
          
        self._setup_plots()  
          
    def _setup_plots(self):  
        """设置图形"""  
        # 奖励图  
        self.axes[0, 0].set_title('Episode Reward')  
        self.axes[0, 0].set_xlabel('Episode')  
        self.axes[0, 0].set_ylabel('Reward')  
        self.reward_line, = self.axes[0, 0].plot([], [], 'b-')  
          
        # 损失图  
        self.axes[0, 1].set_title('Training Loss')  
        self.axes[0, 1].set_xlabel('Step')  
        self.axes[0, 1].set_ylabel('Loss')  
        self.loss_line, = self.axes[0, 1].plot([], [], 'r-')  
          
        # 位置图  
        self.axes[1, 0].set_title('Current Position (XY)')  
        self.axes[1, 0].set_xlabel('X')  
        self.axes[1, 0].set_ylabel('Y')  
        self.position_scatter = self.axes[1, 0].scatter([], [], c='blue', s=20)  
          
        # 状态文本  
        self.axes[1, 1].axis('off')  
        self.status_text = self.axes[1, 1].text(0.1, 0.5, '', fontsize=12,  
                                                 family='monospace',  
                                                 verticalalignment='center')  
          
        plt.tight_layout()  
          
    def update(self,   
               reward: Optional[float] = None,  
               loss: Optional[float] = None,  
               position: Optional[np.ndarray] = None,  
               status: Optional[Dict] = None):  
        """  
        更新可视化  
          
        Args:  
            reward: 当前奖励  
            loss: 当前损失  
            position: 当前位置  
            status: 状态信息字典  
        """  
        self.step_count += 1  
          
        if reward is not None:  
            self.reward_buffer.append(reward)  
        if loss is not None:  
            self.loss_buffer.append(loss)  
        if position is not None:  
            self.position_buffer.append(position)  
              
        # 只在特定频率更新显示  
        if self.step_count % self.update_freq != 0:  
            return  
              
        # 更新奖励图  
        if self.reward_buffer:  
            self.reward_line.set_data(range(len(self.reward_buffer)),   
                                      self.reward_buffer)  
            self.axes[0, 0].relim()  
            self.axes[0, 0].autoscale_view()  
              
        # 更新损失图  
        if self.loss_buffer:  
            self.loss_line.set_data(range(len(self.loss_buffer)),   
                                    self.loss_buffer)  
            self.axes[0, 1].relim()  
            self.axes[0, 1].autoscale_view()  
              
        # 更新位置图  
        if self.position_buffer:  
            positions = np.array(self.position_buffer[-1000:])  # 只显示最近1000个点  
            self.position_scatter.set_offsets(positions[:, :2])  
            self.axes[1, 0].relim()  
            self.axes[1, 0].autoscale_view()  
              
        # 更新状态文本  
        if status is not None:  
            status_str = '\n'.join([f'{k}: {v}' for k, v in status.items()])  
            self.status_text.set_text(status_str)  
              
        # 刷新显示  
        self.fig.canvas.draw()  
        self.fig.canvas.flush_events()  
          
    def close(self):  
        """关闭可视化器"""  
        plt.ioff()  
        plt.close(self.fig)  
