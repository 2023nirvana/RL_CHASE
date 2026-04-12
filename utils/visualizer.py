"""  
可视化模块  
==========  
  
提供训练曲线、3D轨迹、状态分析等可视化功能  
"""  
  
import os  
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.figure import Figure  
from mpl_toolkits.mplot3d import Axes3D  
from typing import Dict, List, Optional, Tuple, Union  
from collections import defaultdict  
import json  
  
  
# 设置matplotlib样式  
try:  
    plt.style.use('seaborn-v0_8-whitegrid')  
except:  
    try:  
        plt.style.use('seaborn-whitegrid')  
    except:  
        pass  # 使用默认样式  
  
plt.rcParams['figure.figsize'] = (10, 6)  
plt.rcParams['font.size'] = 12  
plt.rcParams['axes.labelsize'] = 12  
plt.rcParams['axes.titlesize'] = 14  
plt.rcParams['legend.fontsize'] = 10  
  
  
class TrainingVisualizer:  
    """  
    训练过程可视化器  
    """  
      
    def __init__(self, save_dir: str = "visualization/plots"):  
        """  
        初始化  
          
        Args:  
            save_dir: 图像保存目录  
        """  
        self.save_dir = save_dir  
        os.makedirs(save_dir, exist_ok=True)  
          
        # 数据存储  
        self.data = defaultdict(list)  
          
    def add_scalar(self, tag: str, value: float, step: int):  
        """添加标量数据"""  
        self.data[tag].append((step, value))  
          
    def add_scalars(self, data_dict: Dict[str, float], step: int):  
        """添加多个标量"""  
        for tag, value in data_dict.items():  
            self.add_scalar(tag, value, step)  
              
    def plot_training_curves(self,   
                             metrics: List[str] = None,  
                             window_size: int = 10,  
                             save_name: str = "training_curves.png") -> Figure:  
        """  
        绘制训练曲线  
          
        Args:  
            metrics: 要绘制的指标列表（默认全部）  
            window_size: 平滑窗口大小  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        if metrics is None:  
            metrics = list(self.data.keys())  
              
        n_metrics = len(metrics)  
        if n_metrics == 0:  
            print("No metrics to plot!")  
            return None  
              
        # 确定子图布局  
        n_cols = min(2, n_metrics)  
        n_rows = (n_metrics + n_cols - 1) // n_cols  
          
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))  
          
        # 处理axes的不同情况  
        if n_metrics == 1:  
            axes = [axes]  
        elif n_rows == 1:  
            axes = list(axes)  
        else:  
            axes = axes.flatten()  
              
        for idx, metric in enumerate(metrics):  
            ax = axes[idx]  
            data = self.data.get(metric, [])  
              
            if not data:  
                ax.set_title(f"{metric} (no data)")  
                continue  
                  
            steps, values = zip(*data)  
            steps = np.array(steps)  
            values = np.array(values)  
              
            # 原始数据（透明）  
            ax.plot(steps, values, alpha=0.3, color='blue')  
              
            # 平滑数据 - 动态调整窗口大小  
            actual_window = min(window_size, max(1, len(values) // 2))  
            if actual_window >= 2 and len(values) >= actual_window:  
                smoothed = self._moving_average(values, actual_window)  
                # 正确对齐x轴  
                smooth_steps = steps[actual_window-1:]  
                if len(smooth_steps) == len(smoothed):  
                    ax.plot(smooth_steps, smoothed, color='blue', linewidth=2,  
                           label=f'Smoothed (window={actual_window})')  
                    ax.legend()  
            else:  
                ax.plot(steps, values, color='blue', linewidth=2)  
                  
            ax.set_xlabel('Step')  
            ax.set_ylabel(metric)  
            ax.set_title(metric)  
            ax.grid(True)  
              
        # 隐藏多余的子图  
        for idx in range(n_metrics, len(axes)):  
            axes[idx].set_visible(False)  
              
        plt.tight_layout()  
          
        # 保存  
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
        print(f"Training curves saved to {save_path}")  
          
        return fig  
      
    def plot_reward_curve(self,   
                          reward_key: str = "episode/total_reward",  
                          window_size: int = 50,  
                          save_name: str = "reward_curve.png") -> Figure:  
        """  
        专门绘制奖励曲线  
          
        Args:  
            reward_key: 奖励数据的键名  
            window_size: 平滑窗口大小  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        data = self.data.get(reward_key, [])  
        if not data:  
            print(f"No data for {reward_key}")  
            return None  
              
        steps, rewards = zip(*data)  
        steps = np.array(steps)  
        rewards = np.array(rewards)  
          
        fig, ax = plt.subplots(figsize=(12, 6))  
          
        # 原始奖励（散点）  
        ax.scatter(steps, rewards, alpha=0.3, s=5, color='blue', label='Episode Reward')  
          
        # 平滑曲线 - 动态调整窗口大小  
        actual_window = min(window_size, max(1, len(rewards) // 2))  
        if actual_window >= 2 and len(rewards) >= actual_window:  
            smoothed = self._moving_average(rewards, actual_window)  
            smooth_steps = steps[actual_window-1:]  
            if len(smooth_steps) == len(smoothed):  
                ax.plot(smooth_steps, smoothed, color='red', linewidth=2,  
                       label=f'Moving Average (window={actual_window})')  
              
        # 标注最大值  
        if len(rewards) > 0:  
            max_idx = np.argmax(rewards)  
            ax.scatter(steps[max_idx], rewards[max_idx], color='green', s=100,   
                       zorder=5, label=f'Best: {rewards[max_idx]:.2f}')  
          
        ax.set_xlabel('Episode')  
        ax.set_ylabel('Total Reward')  
        ax.set_title('Training Reward Curve')  
        ax.legend()  
        ax.grid(True)  
          
        # 添加趋势信息  
        if len(rewards) > 100:  
            recent_mean = np.mean(rewards[-100:])  
            ax.axhline(y=recent_mean, color='orange', linestyle='--',   
                       label=f'Recent Mean: {recent_mean:.2f}')  
              
        plt.tight_layout()  
          
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
        print(f"Reward curve saved to {save_path}")  
          
        return fig  
      
    def plot_loss_curves(self,  
                         actor_loss_key: str = "train/actor_loss",  
                         critic_loss_key: str = "train/critic_loss",  
                         window_size: int = 100,  
                         save_name: str = "loss_curves.png") -> Figure:  
        """  
        绘制损失曲线  
          
        Args:  
            actor_loss_key: Actor损失数据的键名  
            critic_loss_key: Critic损失数据的键名  
            window_size: 平滑窗口大小  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))  
          
        for ax, key, title, color in [  
            (axes[0], actor_loss_key, "Actor Loss", "blue"),  
            (axes[1], critic_loss_key, "Critic Loss", "red")  
        ]:  
            data = self.data.get(key, [])  
            if not data:  
                ax.set_title(f"{title} (no data)")  
                continue  
                  
            steps, values = zip(*data)  
            steps = np.array(steps)  
            values = np.array(values)  
              
            ax.plot(steps, values, alpha=0.3, color=color)  
              
            # 动态调整窗口大小  
            actual_window = min(window_size, max(1, len(values) // 2))  
            if actual_window >= 2 and len(values) >= actual_window:  
                smoothed = self._moving_average(values, actual_window)  
                smooth_steps = steps[actual_window-1:]  
                if len(smooth_steps) == len(smoothed):  
                    ax.plot(smooth_steps, smoothed, color=color, linewidth=2)  
                  
            ax.set_xlabel('Step')  
            ax.set_ylabel('Loss')  
            ax.set_title(title)  
            ax.grid(True)  
              
            # 使用对数刻度（如果值范围大）  
            if len(values) > 0:  
                val_min = min(values)  
                val_max = max(values)  
                if val_min > 0 and val_max / (val_min + 1e-8) > 100:  
                    ax.set_yscale('log')  
                  
        plt.tight_layout()  
          
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
        print(f"Loss curves saved to {save_path}")  
          
        return fig  
      
    def plot_action_distribution(self,   
                                 action_counts: Dict[int, int],  
                                 action_names: List[str] = None,  
                                 save_name: str = "action_distribution.png") -> Figure:  
        """  
        绘制动作分布  
          
        Args:  
            action_counts: 动作计数字典  
            action_names: 动作名称列表  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        actions = list(action_counts.keys())  
        counts = list(action_counts.values())  
          
        fig, ax = plt.subplots(figsize=(12, 6))  
          
        if action_names is None:  
            labels = [f"Action {a}" for a in actions]  
        else:  
            labels = [action_names[a] if a < len(action_names) else f"Action {a}"   
                     for a in actions]  
              
        bars = ax.bar(range(len(actions)), counts, color='steelblue')  
        ax.set_xticks(range(len(actions)))  
        ax.set_xticklabels(labels, rotation=45, ha='right')  
        ax.set_xlabel('Action')  
        ax.set_ylabel('Count')  
        ax.set_title('Action Distribution')  
          
        # 添加数值标签  
        for bar, count in zip(bars, counts):  
            height = bar.get_height()  
            ax.annotate(f'{count}',  
                       xy=(bar.get_x() + bar.get_width() / 2, height),  
                       xytext=(0, 3),  
                       textcoords="offset points",  
                       ha='center', va='bottom')  
              
        plt.tight_layout()  
          
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
          
        return fig  
      
    @staticmethod  
    def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:  
        """计算移动平均"""  
        if window_size < 1:  
            return data  
        weights = np.ones(window_size) / window_size  
        return np.convolve(data, weights, mode='valid')  
      
    def save_data(self, filename: str = "training_data.json"):  
        """保存所有数据到JSON"""  
        save_path = os.path.join(self.save_dir, filename)  
          
        # 转换数据格式  
        data_to_save = {}  
        for key, values in self.data.items():  
            if values:  
                steps, vals = zip(*values)  
                data_to_save[key] = {  
                    'steps': list(steps),  
                    'values': list(vals)  
                }  
            else:  
                data_to_save[key] = {'steps': [], 'values': []}  
              
        with open(save_path, 'w') as f:  
            json.dump(data_to_save, f)  
              
        print(f"Training data saved to {save_path}")  
          
    def load_data(self, filename: str = "training_data.json"):  
        """从JSON加载数据"""  
        load_path = os.path.join(self.save_dir, filename)  
          
        with open(load_path, 'r') as f:  
            loaded_data = json.load(f)  
              
        self.data.clear()  
        for key, values in loaded_data.items():  
            steps = values['steps']  
            vals = values['values']  
            self.data[key] = list(zip(steps, vals))  
              
        print(f"Training data loaded from {load_path}")  
  
  
class TrajectoryVisualizer:  
    """  
    轨迹可视化器  
      
    用于可视化AUV的运动轨迹  
    """  
      
    def __init__(self, save_dir: str = "visualization/plots"):  
        """  
        初始化  
          
        Args:  
            save_dir: 图像保存目录  
        """  
        self.save_dir = save_dir  
        os.makedirs(save_dir, exist_ok=True)  
          
    def plot_2d_trajectory(self,  
                           positions: np.ndarray,  
                           target: Optional[np.ndarray] = None,  
                           obstacles: Optional[List[np.ndarray]] = None,  
                           save_name: str = "trajectory_2d.png") -> Figure:  
        """  
        绘制2D轨迹（俯视图）  
          
        Args:  
            positions: 位置数组 [N, 3] 或 [N, 2]  
            target: 目标位置  
            obstacles: 障碍物列表  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        fig, ax = plt.subplots(figsize=(10, 10))  
          
        # 绘制轨迹  
        if positions.shape[1] >= 2:  
            ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')  
            ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=15, label='Start')  
            ax.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=15, label='End')  
              
            # 添加方向箭头  
            n_arrows = min(10, len(positions) // 10)  
            if n_arrows > 0 and len(positions) > 1:  
                indices = np.linspace(0, len(positions)-2, n_arrows, dtype=int)  
                for idx in indices:  
                    dx = positions[idx+1, 0] - positions[idx, 0]  
                    dy = positions[idx+1, 1] - positions[idx, 1]  
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:  
                        ax.arrow(positions[idx, 0], positions[idx, 1], dx*0.5, dy*0.5,  
                                head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.5)  
                  
        # 绘制目标  
        if target is not None:  
            ax.plot(target[0], target[1], 'r*', markersize=20, label='Target')  
              
        # 绘制障碍物  
        if obstacles is not None:  
            for obs in obstacles:  
                circle = plt.Circle((obs[0], obs[1]), obs[3] if len(obs) > 3 else 1.0,  
                                   color='gray', alpha=0.5)  
                ax.add_patch(circle)  
                  
        ax.set_xlabel('X (m)')  
        ax.set_ylabel('Y (m)')  
        ax.set_title('2D Trajectory (Top View)')  
        ax.legend()  
        ax.axis('equal')  
        ax.grid(True)  
          
        plt.tight_layout()  
          
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
          
        return fig  
      
    def plot_3d_trajectory(self,  
                           positions: np.ndarray,  
                           orientations: Optional[np.ndarray] = None,  
                           target: Optional[np.ndarray] = None,  
                           save_name: str = "trajectory_3d.png") -> Figure:  
        """  
        绘制3D轨迹  
          
        Args:  
            positions: 位置数组 [N, 3]  
            orientations: 姿态数组 [N, 3]（可选，用于显示朝向）  
            target: 目标位置  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        fig = plt.figure(figsize=(12, 10))  
        ax = fig.add_subplot(111, projection='3d')  
          
        # 绘制轨迹  
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],  
               'b-', linewidth=2, label='Trajectory')  
          
        # 起点和终点  
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],  
                  c='green', s=200, marker='o', label='Start')  
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],  
                  c='red', s=200, marker='s', label='End')  
          
        # 目标点  
        if target is not None:  
            ax.scatter(target[0], target[1], target[2],  
                      c='gold', s=300, marker='*', label='Target')  
              
        # 绘制朝向箭头  
        if orientations is not None and len(positions) > 1:  
            n_arrows = min(15, len(positions) // 5)  
            if n_arrows > 0:  
                indices = np.linspace(0, len(positions)-1, n_arrows, dtype=int)  
                  
                for idx in indices:  
                    pos = positions[idx]  
                    yaw = orientations[idx, 2] if orientations.shape[1] > 2 else 0  
                      
                    # 计算朝向向量  
                    dx = np.cos(yaw) * 0.5  
                    dy = np.sin(yaw) * 0.5  
                      
                    ax.quiver(pos[0], pos[1], pos[2], dx, dy, 0,  
                             color='orange', arrow_length_ratio=0.3, alpha=0.7)  
                  
        ax.set_xlabel('X (m)')  
        ax.set_ylabel('Y (m)')  
        ax.set_zlabel('Z (m)')  
        ax.set_title('3D Trajectory')  
        ax.legend()  
          
        # 设置相等的轴比例  
        if len(positions) > 0:  
            max_range = np.max([  
                positions[:, 0].max() - positions[:, 0].min(),  
                positions[:, 1].max() - positions[:, 1].min(),  
                positions[:, 2].max() - positions[:, 2].min()  
            ])  
            if max_range > 0:  
                mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2  
                mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2  
                mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2  
                ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)  
                ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)  
                ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)  
          
        plt.tight_layout()  
          
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
          
        return fig  
      
    def plot_state_evolution(self,  
                             time: np.ndarray,  
                             positions: np.ndarray,  
                             orientations: np.ndarray,  
                             velocities: Optional[np.ndarray] = None,  
                             save_name: str = "state_evolution.png") -> Figure:  
        """  
        绘制状态随时间的变化  
          
        Args:  
            time: 时间数组  
            positions: 位置数组 [N, 3]  
            orientations: 姿态数组 [N, 3]  
            velocities: 速度数组 [N, 6]（可选）  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        n_subplots = 2 if velocities is None else 3  
        fig, axes = plt.subplots(n_subplots, 2, figsize=(14, 4*n_subplots))  
          
        # 位置  
        axes[0, 0].plot(time, positions[:, 0], label='x')  
        axes[0, 0].plot(time, positions[:, 1], label='y')  
        axes[0, 0].plot(time, positions[:, 2], label='z')  
        axes[0, 0].set_xlabel('Time (s)')  
        axes[0, 0].set_ylabel('Position (m)')  
        axes[0, 0].set_title('Position vs Time')  
        axes[0, 0].legend()  
        axes[0, 0].grid(True)  
          
        # 姿态  
        axes[0, 1].plot(time, np.degrees(orientations[:, 0]), label='roll')  
        axes[0, 1].plot(time, np.degrees(orientations[:, 1]), label='pitch')  
        axes[0, 1].plot(time, np.degrees(orientations[:, 2]), label='yaw')  
        axes[0, 1].set_xlabel('Time (s)')  
        axes[0, 1].set_ylabel('Angle (deg)')  
        axes[0, 1].set_title('Orientation vs Time')  
        axes[0, 1].legend()  
        axes[0, 1].grid(True)  
          
        # 位置距离  
        axes[1, 0].plot(time, np.linalg.norm(positions, axis=1))  
        axes[1, 0].set_xlabel('Time (s)')  
        axes[1, 0].set_ylabel('Distance from Origin (m)')  
        axes[1, 0].set_title('Distance from Origin')  
        axes[1, 0].grid(True)  
          
        # 姿态角范数  
        axes[1, 1].plot(time, np.linalg.norm(orientations, axis=1))  
        axes[1, 1].set_xlabel('Time (s)')  
        axes[1, 1].set_ylabel('Orientation Norm (rad)')  
        axes[1, 1].set_title('Total Rotation')  
        axes[1, 1].grid(True)  
          
        # 速度  
        if velocities is not None:  
            # 线速度  
            axes[2, 0].plot(time, velocities[:, 0], label='u')  
            axes[2, 0].plot(time, velocities[:, 1], label='v')  
            axes[2, 0].plot(time, velocities[:, 2], label='w')  
            axes[2, 0].set_xlabel('Time (s)')  
            axes[2, 0].set_ylabel('Linear Velocity (m/s)')  
            axes[2, 0].set_title('Linear Velocity vs Time')  
            axes[2, 0].legend()  
            axes[2, 0].grid(True)  
              
            # 角速度  
            axes[2, 1].plot(time, np.degrees(velocities[:, 3]), label='p')  
            axes[2, 1].plot(time, np.degrees(velocities[:, 4]), label='q')  
            axes[2, 1].plot(time, np.degrees(velocities[:, 5]), label='r')  
            axes[2, 1].set_xlabel('Time (s)')  
            axes[2, 1].set_ylabel('Angular Velocity (deg/s)')  
            axes[2, 1].set_title('Angular Velocity vs Time')  
            axes[2, 1].legend()  
            axes[2, 1].grid(True)  
              
        plt.tight_layout()  
          
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
          
        return fig  
  
  
class MultiAgentVisualizer:  
    """  
    多智能体可视化器  
      
    用于可视化追逃博弈中多个智能体的轨迹  
    """  
      
    def __init__(self, save_dir: str = "visualization/plots"):  
        """  
        初始化  
          
        Args:  
            save_dir: 图像保存目录  
        """  
        self.save_dir = save_dir  
        os.makedirs(save_dir, exist_ok=True)  
          
        # 颜色方案  
        self.pursuer_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))  
        self.evader_color = 'red'  
          
    def plot_pursuit_scenario(self,  
                              pursuer_trajectories: List[np.ndarray],  
                              evader_trajectory: np.ndarray,  
                              terrain_heightmap: Optional[np.ndarray] = None,  
                              save_name: str = "pursuit_scenario.png") -> Figure:  
        """  
        绘制追逃场景  
          
        Args:  
            pursuer_trajectories: 追击者轨迹列表，每个元素为[N, 3]  
            evader_trajectory: 逃逸者轨迹 [N, 3]  
            terrain_heightmap: 地形高度图  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        fig = plt.figure(figsize=(14, 10))  
          
        # 2D俯视图  
        ax1 = fig.add_subplot(121)  
          
        # 绘制追击者  
        for i, traj in enumerate(pursuer_trajectories):  
            color = self.pursuer_colors[i % len(self.pursuer_colors)]  
            ax1.plot(traj[:, 0], traj[:, 1], '-', color=color,   
                    linewidth=2, label=f'Pursuer {i+1}')  
            ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=10)  
            ax1.plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=10)  
              
        # 绘制逃逸者  
        ax1.plot(evader_trajectory[:, 0], evader_trajectory[:, 1], '-',  
                color=self.evader_color, linewidth=2, label='Evader')  
        ax1.plot(evader_trajectory[0, 0], evader_trajectory[0, 1], 'o',  
                color=self.evader_color, markersize=10)  
        ax1.plot(evader_trajectory[-1, 0], evader_trajectory[-1, 1], 's',  
                color=self.evader_color, markersize=10)  
          
        ax1.set_xlabel('X (m)')  
        ax1.set_ylabel('Y (m)')  
        ax1.set_title('Pursuit-Evasion Scenario (Top View)')  
        ax1.legend()  
        ax1.axis('equal')  
        ax1.grid(True)  
          
        # 3D视图  
        ax2 = fig.add_subplot(122, projection='3d')  
          
        # 绘制地形  
        if terrain_heightmap is not None:  
            x = np.arange(terrain_heightmap.shape[0])  
            y = np.arange(terrain_heightmap.shape[1])  
            X, Y = np.meshgrid(x, y)  
            ax2.plot_surface(X, Y, terrain_heightmap.T, cmap='terrain',  
                           alpha=0.5, rstride=5, cstride=5)  
              
        # 绘制追击者  
        for i, traj in enumerate(pursuer_trajectories):  
            color = self.pursuer_colors[i % len(self.pursuer_colors)]  
            ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-',  
                    color=color, linewidth=2, label=f'Pursuer {i+1}')  
              
        # 绘制逃逸者  
        ax2.plot(evader_trajectory[:, 0], evader_trajectory[:, 1],  
                evader_trajectory[:, 2], '-', color=self.evader_color,  
                linewidth=2, label='Evader')  
          
        ax2.set_xlabel('X (m)')  
        ax2.set_ylabel('Y (m)')  
        ax2.set_zlabel('Z (m)')  
        ax2.set_title('3D Pursuit-Evasion Scenario')  
        ax2.legend()  
          
        plt.tight_layout()  
          
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
          
        return fig  
      
    def plot_distance_over_time(self,  
                                pursuer_trajectories: List[np.ndarray],  
                                evader_trajectory: np.ndarray,  
                                dt: float = 0.02,  
                                capture_radius: float = 1.0,  
                                save_name: str = "distance_plot.png") -> Figure:  
        """  
        绘制距离随时间变化  
          
        Args:  
            pursuer_trajectories: 追击者轨迹列表  
            evader_trajectory: 逃逸者轨迹  
            dt: 时间步长  
            capture_radius: 捕获半径  
            save_name: 保存文件名  
              
        Returns:  
            matplotlib Figure对象  
        """  
        fig, ax = plt.subplots(figsize=(12, 6))  
          
        n_steps = len(evader_trajectory)  
        time = np.arange(n_steps) * dt  
          
        # 计算每个追击者与逃逸者的距离  
        for i, pursuer_traj in enumerate(pursuer_trajectories):  
            min_len = min(len(pursuer_traj), len(evader_trajectory))  
            distances = np.linalg.norm(  
                pursuer_traj[:min_len] - evader_trajectory[:min_len],   
                axis=1  
            )  
              
            color = self.pursuer_colors[i % len(self.pursuer_colors)]  
            ax.plot(time[:min_len], distances, '-', color=color,   
                   linewidth=2, label=f'Pursuer {i+1}')  
              
        # 捕获半径线  
        ax.axhline(y=capture_radius, color='red', linestyle='--',  
                  linewidth=2, label=f'Capture Radius ({capture_radius}m)')  
          
        ax.set_xlabel('Time (s)')  
        ax.set_ylabel('Distance to Evader (m)')  
        ax.set_title('Pursuer-Evader Distance Over Time')  
        ax.legend()  
        ax.grid(True)  
          
        # 找到最近距离  
        min_distances = []  
        for pursuer_traj in pursuer_trajectories:  
            min_len = min(len(pursuer_traj), len(evader_trajectory))  
            distances = np.linalg.norm(  
                pursuer_traj[:min_len] - evader_trajectory[:min_len],   
                axis=1  
            )  
            min_distances.append(np.min(distances))  
              
        min_dist = min(min_distances) if min_distances else 0  
        ax.annotate(f'Min Distance: {min_dist:.2f}m',  
                   xy=(0.02, 0.98), xycoords='axes fraction',  
                   ha='left', va='top', fontsize=12,  
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))  
          
        plt.tight_layout()  
          
        save_path = os.path.join(self.save_dir, save_name)  
        plt.savefig(save_path, dpi=150, bbox_inches='tight')  
          
        return fig  
  
  
def create_summary_figure(training_stats: Dict,  
                          save_path: str = "visualization/plots/summary.png") -> Figure:  
    """  
    创建训练总结图  
      
    Args:  
        training_stats: 训练统计信息字典  
        save_path: 保存路径  
          
    Returns:  
        matplotlib Figure对象  
    """  
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
      
    # 1. 奖励分布  
    if 'rewards' in training_stats and len(training_stats['rewards']) > 0:  
        axes[0, 0].hist(training_stats['rewards'], bins=50, color='steelblue', edgecolor='black')  
        axes[0, 0].set_xlabel('Episode Reward')  
        axes[0, 0].set_ylabel('Frequency')  
        axes[0, 0].set_title('Reward Distribution')  
    else:  
        axes[0, 0].set_title('Reward Distribution (no data)')  
          
    # 2. 成功率趋势  
    if 'success_rates' in training_stats and len(training_stats['success_rates']) > 0:  
        axes[0, 1].plot(training_stats['success_rates'], color='green', linewidth=2)  
        axes[0, 1].set_xlabel('Evaluation')  
        axes[0, 1].set_ylabel('Success Rate')  
        axes[0, 1].set_title('Success Rate Over Training')  
        axes[0, 1].set_ylim([0, 1])  
    else:  
        axes[0, 1].set_title('Success Rate (no data)')  
          
    # 3. Episode长度  
    if 'episode_lengths' in training_stats and len(training_stats['episode_lengths']) > 0:  
        axes[0, 2].plot(training_stats['episode_lengths'], color='purple', alpha=0.5)  
        axes[0, 2].set_xlabel('Episode')  
        axes[0, 2].set_ylabel('Length')  
        axes[0, 2].set_title('Episode Length')  
    else:  
        axes[0, 2].set_title('Episode Length (no data)')  
          
    # 4. Actor Loss  
    if 'actor_losses' in training_stats and len(training_stats['actor_losses']) > 0:  
        axes[1, 0].plot(training_stats['actor_losses'], color='blue', linewidth=1)  
        axes[1, 0].set_xlabel('Update')  
        axes[1, 0].set_ylabel('Loss')  
        axes[1, 0].set_title('Actor Loss')  
    else:  
        axes[1, 0].set_title('Actor Loss (no data)')  
          
    # 5. Critic Loss  
    if 'critic_losses' in training_stats and len(training_stats['critic_losses']) > 0:  
        axes[1, 1].plot(training_stats['critic_losses'], color='red', linewidth=1)  
        axes[1, 1].set_xlabel('Update')  
        axes[1, 1].set_ylabel('Loss')  
        axes[1, 1].set_title('Critic Loss')  
    else:  
        axes[1, 1].set_title('Critic Loss (no data)')  
          
    # 6. 熵  
    if 'entropies' in training_stats and len(training_stats['entropies']) > 0:  
        axes[1, 2].plot(training_stats['entropies'], color='orange', linewidth=1)  
        axes[1, 2].set_xlabel('Update')  
        axes[1, 2].set_ylabel('Entropy')  
        axes[1, 2].set_title('Policy Entropy')  
    else:  
        axes[1, 2].set_title('Policy Entropy (no data)')  
          
    plt.tight_layout()  
      
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  
    print(f"Summary figure saved to {save_path}")  
      
    return fig  
