"""
高质量追逃GIF可视化器
====================
- 追踪双方速度、加速度
- 实时距离显示
- 轨迹渐变色
- 子目标可视化
- 障碍物渲染

依赖: pip install matplotlib pillow numpy
"""

import os
import sys
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ==================== 数据结构 ====================

@dataclass
class AgentState:
    """智能体状态"""
    pos: np.ndarray      # [x, y]
    vel: np.ndarray      # [vx, vy]
    yaw: float           # 朝向角 (rad)
    speed: float         # 速度标量
    acc: float           # 加速度标量


@dataclass
class TrajectoryData:
    """轨迹数据"""
    pursuer_states: List[AgentState]
    evader_states: List[AgentState]
    subgoals: List[np.ndarray]      # 子目标位置
    distances: List[float]
    success: bool
    dt: float = 0.02
    
    @property
    def n_steps(self) -> int:
        return len(self.pursuer_states)
    
    def get_pursuer_speeds(self) -> np.ndarray:
        return np.array([s.speed for s in self.pursuer_states])
    
    def get_evader_speeds(self) -> np.ndarray:
        return np.array([s.speed for s in self.evader_states])
    
    def get_pursuer_accs(self) -> np.ndarray:
        return np.array([s.acc for s in self.pursuer_states])
    
    def get_evader_accs(self) -> np.ndarray:
        return np.array([s.acc for s in self.evader_states])
    
    def get_pursuer_positions(self) -> np.ndarray:
        return np.array([s.pos for s in self.pursuer_states])
    
    def get_evader_positions(self) -> np.ndarray:
        return np.array([s.pos for s in self.evader_states])


# ==================== 配色方案 ====================

class ColorScheme:
    """可视化配色"""
    PURSUER = '#2E86AB'       # 追击者 - 深蓝
    EVADER = '#C73E1D'        # 逃脱者 - 红色
    SUBGOAL = '#F18F01'       # 子目标 - 橙色
    OBSTACLE = '#555555'      # 障碍物 - 灰色
    BOUNDARY = '#1a1a1a'      # 边界 - 黑色
    CATCH_RADIUS = '#3A7D44'  # 捕获半径 - 绿色
    GRID = '#e0e0e0'          # 网格
    TEXT_BG = '#ffffffee'     # 文字背景
    
    # 速度/加速度曲线
    V_PURSUER = '#2E86AB'
    V_EVADER = '#C73E1D'
    A_PURSUER = '#7B2CBF'     # 紫色
    A_EVADER = '#E36414'      # 橙红


# ==================== 轨迹渐变色 ====================

def create_trajectory_cmap(base_color: str, n_points: int) -> np.ndarray:
    """创建轨迹渐变颜色 (旧->新: 透明->不透明)"""
    import matplotlib.colors as mcolors
    rgba = mcolors.to_rgba(base_color)
    colors = np.zeros((n_points, 4))
    for i in range(n_points):
        alpha = 0.1 + 0.9 * (i / max(1, n_points - 1))
        colors[i] = (*rgba[:3], alpha)
    return colors


def draw_gradient_trajectory(ax, positions: np.ndarray, color: str, 
                            linewidth: float = 2.0, max_points: int = 200):
    """绘制渐变色轨迹"""
    n = len(positions)
    if n < 2:
        return
    
    # 限制点数
    if n > max_points:
        indices = np.linspace(0, n-1, max_points, dtype=int)
        positions = positions[indices]
        n = len(positions)
    
    # 创建线段集合
    points = positions.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 颜色渐变
    colors = create_trajectory_cmap(color, n - 1)
    
    lc = LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)


# ==================== 障碍物绘制 ====================

@dataclass
class RectObstacle:
    """矩形障碍物"""
    cx: float
    cy: float
    w: float
    h: float
    
    def draw(self, ax, color=ColorScheme.OBSTACLE, alpha=0.7):
        rect = Rectangle(
            (self.cx - self.w/2, self.cy - self.h/2),
            self.w, self.h,
            facecolor=color, edgecolor='#333333',
            alpha=alpha, linewidth=1, zorder=2
        )
        ax.add_patch(rect)


def get_default_obstacles() -> List[RectObstacle]:
    """获取默认障碍物配置"""
    return [
        RectObstacle(0.0, 2.0, 8.0, 2.0),
        RectObstacle(0.0, -2.0, 2.5, 6.0),
        RectObstacle(-9.0, 9.0, 4.0, 2.0),
        RectObstacle(-10.0, 7.0, 2.0, 4.0),
        RectObstacle(9.0, 9.0, 4.0, 2.0),
        RectObstacle(10.0, 7.0, 2.0, 4.0),
        RectObstacle(-9.0, -9.0, 4.0, 2.0),
        RectObstacle(-10.0, -7.0, 2.0, 4.0),
        RectObstacle(9.0, -9.0, 4.0, 2.0),
        RectObstacle(10.0, -7.0, 2.0, 4.0),
        RectObstacle(-5.5, -5.0, 2.5, 2.5),
        RectObstacle(5.5, 5.0, 2.5, 2.5),
    ]


# ==================== 智能体符号绘制 ====================

def draw_agent_marker(ax, pos: np.ndarray, yaw: float, color: str, 
                      size: float = 0.8, label: str = None, zorder: int = 10):
    """绘制智能体标记 (带朝向的三角形)"""
    # 三角形顶点
    L = size
    pts = np.array([
        [L * 0.7, 0],           # 前端
        [-L * 0.4, L * 0.4],    # 左后
        [-L * 0.4, -L * 0.4],   # 右后
    ])
    
    # 旋转
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T + pos
    
    triangle = Polygon(pts, facecolor=color, edgecolor='white', 
                      linewidth=1.5, zorder=zorder, label=label)
    ax.add_patch(triangle)
    
    return triangle


def draw_velocity_arrow(ax, pos: np.ndarray, vel: np.ndarray, color: str,
                       scale: float = 1.5, zorder: int = 9):
    """绘制速度箭头"""
    if np.linalg.norm(vel) < 0.01:
        return None
    
    arrow = FancyArrowPatch(
        pos, pos + vel * scale,
        arrowstyle='->', mutation_scale=12,
        color=color, linewidth=1.5, alpha=0.7, zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow


# ==================== 信息面板 ====================

def draw_info_panel(ax, frame: int, total_frames: int, 
                    p_state: AgentState, e_state: AgentState,
                    distance: float, catch_radius: float,
                    v_max_p: float, v_max_e: float,
                    a_max_p: float, a_max_e: float):
    """绘制信息面板"""
    
    # 清除旧文本
    for txt in ax.texts[:]:
        txt.remove()
    
    time_s = frame * 0.02  # dt=0.02
    
    # 主要状态信息
    info_text = (
        f"Time: {time_s:.2f}s  |  Frame: {frame}/{total_frames}\n"
        f"Distance: {distance:.2f}m  |  Catch: {catch_radius:.1f}m\n"
        f"─── Pursuer (Blue) ───\n"
        f"  Speed: {p_state.speed:.3f} / {v_max_p:.2f} m/s\n"
        f"  Accel: {p_state.acc:.3f} m/s²\n"
        f"─── Evader (Red) ───\n"
        f"  Speed: {e_state.speed:.3f} / {v_max_e:.2f} m/s\n"
        f"  Accel: {e_state.acc:.3f} / {a_max_e:.2f} m/s²"
    )
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=ColorScheme.TEXT_BG, 
                     edgecolor='#cccccc', linewidth=0.5))


# ==================== 子图: v-t 和 a-t 曲线 ====================

def update_velocity_plot(ax, frame: int, traj: TrajectoryData,
                        v_max_p: float, v_max_e: float):
    """更新速度-时间子图"""
    ax.clear()
    
    # 获取到当前帧的数据
    t = np.arange(frame + 1) * traj.dt
    p_speeds = traj.get_pursuer_speeds()[:frame + 1]
    e_speeds = traj.get_evader_speeds()[:frame + 1]
    
    # 绘制曲线
    ax.plot(t, p_speeds, color=ColorScheme.V_PURSUER, linewidth=2, label='Pursuer')
    ax.plot(t, e_speeds, color=ColorScheme.V_EVADER, linewidth=2, label='Evader')
    
    # 最大速度参考线
    ax.axhline(y=v_max_p, color=ColorScheme.V_PURSUER, linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=v_max_e, color=ColorScheme.V_EVADER, linestyle='--', alpha=0.5, linewidth=1)
    
    # 当前点
    ax.scatter([t[-1]], [p_speeds[-1]], color=ColorScheme.V_PURSUER, s=50, zorder=5)
    ax.scatter([t[-1]], [e_speeds[-1]], color=ColorScheme.V_EVADER, s=50, zorder=5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Velocity', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, traj.n_steps * traj.dt)
    ax.set_ylim(0, max(v_max_p, v_max_e) * 1.2)
    ax.grid(True, alpha=0.3)


def update_acceleration_plot(ax, frame: int, traj: TrajectoryData,
                            a_max_p: float, a_max_e: float):
    """更新加速度-时间子图"""
    ax.clear()
    
    t = np.arange(frame + 1) * traj.dt
    p_accs = traj.get_pursuer_accs()[:frame + 1]
    e_accs = traj.get_evader_accs()[:frame + 1]
    
    ax.plot(t, p_accs, color=ColorScheme.A_PURSUER, linewidth=2, label='Pursuer')
    ax.plot(t, e_accs, color=ColorScheme.A_EVADER, linewidth=2, label='Evader')
    
    # 最大加速度参考线
    ax.axhline(y=a_max_p, color=ColorScheme.A_PURSUER, linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=a_max_e, color=ColorScheme.A_EVADER, linestyle='--', alpha=0.5, linewidth=1)
    
    ax.scatter([t[-1]], [p_accs[-1]], color=ColorScheme.A_PURSUER, s=50, zorder=5)
    ax.scatter([t[-1]], [e_accs[-1]], color=ColorScheme.A_EVADER, s=50, zorder=5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accel (m/s²)')
    ax.set_title('Acceleration', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, traj.n_steps * traj.dt)
    y_max = max(a_max_p, a_max_e, np.max(np.abs(p_accs)), np.max(np.abs(e_accs))) * 1.2
    ax.set_ylim(-0.5, y_max)
    ax.grid(True, alpha=0.3)


def update_distance_plot(ax, frame: int, traj: TrajectoryData, catch_radius: float):
    """更新距离-时间子图"""
    ax.clear()
    
    t = np.arange(frame + 1) * traj.dt
    distances = np.array(traj.distances[:frame + 1])
    
    ax.plot(t, distances, color='#3A7D44', linewidth=2)
    ax.axhline(y=catch_radius, color='#C73E1D', linestyle='--', linewidth=1.5, 
              label=f'Catch radius ({catch_radius}m)')
    
    ax.scatter([t[-1]], [distances[-1]], color='#3A7D44', s=50, zorder=5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Distance to Evader', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, traj.n_steps * traj.dt)
    ax.set_ylim(0, max(distances) * 1.1 + 1)
    ax.grid(True, alpha=0.3)


# ==================== GIF 生成类 ====================

class PursuitGIFGenerator:
    """高质量追逃GIF生成器"""
    
    def __init__(self, 
                 world_size: float = 15.0,
                 catch_radius: float = 1.0,
                 v_max_pursuer: float = 0.65,
                 v_max_evader: float = 0.72,
                 a_max_pursuer: float = 1.73,
                 a_max_evader: float = 0.15,
                 obstacles: List[RectObstacle] = None):
        
        self.world_size = world_size
        self.half = world_size / 2
        self.catch_radius = catch_radius
        self.v_max_p = v_max_pursuer
        self.v_max_e = v_max_evader
        self.a_max_p = a_max_pursuer
        self.a_max_e = a_max_evader
        self.obstacles = obstacles or get_default_obstacles()
    
    def generate_gif(self, traj: TrajectoryData, save_path: str,
                    fps: int = 20, skip_frames: int = 2,
                    figsize: Tuple[int, int] = (16, 9),
                    show_velocity_arrows: bool = True,
                    show_subgoals: bool = True,
                    title: str = "Pursuit-Evasion"):
        """
        生成高质量GIF
        
        Args:
            traj: 轨迹数据
            save_path: GIF保存路径
            fps: 帧率
            skip_frames: 跳帧数 (加速)
            figsize: 图像大小
            show_velocity_arrows: 显示速度箭头
            show_subgoals: 显示子目标
            title: 标题
        """
        print(f"Generating GIF: {traj.n_steps} frames, skip={skip_frames}, fps={fps}")
        
        # 采样帧索引
        frame_indices = list(range(0, traj.n_steps, skip_frames))
        if frame_indices[-1] != traj.n_steps - 1:
            frame_indices.append(traj.n_steps - 1)
        
        n_frames = len(frame_indices)
        print(f"  Rendering {n_frames} frames...")
        
        # 创建Figure布局: 主场景 + 3个子图
        fig = plt.figure(figsize=figsize)
        
        # GridSpec布局
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1],
                             hspace=0.25, wspace=0.3)
        
        # 主场景 (左侧, 占2行)
        ax_main = fig.add_subplot(gs[:, 0])
        
        # 右侧子图
        ax_vel = fig.add_subplot(gs[0, 1])     # 速度
        ax_acc = fig.add_subplot(gs[0, 2])     # 加速度
        ax_dist = fig.add_subplot(gs[1, 1:])   # 距离
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        # 预先获取位置数据
        p_positions = traj.get_pursuer_positions()
        e_positions = traj.get_evader_positions()
        
        # 动画更新函数
        def update(frame_idx):
            i = frame_indices[frame_idx]
            
            # === 主场景 ===
            ax_main.clear()
            
            # 边界
            boundary = Rectangle((-self.half, -self.half), self.world_size, self.world_size,
                                fill=False, edgecolor=ColorScheme.BOUNDARY, linewidth=2)
            ax_main.add_patch(boundary)
            
            # 障碍物
            for obs in self.obstacles:
                obs.draw(ax_main)
            
            # 历史轨迹 (渐变色)
            if i > 1:
                draw_gradient_trajectory(ax_main, p_positions[:i+1], ColorScheme.PURSUER, linewidth=2.5)
                draw_gradient_trajectory(ax_main, e_positions[:i+1], ColorScheme.EVADER, linewidth=2.5)
            
            # 当前状态
            p_state = traj.pursuer_states[i]
            e_state = traj.evader_states[i]
            
            # 子目标
            if show_subgoals and i < len(traj.subgoals) and traj.subgoals[i] is not None:
                sg = traj.subgoals[i]
                ax_main.scatter(sg[0], sg[1], s=100, c=ColorScheme.SUBGOAL, 
                              marker='*', zorder=8, edgecolors='white', linewidth=0.5)
                # 连接线
                ax_main.plot([p_state.pos[0], sg[0]], [p_state.pos[1], sg[1]],
                           '--', color=ColorScheme.SUBGOAL, alpha=0.5, linewidth=1)
            
            # 捕获半径圈
            catch_circle = Circle(e_state.pos, self.catch_radius, fill=False,
                                 edgecolor=ColorScheme.CATCH_RADIUS, linestyle='--',
                                 linewidth=1.5, alpha=0.7, zorder=3)
            ax_main.add_patch(catch_circle)
            
            # 绘制智能体
            draw_agent_marker(ax_main, p_state.pos, p_state.yaw, ColorScheme.PURSUER, 
                            size=0.8, label='Pursuer')
            draw_agent_marker(ax_main, e_state.pos, e_state.yaw, ColorScheme.EVADER, 
                            size=0.6, label='Evader')
            
            # 速度箭头
            if show_velocity_arrows:
                draw_velocity_arrow(ax_main, p_state.pos, p_state.vel, ColorScheme.PURSUER, scale=2.0)
                draw_velocity_arrow(ax_main, e_state.pos, e_state.vel, ColorScheme.EVADER, scale=2.0)
            
            # 信息面板
            draw_info_panel(ax_main, i, traj.n_steps, p_state, e_state,
                          traj.distances[i], self.catch_radius,
                          self.v_max_p, self.v_max_e, self.a_max_p, self.a_max_e)
            
            # 主场景设置
            ax_main.set_xlim(-self.half - 1, self.half + 1)
            ax_main.set_ylim(-self.half - 1, self.half + 1)
            ax_main.set_aspect('equal')
            ax_main.set_xlabel('X (m)')
            ax_main.set_ylabel('Y (m)')
            ax_main.grid(True, alpha=0.2)
            ax_main.legend(loc='lower left', fontsize=9)
            
            # === 子图更新 ===
            update_velocity_plot(ax_vel, i, traj, self.v_max_p, self.v_max_e)
            update_acceleration_plot(ax_acc, i, traj, self.a_max_p, self.a_max_e)
            update_distance_plot(ax_dist, i, traj, self.catch_radius)
            
            # 进度提示
            if frame_idx % 50 == 0:
                print(f"    Frame {frame_idx + 1}/{n_frames}")
            
            return []
        
        # 创建动画
        anim = animation.FuncAnimation(fig, update, frames=n_frames, 
                                       interval=1000/fps, blit=False)
        
        # 保存GIF
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        
        print(f"Saved: {save_path}")
        
        # 同时保存静态最终帧
        self._save_final_frame(traj, save_path.replace('.gif', '_final.png'),
                              title, show_velocity_arrows, show_subgoals)
    
    def _save_final_frame(self, traj: TrajectoryData, save_path: str,
                         title: str, show_velocity_arrows: bool, show_subgoals: bool):
        """保存最终帧静态图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        ax_main = axes[0, 0]
        ax_vel = axes[0, 1]
        ax_acc = axes[1, 0]
        ax_dist = axes[1, 1]
        
        # 主场景 - 完整轨迹
        ax_main.add_patch(Rectangle((-self.half, -self.half), self.world_size, self.world_size,
                                   fill=False, edgecolor=ColorScheme.BOUNDARY, linewidth=2))
        for obs in self.obstacles:
            obs.draw(ax_main)
        
        p_pos = traj.get_pursuer_positions()
        e_pos = traj.get_evader_positions()
        
        ax_main.plot(p_pos[:, 0], p_pos[:, 1], color=ColorScheme.PURSUER, 
                    linewidth=2, label='Pursuer', alpha=0.8)
        ax_main.plot(e_pos[:, 0], e_pos[:, 1], color=ColorScheme.EVADER, 
                    linewidth=2, label='Evader', alpha=0.8)
        
        # 起点和终点
        ax_main.scatter(p_pos[0, 0], p_pos[0, 1], s=100, c=ColorScheme.PURSUER, 
                       marker='o', edgecolors='white', zorder=10)
        ax_main.scatter(p_pos[-1, 0], p_pos[-1, 1], s=100, c=ColorScheme.PURSUER, 
                       marker='X', edgecolors='white', zorder=10)
        ax_main.scatter(e_pos[0, 0], e_pos[0, 1], s=100, c=ColorScheme.EVADER, 
                       marker='o', edgecolors='white', zorder=10)
        ax_main.scatter(e_pos[-1, 0], e_pos[-1, 1], s=100, c=ColorScheme.EVADER, 
                       marker='X', edgecolors='white', zorder=10)
        
        ax_main.set_xlim(-self.half - 1, self.half + 1)
        ax_main.set_ylim(-self.half - 1, self.half + 1)
        ax_main.set_aspect('equal')
        ax_main.set_xlabel('X (m)')
        ax_main.set_ylabel('Y (m)')
        ax_main.set_title('Complete Trajectory' + (' [CAUGHT]' if traj.success else ' [TIMEOUT]'))
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # 速度曲线 - 完整
        t = np.arange(traj.n_steps) * traj.dt
        ax_vel.plot(t, traj.get_pursuer_speeds(), color=ColorScheme.V_PURSUER, 
                   linewidth=2, label='Pursuer')
        ax_vel.plot(t, traj.get_evader_speeds(), color=ColorScheme.V_EVADER, 
                   linewidth=2, label='Evader')
        ax_vel.axhline(y=self.v_max_p, color=ColorScheme.V_PURSUER, linestyle='--', alpha=0.5)
        ax_vel.axhline(y=self.v_max_e, color=ColorScheme.V_EVADER, linestyle='--', alpha=0.5)
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Speed (m/s)')
        ax_vel.set_title('Velocity Profile')
        ax_vel.legend()
        ax_vel.grid(True, alpha=0.3)
        
        # 加速度曲线 - 完整
        ax_acc.plot(t, traj.get_pursuer_accs(), color=ColorScheme.A_PURSUER, 
                   linewidth=2, label='Pursuer')
        ax_acc.plot(t, traj.get_evader_accs(), color=ColorScheme.A_EVADER, 
                   linewidth=2, label='Evader')
        ax_acc.axhline(y=self.a_max_p, color=ColorScheme.A_PURSUER, linestyle='--', alpha=0.5)
        ax_acc.axhline(y=self.a_max_e, color=ColorScheme.A_EVADER, linestyle='--', alpha=0.5)
        ax_acc.set_xlabel('Time (s)')
        ax_acc.set_ylabel('Acceleration (m/s²)')
        ax_acc.set_title('Acceleration Profile')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        
        # 距离曲线
        ax_dist.plot(t, traj.distances, color='#3A7D44', linewidth=2)
        ax_dist.axhline(y=self.catch_radius, color='#C73E1D', linestyle='--', 
                       linewidth=1.5, label=f'Catch radius')
        ax_dist.set_xlabel('Time (s)')
        ax_dist.set_ylabel('Distance (m)')
        ax_dist.set_title('Distance to Evader')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        fig.suptitle(f'{title} - Summary', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved final frame: {save_path}")


# ==================== 从环境收集轨迹 ====================

def collect_trajectory_from_env(env, high_policy, device, 
                               deterministic: bool = True,
                               max_steps: int = 6000) -> TrajectoryData:
    """
    从环境中运行策略并收集轨迹数据
    
    Args:
        env: PursuitEnvRealAUVFixed 环境
        high_policy: 高层策略网络
        device: torch device
        deterministic: 是否确定性动作
        max_steps: 最大步数
    
    Returns:
        TrajectoryData 对象
    """
    import torch
    
    pursuer_states = []
    evader_states = []
    subgoals = []
    distances = []
    
    obs = env.reset()
    done = False
    step = 0
    
    prev_p_vel = np.zeros(2)
    prev_e_vel = np.zeros(2)
    dt = env.dt_evader * env.low_steps
    
    while not done and step < max_steps:
        # 获取当前状态
        eta = env.auv_env.inner.dynamics.eta
        nu = env.auv_env.inner.dynamics.nu
        
        p_pos = np.array([eta[0], eta[1]])
        p_vel = np.array([nu[0], nu[1]])
        p_yaw = float(eta[5])
        p_speed = np.linalg.norm(p_vel)
        p_acc = np.linalg.norm(p_vel - prev_p_vel) / dt if step > 0 else 0.0
        
        e_pos = env.e.copy()
        e_vel = env.ve.copy()
        e_yaw = env.psi_e if hasattr(env, 'psi_e') else np.arctan2(e_vel[1], e_vel[0])
        e_speed = np.linalg.norm(e_vel)
        e_acc = np.linalg.norm(e_vel - prev_e_vel) / dt if step > 0 else 0.0
        
        # 记录状态
        pursuer_states.append(AgentState(p_pos, p_vel, p_yaw, p_speed, p_acc))
        evader_states.append(AgentState(e_pos, e_vel, e_yaw, e_speed, e_acc))
        distances.append(float(np.linalg.norm(p_pos - e_pos)))
        
        # 高层动作
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if hasattr(high_policy, 'get_action'):
                result = high_policy.get_action(obs_t, deterministic=deterministic)
                action = result[0].cpu().numpy().flatten()  # 只取第一个返回值(action)
            else:
                # Actor-only
                action = high_policy(obs_t).cpu().numpy().flatten()
        
        # 计算子目标
        subgoal = p_pos + action * env.subgoal_range
        subgoals.append(subgoal.copy())
        
        # 环境步进
        obs, reward, done, info = env.step(action)
        
        prev_p_vel = p_vel.copy()
        prev_e_vel = e_vel.copy()
        step += 1
    
    # 最后一帧
    eta = env.auv_env.inner.dynamics.eta
    nu = env.auv_env.inner.dynamics.nu
    p_pos = np.array([eta[0], eta[1]])
    p_vel = np.array([nu[0], nu[1]])
    e_pos = env.e.copy()
    e_vel = env.ve.copy()
    
    pursuer_states.append(AgentState(p_pos, p_vel, float(eta[5]), 
                                     np.linalg.norm(p_vel), 0.0))
    evader_states.append(AgentState(e_pos, e_vel, 
                                    env.psi_e if hasattr(env, 'psi_e') else 0.0,
                                    np.linalg.norm(e_vel), 0.0))
    distances.append(float(np.linalg.norm(p_pos - e_pos)))
    subgoals.append(None)
    
    success = info.get('captured', False) or distances[-1] < env.catch_radius
    
    return TrajectoryData(
        pursuer_states=pursuer_states,
        evader_states=evader_states,
        subgoals=subgoals,
        distances=distances,
        success=success,
        dt=env.dt_evader * env.low_steps
    )


def _check_obstacle_collision(pos: np.ndarray, obstacles: List[RectObstacle], margin: float = 0.3) -> Tuple[bool, np.ndarray]:
    """检查位置是否与障碍物碰撞，返回 (是否碰撞, 推开后的位置)"""
    for obs in obstacles:
        # 检查是否在障碍物内部（含margin）
        if (abs(pos[0] - obs.cx) <= obs.w/2 + margin and 
            abs(pos[1] - obs.cy) <= obs.h/2 + margin):
            # 计算推开方向
            dx = pos[0] - obs.cx
            dy = pos[1] - obs.cy
            
            # 找最近的边
            dist_left = abs(pos[0] - (obs.cx - obs.w/2))
            dist_right = abs(pos[0] - (obs.cx + obs.w/2))
            dist_bottom = abs(pos[1] - (obs.cy - obs.h/2))
            dist_top = abs(pos[1] - (obs.cy + obs.h/2))
            
            min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            
            new_pos = pos.copy()
            if min_dist == dist_left:
                new_pos[0] = obs.cx - obs.w/2 - margin - 0.1
            elif min_dist == dist_right:
                new_pos[0] = obs.cx + obs.w/2 + margin + 0.1
            elif min_dist == dist_bottom:
                new_pos[1] = obs.cy - obs.h/2 - margin - 0.1
            else:
                new_pos[1] = obs.cy + obs.h/2 + margin + 0.1
            
            return True, new_pos
    return False, pos


def _get_obstacle_avoidance_force(pos: np.ndarray, obstacles: List[RectObstacle], 
                                   safe_dist: float = 1.5) -> np.ndarray:
    """计算障碍物斥力（用于路径规划）"""
    force = np.zeros(2)
    for obs in obstacles:
        # 到障碍物的最近距离
        dx = max(abs(pos[0] - obs.cx) - obs.w/2, 0)
        dy = max(abs(pos[1] - obs.cy) - obs.h/2, 0)
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < safe_dist:
            # 斥力方向：从障碍物中心指向智能体
            direction = pos - np.array([obs.cx, obs.cy])
            norm = np.linalg.norm(direction)
            if norm > 0.01:
                direction = direction / norm
                # 斥力大小与距离成反比
                strength = (safe_dist - dist) / safe_dist * 0.5
                force += direction * strength
    
    return force


def create_demo_trajectory(n_steps: int = 300, dt: float = 0.04) -> TrajectoryData:
    """创建演示轨迹 (用于测试) - 带障碍物避让"""
    pursuer_states = []
    evader_states = []
    subgoals = []
    distances = []
    
    # 获取障碍物
    obstacles = get_default_obstacles()
    world_half = 15.0
    
    # 初始位置 - 选择不在障碍物内的位置
    p_pos = np.array([-12.0, -10.0])
    e_pos = np.array([12.0, 10.0])
    p_vel = np.array([0.0, 0.0])
    e_vel = np.array([0.0, 0.0])
    
    for i in range(n_steps):
        t = i * dt
        
        # === 追击者运动 ===
        # 目标方向：朝向逃脱者
        direction = e_pos - p_pos
        dist = np.linalg.norm(direction)
        if dist > 0.1:
            target_dir = direction / dist
        else:
            target_dir = np.array([1.0, 0.0])
        
        # 障碍物斥力
        avoid_force = _get_obstacle_avoidance_force(p_pos, obstacles, safe_dist=2.0)
        
        # 组合：追击 + 避障
        combined_dir = target_dir + avoid_force * 2.0
        norm = np.linalg.norm(combined_dir)
        if norm > 0.01:
            combined_dir = combined_dir / norm
        
        # 目标速度
        target_vel = combined_dir * 0.55
        
        # 平滑加速 (追击者加速度大)
        p_acc_vec = np.clip((target_vel - p_vel) * 0.5, -1.73, 1.73)
        p_acc = np.linalg.norm(p_acc_vec)
        p_vel = p_vel + p_acc_vec * dt
        p_vel = np.clip(p_vel, -0.65, 0.65)
        
        # 更新位置
        p_pos_new = p_pos + p_vel * dt
        
        # 碰撞检测和修正
        collided, p_pos_new = _check_obstacle_collision(p_pos_new, obstacles)
        if collided:
            p_vel = p_vel * 0.1  # 碰撞后减速
        
        # 边界约束
        p_pos_new = np.clip(p_pos_new, -world_half + 0.5, world_half - 0.5)
        p_pos = p_pos_new
        p_yaw = np.arctan2(p_vel[1], p_vel[0]) if np.linalg.norm(p_vel) > 0.01 else 0.0
        
        # === 逃脱者运动 ===
        # 逃离方向
        escape_dir = p_pos - e_pos  # 注意：是 p-e，然后取反得到逃跑方向
        escape_dist = np.linalg.norm(escape_dir)
        if escape_dist > 0.1:
            escape_dir = escape_dir / escape_dist
        else:
            escape_dir = np.array([1.0, 0.0])
        
        # 添加随机扰动
        noise = np.array([np.sin(t * 3), np.cos(t * 2.5)]) * 0.15
        target_e_vel = (escape_dir + noise) * 0.72
        
        e_acc_vec = np.clip((target_e_vel - e_vel) * 0.08, -0.15, 0.15)
        e_acc = np.linalg.norm(e_acc_vec)
        e_vel = e_vel + e_acc_vec * dt
        e_vel = np.clip(e_vel, -0.72, 0.72)
        e_pos = e_pos + e_vel * dt
        e_pos = np.clip(e_pos, -14, 14)
        e_yaw = np.arctan2(e_vel[1], e_vel[0])
        
        # 记录
        pursuer_states.append(AgentState(p_pos.copy(), p_vel.copy(), p_yaw, 
                                        np.linalg.norm(p_vel), p_acc))
        evader_states.append(AgentState(e_pos.copy(), e_vel.copy(), e_yaw,
                                       np.linalg.norm(e_vel), e_acc))
        subgoals.append(p_pos + direction / max(dist, 0.1) * 0.7)
        distances.append(dist)
        
        if dist < 1.0:
            break
    
    return TrajectoryData(
        pursuer_states=pursuer_states,
        evader_states=evader_states,
        subgoals=subgoals,
        distances=distances,
        success=distances[-1] < 1.0,
        dt=dt
    )


# ==================== CLI 入口 ====================

def main():
    parser = argparse.ArgumentParser(description='High-quality pursuit-evasion GIF generator')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='High-level policy checkpoint (best.pth)')
    parser.add_argument('--low-level-ckpt', type=str, default=None,
                       help='Low-level controller checkpoint')
    parser.add_argument('--output', type=str, default='pursuit_demo.gif',
                       help='Output GIF path')
    parser.add_argument('--fps', type=int, default=20, help='GIF frame rate')
    parser.add_argument('--skip', type=int, default=2, help='Skip frames')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic trajectory')
    parser.add_argument('--world-size', type=float, default=30.0)
    parser.add_argument('--catch-radius', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generator = PursuitGIFGenerator(
        world_size=args.world_size,
        catch_radius=args.catch_radius,
        v_max_pursuer=0.65,
        v_max_evader=0.72,
        a_max_pursuer=1.73,
        a_max_evader=0.15
    )
    
    if args.demo:
        print("Generating demo trajectory...")
        traj = create_demo_trajectory(n_steps=400, dt=0.04)
        generator.generate_gif(traj, args.output, fps=args.fps, skip_frames=args.skip,
                              title='Pursuit-Evasion Demo')
    else:
        if args.checkpoint is None:
            print("Error: --checkpoint required (or use --demo)")
            return
        
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载环境和策略
        from high_tabular.hrl_v3_real_auv_fixed import load_v4_nn, PursuitEnvRealAUVFixed
        from training.offline_ppo_pretrain import HighLevelActorCritic
        
        # 底层
        low_ckpt = args.low_level_ckpt or 'low_near/v4_tight_heading_terminal/checkpoints/best.pth'
        nn_policy = load_v4_nn(low_ckpt, device)
        
        # 环境
        env = PursuitEnvRealAUVFixed(
            nn_policy=nn_policy,
            device=device,
            world_size=args.world_size,
            catch_radius=args.catch_radius,
            seed=args.seed
        )
        
        # 高层策略
        ckpt = torch.load(args.checkpoint, map_location=device)
        high_policy = HighLevelActorCritic(obs_dim=21, act_dim=2, hidden=(256, 128))
        
        if 'actor' in ckpt:
            # TD3BC格式
            high_policy.actor_body.load_state_dict(ckpt['actor_body'], strict=False)
            high_policy.mu_head.load_state_dict(ckpt['mu_head'], strict=False)
        else:
            high_policy.load_state_dict(ckpt['model'], strict=False)
        
        high_policy.to(device)
        high_policy.eval()
        
        # 收集轨迹
        print("Running episode...")
        traj = collect_trajectory_from_env(env, high_policy, device)
        print(f"Episode: {traj.n_steps} steps, success={traj.success}")
        
        # 生成GIF
        generator.generate_gif(traj, args.output, fps=args.fps, skip_frames=args.skip,
                              title=f'Pursuit-Evasion ({os.path.basename(args.checkpoint)})')


if __name__ == '__main__':
    main()
