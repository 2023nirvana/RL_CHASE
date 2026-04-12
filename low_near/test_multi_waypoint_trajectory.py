#!/usr/bin/env python3
"""
多路径点轨迹跟踪测试
====================

目的：验证训练好的低层模型在执行较长轨迹时的效果，
      为后续高层控制器连接提供参考。

测试场景：
1. 直线多点：沿直线依次到达多个路径点
2. 矩形轨迹：走矩形路径
3. S形轨迹：测试连续转向能力
4. 随机点序列：模拟高层连续输出子目标

用法：
    python low_near/test_multi_waypoint_trajectory.py --exp-dir low_near/v4_tight
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_continuous import LowLevelAUV2DEnvV3Continuous
from training.train_2d_v3_continuous import GaussianActorCritic


# ============================================================================
# 路径点生成器
# ============================================================================

def generate_line_waypoints(start_pos, direction, num_points=5, spacing=0.8):
    """直线路径点"""
    direction = np.array(direction) / np.linalg.norm(direction)
    waypoints = []
    for i in range(num_points):
        pt = start_pos + direction * spacing * (i + 1)
        waypoints.append(pt)
    return waypoints


def generate_rectangle_waypoints(start_pos, width=2.0, height=1.5):
    """矩形轨迹路径点"""
    x0, y0 = start_pos
    return [
        np.array([x0 + width, y0]),        # 右
        np.array([x0 + width, y0 + height]),  # 右上
        np.array([x0, y0 + height]),        # 左上
        np.array([x0, y0]),                # 回原点
    ]


def generate_s_curve_waypoints(start_pos, amplitude=1.0, wavelength=2.0, num_points=8):
    """S形曲线路径点"""
    waypoints = []
    for i in range(num_points):
        t = (i + 1) / num_points * 2 * np.pi
        x = start_pos[0] + wavelength * t / (2 * np.pi) * num_points / 4
        y = start_pos[1] + amplitude * np.sin(t)
        waypoints.append(np.array([x, y]))
    return waypoints


def generate_random_waypoints(start_pos, num_points=6, min_dist=0.4, max_dist=0.9):
    """随机路径点序列（模拟高层连续输出）"""
    rng = np.random.default_rng(42)
    waypoints = []
    current = np.array(start_pos)
    for _ in range(num_points):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(min_dist, max_dist)
        next_pt = current + dist * np.array([np.cos(angle), np.sin(angle)])
        waypoints.append(next_pt)
        current = next_pt
    return waypoints


def generate_zigzag_waypoints(start_pos, num_points=6, forward=0.8, lateral=0.5):
    """Z字形路径点（测试快速转向）"""
    waypoints = []
    current = np.array(start_pos)
    for i in range(num_points):
        side = 1 if i % 2 == 0 else -1
        next_pt = current + np.array([forward, side * lateral])
        waypoints.append(next_pt)
        current = next_pt
    return waypoints


# ============================================================================
# 轨迹执行器
# ============================================================================

class MultiWaypointTracker:
    """多路径点轨迹跟踪器"""
    
    def __init__(self, model, env, device, reach_threshold=0.12, max_steps_per_wp=150):
        self.model = model
        self.env = env
        self.device = device
        self.reach_threshold = reach_threshold
        self.max_steps_per_wp = max_steps_per_wp
    
    def _set_target(self, target_pos):
        """设置新目标位置"""
        self.env.inner.target_state[:2] = target_pos
        self.env.inner.target_state[2] = self.env.fixed_z
        # 重置 prev_pos_error 以确保奖励计算正确
        eta = self.env.inner.dynamics.eta
        self.env.prev_pos_error = float(np.linalg.norm(eta[:2] - target_pos))
    
    def _get_current_pos(self):
        """获取当前位置"""
        return self.env.inner.dynamics.eta[:2].copy()
    
    def _step_with_fixed_target(self, action):
        """执行单步动作，保持目标不变（绕过 step 中的终止逻辑）"""
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)
        
        eta = self.env.inner.dynamics.eta
        nu = self.env.inner.dynamics.nu
        
        # 动作映射（与环境 step 逻辑一致）
        linear_cmd = np.array([
            float(action[0]) * self.env.surge_step,
            float(action[1]) * self.env.sway_step,
            0.0,
        ], dtype=np.float64)
        
        # 速度环
        current_vel_body = nu[:3]
        desired_vel_body = linear_cmd / max(self.env.inner.dt, 1e-6)
        vel_error = desired_vel_body - current_vel_body
        
        force_cmd = (
            self.env.inner.integrated_controller.vel_kp * vel_error
            - self.env.inner.integrated_controller.vel_kd * current_vel_body
        )
        force_cmd = np.clip(
            force_cmd,
            -self.env.inner.integrated_controller.max_force,
            self.env.inner.integrated_controller.max_force,
        )
        
        # 姿态环
        yaw_current = eta[5]
        yaw_delta = float(action[2]) * self.env.yaw_delta
        target_yaw = np.arctan2(
            np.sin(yaw_current + yaw_delta),
            np.cos(yaw_current + yaw_delta),
        )
        angle_target = np.array([target_yaw, 0.0, 0.0], dtype=np.float64)
        angle_current = np.array([yaw_current, eta[4], eta[3]], dtype=np.float64)
        omega_current = np.array([nu[5], nu[4], nu[3]], dtype=np.float64)
        
        tau_attitude, _ = self.env.inner.integrated_controller.pid_controller.compute(
            angle_target, angle_current, omega_current, self.env.inner.dt
        )
        thrust_command = self.env.inner.integrated_controller.thrust_allocator.allocate_with_attitude(
            force_cmd, tau_attitude
        )
        self.env.inner.last_thrust_command = thrust_command.copy()
        
        # 物理步进
        external_force = self.env.inner._compute_external_force()
        self.env.inner.dynamics.step(thrust_command, external_force)
        self.env.inner.water_current.step(self.env.inner.dt)
        self.env.inner.current_step += 1
        
        self.env._enforce_2d_state()
    
    def _rollout_to_target(self, target_pos, record_trajectory=True):
        """执行到单个目标点"""
        self._set_target(target_pos)
        
        trajectory = []
        steps = 0
        reached = False
        
        for _ in range(self.max_steps_per_wp):
            obs = self.env._get_observation()
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                action, _, _ = self.model.get_action(x, deterministic=True)
            action_np = action[0].cpu().numpy()
            
            # 执行单步
            self._step_with_fixed_target(action_np)
            steps += 1
            
            current_pos = self._get_current_pos()
            if record_trajectory:
                trajectory.append(current_pos.copy())
            
            # 检查是否到达
            dist = np.linalg.norm(current_pos - target_pos)
            if dist < self.reach_threshold:
                reached = True
                break
        
        return {
            "reached": reached,
            "steps": steps,
            "final_dist": np.linalg.norm(self._get_current_pos() - target_pos),
            "trajectory": np.array(trajectory) if trajectory else np.array([]),
        }
    
    def execute_waypoint_sequence(self, waypoints, verbose=True):
        """执行完整路径点序列"""
        # 重置环境
        obs, info = self.env.reset()
        self.env._enforce_2d_state()
        
        start_pos = self._get_current_pos()
        all_trajectory = [start_pos.copy()]
        waypoint_results = []
        total_steps = 0
        success_count = 0
        
        for i, wp in enumerate(waypoints):
            result = self._rollout_to_target(wp, record_trajectory=True)
            waypoint_results.append(result)
            total_steps += result["steps"]
            
            if result["trajectory"].size > 0:
                all_trajectory.extend(result["trajectory"].tolist())
            
            if result["reached"]:
                success_count += 1
                status = "✓"
            else:
                status = f"✗ (dist={result['final_dist']:.3f})"
            
            if verbose:
                print(f"  WP {i+1}/{len(waypoints)}: {wp} -> {status} ({result['steps']} steps)")
        
        return {
            "start_pos": start_pos,
            "waypoints": waypoints,
            "trajectory": np.array(all_trajectory),
            "waypoint_results": waypoint_results,
            "total_steps": total_steps,
            "success_count": success_count,
            "success_rate": success_count / len(waypoints),
        }


# ============================================================================
# 可视化
# ============================================================================

def visualize_trajectory(result, title, save_path):
    """可视化单条轨迹"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    traj = result["trajectory"]
    waypoints = result["waypoints"]
    start = result["start_pos"]
    
    # 绘制轨迹
    ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.5, alpha=0.7, label='Trajectory')
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=150, marker='o', zorder=5, label='Start')
    ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=150, marker='x', zorder=5, label='End')
    
    # 绘制路径点
    wp_array = np.array(waypoints)
    for i, wp in enumerate(waypoints):
        reached = result["waypoint_results"][i]["reached"]
        color = 'green' if reached else 'red'
        ax.scatter(wp[0], wp[1], c=color, s=100, marker='*', zorder=4)
        circle = Circle(wp, 0.12, fill=False, color=color, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.annotate(f'{i+1}', xy=wp, xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 连接路径点（规划路径）
    full_path = np.vstack([[start], wp_array])
    ax.plot(full_path[:, 0], full_path[:, 1], 'g--', lw=1, alpha=0.5, label='Planned Path')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{title}\nSuccess: {result["success_count"]}/{len(waypoints)} ({result["success_rate"]:.0%}), Total Steps: {result["total_steps"]}')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def visualize_all_scenarios(all_results, save_path):
    """汇总所有场景的可视化"""
    n = len(all_results)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 10))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (name, result) in enumerate(all_results.items()):
        ax = axes[i]
        traj = result["trajectory"]
        waypoints = result["waypoints"]
        start = result["start_pos"]
        
        ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.5, alpha=0.7)
        ax.scatter(start[0], start[1], c='green', s=100, marker='o', zorder=5)
        
        for j, wp in enumerate(waypoints):
            reached = result["waypoint_results"][j]["reached"]
            color = 'green' if reached else 'red'
            ax.scatter(wp[0], wp[1], c=color, s=80, marker='*', zorder=4)
        
        ax.set_title(f'{name}\n{result["success_count"]}/{len(waypoints)} ({result["success_rate"]:.0%})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Summary saved: {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def load_model(exp_dir, checkpoint_name, device):
    """加载训练好的模型"""
    ckpt_path = os.path.join(exp_dir, "checkpoints", f"{checkpoint_name}.pth")
    if not os.path.exists(ckpt_path):
        # 尝试直接作为完整路径
        ckpt_path = checkpoint_name if os.path.exists(checkpoint_name) else None
        if ckpt_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {exp_dir}/checkpoints/{checkpoint_name}.pth")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    hidden_cfg = ckpt.get("config", {}).get("network", {}).get("feature_extractor", {}).get("hidden_sizes", [128, 64])
    
    model = GaussianActorCritic(obs_dim, act_dim, hidden=tuple(hidden_cfg)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    print(f"Model loaded: {ckpt_path}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, hidden={hidden_cfg}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Multi-waypoint trajectory tracking test")
    p.add_argument("--exp-dir", type=str, default="low_near/v4_tight", help="Experiment directory")
    p.add_argument("--checkpoint", type=str, default="best", help="Checkpoint name (best/final/latest)")
    p.add_argument("--reach-threshold", type=float, default=0.12, help="Waypoint reach threshold (m)")
    p.add_argument("--max-steps-per-wp", type=int, default=150, help="Max steps per waypoint")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    
    # 设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(PROJECT_ROOT, args.exp_dir) if not os.path.isabs(args.exp_dir) else args.exp_dir
    
    # 输出目录
    out_dir = os.path.join(exp_dir, "waypoint_test")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("Multi-Waypoint Trajectory Tracking Test")
    print("=" * 60)
    print(f"Experiment: {exp_dir}")
    print(f"Device: {device}")
    print(f"Reach threshold: {args.reach_threshold} m")
    print(f"Max steps per waypoint: {args.max_steps_per_wp}")
    print()
    
    # 加载模型
    model = load_model(exp_dir, args.checkpoint, device)
    
    # 创建环境（扩大范围以适应多点轨迹）
    env = LowLevelAUV2DEnvV3Continuous(
        seed=args.seed,
        target_distance_max=2.0,  # 扩大范围
        target_distance_min=0.3,
        reach_threshold=args.reach_threshold,
        max_episode_seconds=60.0,  # 更长时限
    )
    
    # 创建跟踪器
    tracker = MultiWaypointTracker(
        model, env, device,
        reach_threshold=args.reach_threshold,
        max_steps_per_wp=args.max_steps_per_wp
    )
    
    # 定义测试场景
    start_pos = np.array([0.0, 0.0])
    scenarios = {
        "1_line_forward": generate_line_waypoints(start_pos, [1, 0], num_points=5, spacing=0.7),
        "2_line_diagonal": generate_line_waypoints(start_pos, [1, 1], num_points=5, spacing=0.7),
        "3_rectangle": generate_rectangle_waypoints(start_pos, width=1.5, height=1.0),
        "4_s_curve": generate_s_curve_waypoints(start_pos, amplitude=0.6, wavelength=1.5, num_points=6),
        "5_zigzag": generate_zigzag_waypoints(start_pos, num_points=5, forward=0.6, lateral=0.4),
        "6_random": generate_random_waypoints(start_pos, num_points=6, min_dist=0.4, max_dist=0.8),
    }
    
    # 执行测试
    all_results = {}
    summary_stats = []
    
    for name, waypoints in scenarios.items():
        print(f"\n[{name}] {len(waypoints)} waypoints")
        print("-" * 40)
        
        result = tracker.execute_waypoint_sequence(waypoints, verbose=True)
        all_results[name] = result
        
        # 统计
        avg_steps = result["total_steps"] / len(waypoints)
        summary_stats.append({
            "scenario": name,
            "num_waypoints": len(waypoints),
            "success_count": result["success_count"],
            "success_rate": result["success_rate"],
            "total_steps": result["total_steps"],
            "avg_steps_per_wp": avg_steps,
        })
        
        print(f"  Summary: {result['success_rate']:.0%} success, avg {avg_steps:.1f} steps/wp")
        
        # 单独可视化
        save_path = os.path.join(out_dir, f"{name}.png")
        visualize_trajectory(result, name, save_path)
    
    # 汇总可视化
    visualize_all_scenarios(all_results, os.path.join(out_dir, "all_scenarios.png"))
    
    # 保存统计结果
    stats_path = os.path.join(out_dir, "summary_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"\nStats saved: {stats_path}")
    
    # 打印最终汇总
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    total_wp = sum(s["num_waypoints"] for s in summary_stats)
    total_success = sum(s["success_count"] for s in summary_stats)
    overall_rate = total_success / total_wp if total_wp > 0 else 0
    
    print(f"Total waypoints: {total_wp}")
    print(f"Total success: {total_success}")
    print(f"Overall success rate: {overall_rate:.1%}")
    print()
    
    for s in summary_stats:
        print(f"  {s['scenario']:20s}: {s['success_rate']:5.0%} ({s['success_count']}/{s['num_waypoints']}), avg {s['avg_steps_per_wp']:.1f} steps/wp")
    
    print("\n" + "=" * 60)
    print("Analysis for High-Level Integration:")
    print("=" * 60)
    
    if overall_rate >= 0.9:
        print("✓ 低层模型表现优秀，适合直接连接高层控制器")
        print("  建议:")
        print("  - 高层子目标间距: 0.5-0.8m")
        print("  - 低层执行步数: 80-120 steps (根据平均步数)")
        print("  - 高层决策频率: ~100 steps")
    elif overall_rate >= 0.7:
        print("△ 低层模型表现良好，可尝试连接高层但需调参")
        print("  建议:")
        print("  - 减小高层子目标间距")
        print("  - 增加低层执行步数上限")
        print("  - 考虑高层奖励中加入子目标可达性惩罚")
    else:
        print("✗ 低层模型可能需要进一步训练或调整")
        print("  建议:")
        print("  - 检查训练数据中目标分布是否与测试匹配")
        print("  - 可能需要更长距离目标的fine-tuning")


if __name__ == "__main__":
    main()
