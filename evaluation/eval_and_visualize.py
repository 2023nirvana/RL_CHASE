#!/usr/bin/env python3
"""
评估模型并选择代表性案例生成可视化
======================================

评估N个episode，自动选择5个代表性案例:
1. 最快成功 (fastest success)
2. 最慢成功 (slowest success) 
3. 边界成功 (close call - 接近失败)
4. 典型成功 (median success)
5. 典型失败 (if any failures)

生成每个案例的GIF和PNG。
"""

import os
import sys
import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@dataclass
class EpisodeRecord:
    """单个episode记录"""
    seed: int
    success: bool
    steps: int
    total_reward: float
    min_distance: float
    final_distance: float
    collision: bool
    # 轨迹数据（用于可视化）
    pursuer_trajectory: Optional[List] = None
    evader_trajectory: Optional[List] = None
    pursuer_velocities: Optional[List] = None
    evader_velocities: Optional[List] = None
    distances: Optional[List] = None
    subgoals: Optional[List] = None


def evaluate_episodes(
    checkpoint_path: str,
    low_level_ckpt: str,
    n_episodes: int = 100,
    world_size: float = 30.0,
    catch_radius: float = 1.0,
    evader_mode: str = "medium",
    subgoal_range: float = 0.7,
    low_steps: int = 50,
    max_steps: int = 6000,
    deterministic: bool = True,
    verbose: bool = True,
    save_trajectories: bool = True,
) -> List[EpisodeRecord]:
    """评估多个episode并记录详细信息"""
    
    from high_tabular.hrl_v3_real_auv_fixed import load_v4_nn, PursuitEnvRealAUVFixed
    from training.offline_ppo_pretrain import HighLevelActorCritic
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载策略
    nn_policy = load_v4_nn(low_level_ckpt, device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'obs_dim' in ckpt:
        obs_dim = ckpt['obs_dim']
        act_dim = ckpt['act_dim']
        hidden = ckpt.get('hidden', (256, 128))
    else:
        obs_dim, act_dim, hidden = 21, 2, (256, 128)
    
    high_policy = HighLevelActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
    if 'model' in ckpt:
        high_policy.load_state_dict(ckpt['model'], strict=False)
    elif 'actor_body' in ckpt:
        high_policy.actor_body.load_state_dict(ckpt['actor_body'], strict=False)
        high_policy.mu_head.load_state_dict(ckpt['mu_head'], strict=False)
    high_policy.to(device)
    high_policy.eval()
    
    records = []
    
    for ep in range(n_episodes):
        seed = ep
        env = PursuitEnvRealAUVFixed(
            nn_policy=nn_policy,
            device=device,
            world_size=world_size,
            catch_radius=catch_radius,
            max_steps=max_steps,
            subgoal_range=subgoal_range,
            low_steps=low_steps,
            seed=seed,
            evader_mode=evader_mode,
        )
        
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        min_distance = float('inf')
        collision = False
        
        # 轨迹记录
        p_traj, e_traj = [], []
        p_vel, e_vel = [], []
        dists = []
        sgs = []
        
        while not done and steps < max_steps // low_steps:
            eta = env.auv_env.inner.dynamics.eta
            nu = env.auv_env.inner.dynamics.nu
            
            p_pos = [float(eta[0]), float(eta[1])]
            e_pos = env.e.tolist()
            
            p_traj.append(p_pos)
            e_traj.append(e_pos)
            p_vel.append([float(nu[0]), float(nu[1])])
            e_vel.append(env.ve.tolist())
            
            dist = np.linalg.norm(np.array(p_pos) - np.array(e_pos))
            dists.append(float(dist))
            min_distance = min(min_distance, dist)
            
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                result = high_policy.get_action(obs_t, deterministic=deterministic)
                action = result[0].cpu().numpy().flatten()
            
            # 计算子目标
            sg = np.array(p_pos) + action * env.subgoal_range
            sgs.append(sg.tolist())
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if info.get('collision', False):
                collision = True
        
        # 最终状态
        eta = env.auv_env.inner.dynamics.eta
        final_p = np.array([eta[0], eta[1]])
        final_e = env.e
        final_distance = float(np.linalg.norm(final_p - final_e))
        success = info.get('captured', False) or final_distance < catch_radius
        
        record = EpisodeRecord(
            seed=seed,
            success=success,
            steps=steps,
            total_reward=total_reward,
            min_distance=min_distance,
            final_distance=final_distance,
            collision=collision,
        )
        
        if save_trajectories:
            record.pursuer_trajectory = p_traj
            record.evader_trajectory = e_traj
            record.pursuer_velocities = p_vel
            record.evader_velocities = e_vel
            record.distances = dists
            record.subgoals = sgs
        
        records.append(record)
        
        if verbose and (ep + 1) % 10 == 0:
            success_rate = sum(1 for r in records if r.success) / len(records) * 100
            print(f"Evaluated {ep+1}/{n_episodes} episodes, success rate: {success_rate:.1f}%")
    
    return records


def select_representative_episodes(
    records: List[EpisodeRecord],
    n_cases: int = 5
) -> Dict[str, EpisodeRecord]:
    """选择代表性案例"""
    
    successes = [r for r in records if r.success]
    failures = [r for r in records if not r.success]
    
    selected = {}
    
    # 1. 最快成功
    if successes:
        fastest = min(successes, key=lambda r: r.steps)
        selected['fastest_success'] = fastest
    
    # 2. 最慢成功
    if successes:
        slowest = max(successes, key=lambda r: r.steps)
        selected['slowest_success'] = slowest
    
    # 3. 边界成功 (最小距离最接近catch_radius)
    if successes:
        close_call = max(successes, key=lambda r: r.min_distance)
        selected['close_call'] = close_call
    
    # 4. 典型成功 (中位数步数)
    if len(successes) >= 3:
        sorted_by_steps = sorted(successes, key=lambda r: r.steps)
        median_idx = len(sorted_by_steps) // 2
        selected['median_success'] = sorted_by_steps[median_idx]
    elif successes:
        selected['median_success'] = successes[0]
    
    # 5. 典型失败
    if failures:
        # 选择最接近成功的失败（最小距离最小）
        best_failure = min(failures, key=lambda r: r.min_distance)
        selected['best_failure'] = best_failure
    
    return selected


def generate_visualizations(
    selected: Dict[str, EpisodeRecord],
    output_dir: str,
    checkpoint_name: str,
    catch_radius: float = 1.0,
    fps: int = 10,
    skip_frames: int = 1,
):
    """为选中的案例生成GIF和PNG"""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 障碍物定义
    obstacles = [
        (0.0, 2.0, 8.0, 2.0),
        (0.0, -2.0, 2.5, 6.0),
        (-9.0, 9.0, 4.0, 2.0),
        (-10.0, 7.0, 2.0, 4.0),
        (9.0, 9.0, 4.0, 2.0),
        (10.0, 7.0, 2.0, 4.0),
        (-9.0, -9.0, 4.0, 2.0),
        (-10.0, -7.0, 2.0, 4.0),
        (9.0, -9.0, 4.0, 2.0),
        (10.0, -7.0, 2.0, 4.0),
        (-5.5, -5.0, 2.5, 2.5),
        (5.5, 5.0, 2.5, 2.5),
    ]
    
    for case_name, record in selected.items():
        print(f"Generating visualization for {case_name}...")
        
        p_traj = np.array(record.pursuer_trajectory)
        e_traj = np.array(record.evader_trajectory)
        p_vel = np.array(record.pursuer_velocities)
        e_vel = np.array(record.evader_velocities)
        dists = np.array(record.distances)
        
        n_frames = len(p_traj)
        t_axis = np.arange(n_frames)
        speed_p = np.linalg.norm(p_vel, axis=1)
        speed_e = np.linalg.norm(e_vel, axis=1)
        
        # === 生成PNG (静态总结图) ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 轨迹图
        ax = axes[0, 0]
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_aspect('equal')
        ax.plot([-15, 15, 15, -15, -15], [-15, -15, 15, 15, -15], 'k-', lw=2)
        for cx, cy, w, h in obstacles:
            rect = Rectangle((cx - w/2, cy - h/2), w, h, 
                            facecolor='gray', edgecolor='black', alpha=0.6)
            ax.add_patch(rect)
        
        ax.plot(p_traj[:, 0], p_traj[:, 1], 'b-', lw=2, alpha=0.8, label='Pursuer')
        ax.plot(e_traj[:, 0], e_traj[:, 1], 'r-', lw=2, alpha=0.8, label='Evader')
        ax.scatter(p_traj[0, 0], p_traj[0, 1], c='blue', s=100, marker='o', zorder=5)
        ax.scatter(e_traj[0, 0], e_traj[0, 1], c='red', s=100, marker='o', zorder=5)
        ax.scatter(p_traj[-1, 0], p_traj[-1, 1], c='blue', s=150, marker='*', zorder=5)
        ax.scatter(e_traj[-1, 0], e_traj[-1, 1], c='red', s=150, marker='*', zorder=5)
        
        catch_circle = Circle((p_traj[-1, 0], p_traj[-1, 1]), catch_radius,
                              fill=False, color='green', lw=2, ls='--')
        ax.add_patch(catch_circle)
        
        status = "SUCCESS" if record.success else "FAILED"
        ax.set_title(f'{case_name}: {status} (steps={record.steps})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 速度图
        ax = axes[0, 1]
        ax.plot(t_axis, speed_p, 'b-', lw=2, label='Pursuer')
        ax.plot(t_axis, speed_e, 'r-', lw=2, label='Evader')
        ax.axhline(0.72, ls='--', color='red', alpha=0.5, label='Evader v_max')
        ax.axhline(0.65, ls='--', color='blue', alpha=0.5, label='Pursuer v_eff')
        ax.set_xlabel('Step')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Velocity Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 距离图
        ax = axes[1, 0]
        ax.plot(t_axis, dists, 'g-', lw=2)
        ax.axhline(catch_radius, ls='--', color='red', lw=2, label=f'Catch radius={catch_radius}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Distance to Evader')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 统计信息
        ax = axes[1, 1]
        ax.axis('off')
        info_text = f"""
Episode Statistics
==================
Seed: {record.seed}
Status: {status}
Steps: {record.steps}
Total Reward: {record.total_reward:.2f}
Min Distance: {record.min_distance:.3f} m
Final Distance: {record.final_distance:.3f} m
Collision: {record.collision}
        """
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        png_path = os.path.join(output_dir, f'{checkpoint_name}_{case_name}.png')
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {png_path}")
        
        # === 生成GIF (动态回放) ===
        frames = list(range(0, n_frames, max(1, skip_frames)))
        if frames[-1] != n_frames - 1:
            frames.append(n_frames - 1)
        
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], wspace=0.25, hspace=0.3)
        ax_world = fig.add_subplot(gs[:, 0])
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_speed = fig.add_subplot(gs[1, 1])
        
        # 世界设置
        ax_world.set_xlim(-16, 16)
        ax_world.set_ylim(-16, 16)
        ax_world.set_aspect('equal')
        ax_world.plot([-15, 15, 15, -15, -15], [-15, -15, 15, 15, -15], 'k-', lw=2)
        for cx, cy, w, h in obstacles:
            rect = Rectangle((cx - w/2, cy - h/2), w, h,
                            facecolor='gray', edgecolor='black', alpha=0.6)
            ax_world.add_patch(rect)
        ax_world.grid(True, alpha=0.3)
        ax_world.set_title(f'{case_name}')
        
        p_line, = ax_world.plot([], [], 'b-', lw=2, alpha=0.8, label='Pursuer')
        e_line, = ax_world.plot([], [], 'r-', lw=2, alpha=0.8, label='Evader')
        p_dot = ax_world.scatter([], [], c='blue', s=100, zorder=6)
        e_dot = ax_world.scatter([], [], c='red', s=100, zorder=6)
        txt = ax_world.text(0.02, 0.98, '', transform=ax_world.transAxes, va='top', fontsize=10)
        ax_world.legend(loc='upper right')
        
        # 距离图设置
        line_dist, = ax_dist.plot([], [], 'g-', lw=2)
        cursor_d = ax_dist.axvline(0, color='gray', ls='--', alpha=0.7)
        ax_dist.axhline(catch_radius, ls=':', color='red', lw=1.5, label=f'catch_r={catch_radius}')
        ax_dist.set_xlim(0, n_frames - 1)
        ax_dist.set_ylim(0, max(float(np.max(dists)), 2.0) * 1.1)
        ax_dist.set_xlabel('Step')
        ax_dist.set_ylabel('Distance (m)')
        ax_dist.set_title('Distance to Evader')
        ax_dist.legend(loc='upper right')
        ax_dist.grid(True, alpha=0.3)
        
        # 速度图设置
        line_vp, = ax_speed.plot([], [], 'b-', lw=2, label='Pursuer')
        line_ve, = ax_speed.plot([], [], 'r-', lw=2, label='Evader')
        cursor_s = ax_speed.axvline(0, color='gray', ls='--', alpha=0.7)
        ax_speed.set_xlim(0, n_frames - 1)
        vmax = max(float(np.max(speed_p)), float(np.max(speed_e)), 0.1)
        ax_speed.set_ylim(0, vmax * 1.25)
        ax_speed.set_xlabel('Step')
        ax_speed.set_ylabel('Speed (m/s)')
        ax_speed.set_title('Speed Profile')
        ax_speed.legend(loc='upper right')
        ax_speed.grid(True, alpha=0.3)
        
        def update(i):
            t = frames[i]
            
            p_line.set_data(p_traj[:t+1, 0], p_traj[:t+1, 1])
            e_line.set_data(e_traj[:t+1, 0], e_traj[:t+1, 1])
            p_dot.set_offsets([p_traj[t, 0], p_traj[t, 1]])
            e_dot.set_offsets([e_traj[t, 0], e_traj[t, 1]])
            
            line_dist.set_data(t_axis[:t+1], dists[:t+1])
            line_vp.set_data(t_axis[:t+1], speed_p[:t+1])
            line_ve.set_data(t_axis[:t+1], speed_e[:t+1])
            cursor_d.set_xdata([t, t])
            cursor_s.set_xdata([t, t])
            
            status_str = "CAPTURE!" if record.success and t == n_frames - 1 else ""
            txt.set_text(f'step={t:03d} d={dists[t]:.2f}m {status_str}')
            
            return [p_line, e_line, p_dot, e_dot, txt, line_dist, line_vp, line_ve, cursor_d, cursor_s]
        
        ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000/fps), blit=False)
        gif_path = os.path.join(output_dir, f'{checkpoint_name}_{case_name}.gif')
        ani.save(gif_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"  Saved: {gif_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize representative episodes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--low-level-ckpt', type=str, 
                       default='low_near/v4_tight_heading_terminal/checkpoints/best.pth')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: checkpoint_dir/eval_visuals)')
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--n-cases', type=int, default=5)
    parser.add_argument('--world-size', type=float, default=30.0)
    parser.add_argument('--catch-radius', type=float, default=1.0)
    parser.add_argument('--evader-mode', type=str, default='medium')
    parser.add_argument('--subgoal-range', type=float, default=0.7)
    parser.add_argument('--low-steps', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--skip-frames', type=int, default=1)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--save-json', action='store_true', help='Save evaluation results as JSON')
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        args.output_dir = os.path.join(ckpt_dir, 'eval_visuals')
    
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    
    print(f"Evaluating {args.n_episodes} episodes...")
    records = evaluate_episodes(
        checkpoint_path=args.checkpoint,
        low_level_ckpt=args.low_level_ckpt,
        n_episodes=args.n_episodes,
        world_size=args.world_size,
        catch_radius=args.catch_radius,
        evader_mode=args.evader_mode,
        subgoal_range=args.subgoal_range,
        low_steps=args.low_steps,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
    )
    
    # 统计
    n_success = sum(1 for r in records if r.success)
    n_collision = sum(1 for r in records if r.collision)
    avg_steps = np.mean([r.steps for r in records if r.success]) if n_success > 0 else 0
    avg_reward = np.mean([r.total_reward for r in records])
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Success Rate: {n_success}/{len(records)} ({100*n_success/len(records):.1f}%)")
    print(f"Collision Rate: {n_collision}/{len(records)} ({100*n_collision/len(records):.1f}%)")
    print(f"Avg Steps (success): {avg_steps:.1f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    
    # 选择代表性案例
    print(f"\nSelecting {args.n_cases} representative cases...")
    selected = select_representative_episodes(records, n_cases=args.n_cases)
    
    for name, rec in selected.items():
        status = "SUCCESS" if rec.success else "FAILED"
        print(f"  {name}: seed={rec.seed}, {status}, steps={rec.steps}, min_dist={rec.min_distance:.3f}")
    
    # 生成可视化
    print(f"\nGenerating visualizations...")
    generate_visualizations(
        selected=selected,
        output_dir=args.output_dir,
        checkpoint_name=checkpoint_name,
        catch_radius=args.catch_radius,
        fps=args.fps,
        skip_frames=args.skip_frames,
    )
    
    # 保存JSON结果
    if args.save_json:
        # 移除轨迹数据以减小文件大小
        records_slim = []
        for r in records:
            r_dict = asdict(r)
            r_dict.pop('pursuer_trajectory', None)
            r_dict.pop('evader_trajectory', None)
            r_dict.pop('pursuer_velocities', None)
            r_dict.pop('evader_velocities', None)
            r_dict.pop('distances', None)
            r_dict.pop('subgoals', None)
            records_slim.append(r_dict)
        
        json_path = os.path.join(args.output_dir, f'{checkpoint_name}_eval_results.json')
        with open(json_path, 'w') as f:
            json.dump({
                'summary': {
                    'success_rate': n_success / len(records),
                    'collision_rate': n_collision / len(records),
                    'avg_steps_success': avg_steps,
                    'avg_reward': avg_reward,
                    'n_episodes': len(records),
                },
                'selected_cases': {k: asdict(v) for k, v in selected.items()},
                'all_episodes': records_slim,
            }, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        print(f"Saved: {json_path}")
    
    print(f"\nDone! Visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
