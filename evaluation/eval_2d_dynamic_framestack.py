"""
2D 动态目标帧堆叠模型评估和可视化脚本 (含轨迹图 + GIF/视频)
=============================================================

使用方法：
    # Stack-4 模型评估
    python evaluation/eval_2d_dynamic_framestack.py --checkpoint checkpoints/dynamic_stack4/best_model.pth --n-stack 4
    
    # Stack-8 模型 + 生成GIF
    python evaluation/eval_2d_dynamic_framestack.py --checkpoint checkpoints/dynamic_stack8/best_model.pth --n-stack 8 --save-gif

    # 圆周运动目标
    python evaluation/eval_2d_dynamic_framestack.py --checkpoint checkpoints/dynamic_stack4/best_model.pth --n-stack 4 --target-motion circular
"""
import os, sys, argparse, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from core.environment.low_level_env_2d_dynamic import LowLevelAUV2DDynamicEnv


class FrameStackEnv:
    """帧堆叠环境包装器"""
    def __init__(self, env, n_stack=4):
        self.env = env
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        self.obs_dim = env.obs_dim * n_stack
        
    @property
    def observation_space(self):
        return type('Space', (), {'shape': (self.obs_dim,)})()
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def target_pos(self):
        return self.env.target_pos
    
    @property
    def target_vel(self):
        return self.env.target_vel
    
    @property
    def target_motion(self):
        return self.env.target_motion
    
    @property
    def inner(self):
        return self.env.inner
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_stacked_obs(), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, term, trunc, info
    
    def _get_stacked_obs(self):
        return np.concatenate(list(self.frames))


class PolicyEvaluatorFrameStack:
    """帧堆叠策略评估器"""
    
    def __init__(self, checkpoint_path: str, n_stack: int = 4, target_distance: float = 3.0,
                 target_speed: float = 0.3, target_motion: str = 'constant', device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_stack = n_stack
        
        base_env = LowLevelAUV2DDynamicEnv(
            target_distance=target_distance, target_speed=target_speed, target_motion=target_motion
        )
        self.env = FrameStackEnv(base_env, n_stack=n_stack)
        
        self.obs_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self._load_model(checkpoint_path)
        print(f"FrameStack model loaded from {checkpoint_path}")
        print(f"  n_stack={n_stack}, obs_dim={self.obs_dim}, motion={target_motion}")

    def _load_model(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # 帧堆叠版本使用较大的第一层
        hidden1, hidden2 = 256, 64
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, self.num_actions)
        ).to(self.device)
        
        self.actor.load_state_dict(ckpt['actor'])
        self.actor.eval()

    def get_action(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actor(obs_t)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item() if deterministic else torch.distributions.Categorical(probs).sample().item()
        return action

    def run_episode(self, verbose: bool = True):
        obs, info = self.env.reset()
        trajectory = {
            'positions': [], 'yaw_angles': [], 'target_positions': [], 'target_velocities': [],
            'actions': [], 'rewards': [], 'step_target': []
        }
        trajectory['positions'].append(self.env.inner.dynamics.eta[:2].tolist())
        trajectory['yaw_angles'].append(self.env.inner.dynamics.eta[5])
        trajectory['step_target'].append(self.env.target_pos.tolist())
        
        done, total_reward, step = False, 0, 0
        while not done:
            action = self.get_action(obs)
            obs, reward, term, trunc, info = self.env.step(action)
            done = term or trunc
            total_reward += reward
            step += 1
            
            trajectory['positions'].append(self.env.inner.dynamics.eta[:2].tolist())
            trajectory['yaw_angles'].append(self.env.inner.dynamics.eta[5])
            trajectory['step_target'].append(info['target_pos'])
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            
            if verbose and step % 80 == 0:
                print(f"  Step {step}: pos_error={info.get('pos_error', 0):.3f}m, action={LowLevelAUV2DDynamicEnv.ACTION_NAMES[action]}")
        
        success = info.get('termination_reason') == 'success'
        return {
            'success': success, 'steps': step, 'total_reward': total_reward,
            'final_pos_error': info.get('pos_error', 0), 'termination_reason': info.get('termination_reason', 'unknown'),
            'trajectory': trajectory
        }

    def evaluate(self, n_episodes: int = 10, verbose: bool = True):
        results = []
        print(f"\n{'='*60}\nEvaluating FrameStack ({n_episodes} eps, stack={self.n_stack}, motion={self.env.target_motion})\n{'='*60}\n")
        for ep in range(n_episodes):
            print(f"--- Episode {ep+1}/{n_episodes} ---")
            results.append(self.run_episode(verbose=verbose))
        successes = sum(1 for r in results if r['success'])
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        print(f"\nSummary: Success {successes}/{n_episodes} = {successes/n_episodes:.1%}, Avg reward={avg_reward:.2f}, Avg steps={avg_steps:.0f}")
        return results

    def plot_trajectory(self, result: dict, save_path: str = None, title_prefix: str = ""):
        """静态图：AUV 轨迹 + 动态目标轨迹"""
        traj = result['trajectory']
        positions = np.array(traj['positions'])
        target_path = np.array(traj['step_target'])
        yaw = np.array(traj['yaw_angles'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：轨迹
        ax1 = axes[0]
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', lw=2, label='AUV Trajectory', alpha=0.7)
        ax1.plot(target_path[:, 0], target_path[:, 1], 'r--', lw=1.5, label='Target Path', alpha=0.8)
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='AUV Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=10, label='AUV End')
        ax1.plot(target_path[0, 0], target_path[0, 1], 'r^', markersize=10, label='Target Start')
        ax1.plot(target_path[-1, 0], target_path[-1, 1], 'r*', markersize=14, label='Target End')
        
        # 航向箭头
        arrow_interval = max(1, len(positions) // 12)
        al = 0.25
        for i in range(0, len(positions), arrow_interval):
            dx, dy = al * np.cos(yaw[i]), al * np.sin(yaw[i])
            ax1.arrow(positions[i, 0], positions[i, 1], dx, dy, head_width=0.08, head_length=0.04, fc='orange', ec='orange', alpha=0.6)
        
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
        status = "Success" if result["success"] else "Fail"
        ax1.set_title(f'{title_prefix}FrameStack-{self.n_stack}: AUV vs Target ({status})')
        ax1.legend(loc='best', fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
        
        # 右图：奖励和距离
        ax2 = axes[1]
        steps = range(len(traj['rewards']))
        pos_errors = [np.linalg.norm(np.array(traj['positions'][i+1]) - np.array(traj['step_target'][i+1])) for i in range(len(traj['rewards']))]
        ax2_twin = ax2.twinx()
        ax2.plot(steps, traj['rewards'], 'b-', label='Reward')
        ax2_twin.plot(steps, pos_errors, 'r-', label='Distance to Target')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Reward', color='b'); ax2_twin.set_ylabel('Distance (m)', color='r')
        ax2.set_title('Reward and Distance to Moving Target'); ax2.legend(loc='best'); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved {save_path}")
        plt.close(fig)
        return fig

    def plot_multiple_trajectories(self, results: list, save_path: str = None, title_prefix: str = ""):
        fig, ax = plt.subplots(figsize=(10, 10))
        for r in results:
            pos = np.array(r['trajectory']['positions'])
            tgt_path = np.array(r['trajectory']['step_target'])
            color = 'green' if r['success'] else 'red'
            alpha = 0.7 if r['success'] else 0.3
            ax.plot(pos[:, 0], pos[:, 1], color=color, alpha=alpha, lw=1.5)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6)
            ax.plot(tgt_path[:, 0], tgt_path[:, 1], '--', color='gray', alpha=0.5, lw=1)
        
        successes = sum(1 for x in results if x['success'])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'{title_prefix}FrameStack-{self.n_stack} Trajectories (Success: {successes}/{len(results)} = {successes/len(results):.1%})')
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0], color='green', lw=2, label='Success'), Line2D([0],[0], color='red', lw=2, alpha=0.3, label='Fail')])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved {save_path}")
        plt.close(fig)
        return fig

    def save_trajectory_gif(self, result: dict, save_path: str, fps: int = 15, every_n: int = 2):
        """将单条轨迹保存为 GIF"""
        traj = result['trajectory']
        positions = np.array(traj['positions'])
        target_path = np.array(traj['step_target'])
        n_frames = len(positions)
        indices = list(range(0, n_frames, every_n))
        if indices[-1] != n_frames - 1:
            indices.append(n_frames - 1)
        
        try:
            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter
        except ImportError:
            print("matplotlib.animation not available, skipping GIF")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        margin = 1.5
        x_min = min(positions[:, 0].min(), target_path[:, 0].min()) - margin
        x_max = max(positions[:, 0].max(), target_path[:, 0].max()) + margin
        y_min = min(positions[:, 1].min(), target_path[:, 1].min()) - margin
        y_max = max(positions[:, 1].max(), target_path[:, 1].max()) + margin
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        
        line_auv, = ax.plot([], [], 'b-', lw=2, label='AUV')
        line_tgt, = ax.plot([], [], 'r--', lw=1.5, label='Target')
        point_auv, = ax.plot([], [], 'go', markersize=14)
        point_tgt, = ax.plot([], [], 'r*', markersize=18)
        ax.legend(loc='upper right')
        title_text = ax.set_title('')

        def init():
            line_auv.set_data([], []); line_tgt.set_data([], [])
            point_auv.set_data([], []); point_tgt.set_data([], [])
            title_text.set_text('')
            return line_auv, line_tgt, point_auv, point_tgt, title_text

        def animate(i):
            idx = indices[i] if i < len(indices) else indices[-1]
            line_auv.set_data(positions[:idx+1, 0], positions[:idx+1, 1])
            line_tgt.set_data(target_path[:idx+1, 0], target_path[:idx+1, 1])
            point_auv.set_data([positions[idx, 0]], [positions[idx, 1]])
            point_tgt.set_data([target_path[idx, 0]], [target_path[idx, 1]])
            status = "Success" if result["success"] else "Fail"
            title_text.set_text(f'FrameStack-{self.n_stack} | Step {idx}/{n_frames-1} ({status})')
            return line_auv, line_tgt, point_auv, point_tgt, title_text

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(indices), interval=1000//fps, blit=True)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        print(f"GIF saved: {save_path}")

    def save_trajectory_mp4(self, result: dict, save_path: str, fps: int = 15, every_n: int = 2):
        """将单条轨迹保存为 MP4"""
        traj = result['trajectory']
        positions = np.array(traj['positions'])
        target_path = np.array(traj['step_target'])
        n_frames = len(positions)
        indices = list(range(0, n_frames, every_n))
        if indices[-1] != n_frames - 1:
            indices.append(n_frames - 1)
        
        try:
            import matplotlib.animation as animation
            from matplotlib.animation import FFMpegWriter
        except ImportError:
            print("matplotlib.animation not available, skipping MP4")
            return
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        margin = 1.5
        x_min = min(positions[:, 0].min(), target_path[:, 0].min()) - margin
        x_max = max(positions[:, 0].max(), target_path[:, 0].max()) + margin
        y_min = min(positions[:, 1].min(), target_path[:, 1].min()) - margin
        y_max = max(positions[:, 1].max(), target_path[:, 1].max()) + margin
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        
        line_auv, = ax.plot([], [], 'b-', lw=2, label='AUV')
        line_tgt, = ax.plot([], [], 'r--', lw=1.5, label='Target')
        point_auv, = ax.plot([], [], 'go', markersize=14)
        point_tgt, = ax.plot([], [], 'r*', markersize=18)
        ax.legend(loc='upper right')
        title_text = ax.set_title('')

        def init():
            line_auv.set_data([], []); line_tgt.set_data([], [])
            point_auv.set_data([], []); point_tgt.set_data([], [])
            title_text.set_text('')
            return line_auv, line_tgt, point_auv, point_tgt, title_text

        def animate(i):
            idx = indices[i] if i < len(indices) else indices[-1]
            line_auv.set_data(positions[:idx+1, 0], positions[:idx+1, 1])
            line_tgt.set_data(target_path[:idx+1, 0], target_path[:idx+1, 1])
            point_auv.set_data([positions[idx, 0]], [positions[idx, 1]])
            point_tgt.set_data([target_path[idx, 0]], [target_path[idx, 1]])
            status = "Success" if result["success"] else "Fail"
            title_text.set_text(f'FrameStack-{self.n_stack} | Step {idx}/{n_frames-1} ({status})')
            return line_auv, line_tgt, point_auv, point_tgt, title_text

        try:
            writer = FFMpegWriter(fps=fps)
            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(indices), interval=1000//fps, blit=True)
            anim.save(save_path, writer=writer)
            print(f"MP4 saved: {save_path}")
        except Exception as e:
            print(f"MP4 save failed (ffmpeg?): {e}")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="2D Dynamic Target FrameStack Model Evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n-stack", type=int, default=4, help="Frame stack size (must match training)")
    p.add_argument("--target-distance", type=float, default=3.0)
    p.add_argument("--target-speed", type=float, default=0.3)
    p.add_argument("--target-motion", type=str, default='constant', choices=['constant', 'random_walk', 'circular', 'evasive'])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--save-plots", action="store_true", default=True)
    p.add_argument("--save-gif", action="store_true", help="Save trajectory as GIF for first N episodes")
    p.add_argument("--save-mp4", action="store_true", help="Save trajectory as MP4 (requires ffmpeg)")
    p.add_argument("--gif-fps", type=int, default=15)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--output-prefix", type=str, default="")
    args = p.parse_args()

    evaluator = PolicyEvaluatorFrameStack(
        args.checkpoint, n_stack=args.n_stack,
        target_distance=args.target_distance, target_speed=args.target_speed, target_motion=args.target_motion
    )
    results = evaluator.evaluate(n_episodes=args.episodes, verbose=args.verbose)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "visualization/eval_plots")
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"stack{args.n_stack}_" + (args.output_prefix + "_" if args.output_prefix else "")

    for i, r in enumerate(results):
        path = os.path.join(save_dir, f"{prefix}trajectory_{timestamp}_{i}.png") if args.save_plots else None
        evaluator.plot_trajectory(r, path, title_prefix=f"[Stack-{args.n_stack}] ")
        if args.save_gif and i < 3:
            gif_path = os.path.join(save_dir, f"{prefix}trajectory_{timestamp}_{i}.gif")
            evaluator.save_trajectory_gif(r, gif_path, fps=args.gif_fps)
        if args.save_mp4 and i < 2:
            mp4_path = os.path.join(save_dir, f"{prefix}trajectory_{timestamp}_{i}.mp4")
            evaluator.save_trajectory_mp4(r, mp4_path, fps=args.gif_fps)

    path_all = os.path.join(save_dir, f"{prefix}all_trajectories_{timestamp}.png") if args.save_plots else None
    evaluator.plot_multiple_trajectories(results, path_all, title_prefix=f"[Stack-{args.n_stack}] ")
    print(f"\nPlots (and GIF/MP4 if requested) in {save_dir}")


if __name__ == "__main__":
    main()
