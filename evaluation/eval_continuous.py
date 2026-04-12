"""
连续动作模型评估和可视化脚本 (6DOF 力/力矩)
==========================================

使用方法：
    python evaluation/eval_continuous.py --checkpoint checkpoints/continuous_2d/best_model.pth --target-distance 2.0
"""
import os, sys, argparse, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from core.environment.low_level_env_continuous import LowLevelContinuousEnv


class GaussianActorCritic(nn.Module):
    """与 train_continuous 一致的高斯策略网络"""
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 128]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.feature = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs):
        features = self.feature(obs)
        mean = self.actor_mean(features)
        log_std = torch.clamp(self.actor_log_std, -2, 0.5)
        std = torch.exp(log_std)
        value = self.critic(features).squeeze(-1)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        if deterministic:
            action = mean
        else:
            action = torch.distributions.Normal(mean, std).sample()
        action = torch.tanh(action)
        return action, value


class PolicyEvaluatorContinuous:
    def __init__(self, checkpoint_path: str, target_distance: float = 2.0, enable_3d: bool = False, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_distance = target_distance
        self.env = LowLevelContinuousEnv(target_distance=target_distance, enable_3d=enable_3d)
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self._load_model(checkpoint_path)
        print(f"Continuous model loaded from {checkpoint_path}, target={target_distance}m, 3D={enable_3d}")

    def _load_model(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.network = GaussianActorCritic(self.obs_dim, self.action_dim, [256, 128]).to(self.device)
        self.network.load_state_dict(ckpt['network'])
        self.network.eval()

    def get_action(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.network.get_action(obs_t, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def run_episode(self, verbose: bool = True):
        obs, info = self.env.reset()
        trajectory = {'positions': [], 'yaw_angles': [], 'target': info['target_pos'], 'actions': [], 'rewards': []}
        trajectory['positions'].append(self.env.dynamics.eta[:2].tolist())
        trajectory['yaw_angles'].append(self.env.dynamics.eta[5])
        done, total_reward, step = False, 0, 0
        while not done:
            action = self.get_action(obs)
            obs, reward, term, trunc, info = self.env.step(action)
            done = term or trunc
            total_reward += reward
            step += 1
            eta = self.env.dynamics.eta
            trajectory['positions'].append(eta[:2].tolist())
            trajectory['yaw_angles'].append(eta[5])
            trajectory['actions'].append(action.tolist())
            trajectory['rewards'].append(reward)
            if verbose and step % 50 == 0:
                print(f"  Step {step}: pos_error={info.get('pos_error', 0):.3f}m")
        success = info.get('termination_reason') == 'success'
        return {'success': success, 'steps': step, 'total_reward': total_reward, 'final_pos_error': info.get('pos_error', 0),
                'termination_reason': info.get('termination_reason', 'unknown'), 'trajectory': trajectory}

    def evaluate(self, n_episodes: int = 10, verbose: bool = True):
        results = []
        print(f"\n{'='*60}\nEvaluating Continuous ({n_episodes} episodes, target={self.target_distance}m)\n{'='*60}\n")
        for ep in range(n_episodes):
            print(f"--- Episode {ep+1}/{n_episodes} ---")
            results.append(self.run_episode(verbose=verbose))
        successes = sum(1 for r in results if r['success'])
        print(f"\nSummary: Success {successes}/{n_episodes} = {successes/n_episodes:.1%}, Avg reward={np.mean([r['total_reward'] for r in results]):.2f}")
        return results

    def plot_trajectory(self, result: dict, save_path: str = None, title_prefix: str = ""):
        traj = result['trajectory']
        positions = np.array(traj['positions'])
        yaw = np.array(traj['yaw_angles'])
        target = np.array(traj['target'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax1 = axes[0]
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', lw=2, label='Trajectory', alpha=0.7)
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'bs', markersize=10, label='End')
        ax1.plot(target[0], target[1], 'r*', markersize=15, label='Target')
        arrow_interval = max(1, len(positions) // 15)
        al = 0.3
        for i in range(0, len(positions), arrow_interval):
            dx, dy = al * np.cos(yaw[i]), al * np.sin(yaw[i])
            ax1.arrow(positions[i, 0], positions[i, 1], dx, dy, head_width=0.1, head_length=0.05, fc='orange', ec='orange', alpha=0.6)
        ax1.add_patch(plt.Circle((target[0], target[1]), 0.3, color='green', fill=False, linestyle='--', lw=2, label='Success (0.3m)'))
        ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
        ax1.set_title(f'{title_prefix}Continuous Control Trajectory ({"Success" if result["success"] else "Fail"})')
        ax1.legend(loc='best'); ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')
        ax2 = axes[1]
        steps = range(len(traj['rewards']))
        pos_errors = [np.linalg.norm(np.array(traj['positions'][i+1]) - target[:2]) for i in range(len(traj['rewards']))]
        ax2_twin = ax2.twinx()
        ax2.plot(steps, traj['rewards'], 'b-', label='Reward')
        ax2_twin.plot(steps, pos_errors, 'r-', label='Position Error')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Reward', color='b'); ax2_twin.set_ylabel('Position Error (m)', color='r')
        ax2.set_title('Reward and Position Error'); ax2.legend(loc='best'); ax2.grid(True, alpha=0.3)
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
            tgt = np.array(r['trajectory']['target'])
            color = 'green' if r['success'] else 'red'
            alpha = 0.7 if r['success'] else 0.3
            ax.plot(pos[:, 0], pos[:, 1], color=color, alpha=alpha, lw=1.5)
            ax.plot(pos[0, 0], pos[0, 1], 'o', color=color, markersize=6)
            ax.plot(tgt[0], tgt[1], '*', color='gold', markersize=12)
        successes = sum(1 for x in results if x['success'])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'{title_prefix}Continuous Multiple Trajectories (Success: {successes}/{len(results)} = {successes/len(results):.1%})')
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0],[0], color='green', lw=2, label='Success'), Line2D([0],[0], color='red', lw=2, alpha=0.3, label='Fail'), Line2D([0],[0], marker='*', color='gold', markersize=12, linestyle='', label='Target')])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved {save_path}")
        plt.close(fig)
        return fig


def main():
    p = argparse.ArgumentParser(description="Continuous Action Model Evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--target-distance", type=float, default=2.0)
    p.add_argument("--enable-3d", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--save-plots", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--output-prefix", type=str, default="continuous")
    args = p.parse_args()
    evaluator = PolicyEvaluatorContinuous(args.checkpoint, target_distance=args.target_distance, enable_3d=args.enable_3d)
    results = evaluator.evaluate(n_episodes=args.episodes, verbose=args.verbose)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PROJECT_ROOT, "visualization/eval_plots")
    os.makedirs(save_dir, exist_ok=True)
    prefix = (args.output_prefix + "_") if args.output_prefix else ""
    for i, r in enumerate(results):
        path = os.path.join(save_dir, f"{prefix}trajectory_{timestamp}_{i}.png") if args.save_plots else None
        evaluator.plot_trajectory(r, path, title_prefix=f"[{args.output_prefix}] ")
    path_all = os.path.join(save_dir, f"{prefix}all_trajectories_{timestamp}.png") if args.save_plots else None
    evaluator.plot_multiple_trajectories(results, path_all, title_prefix=f"[{args.output_prefix}] ")
    print(f"\nPlots in {save_dir}")


if __name__ == "__main__":
    main()
