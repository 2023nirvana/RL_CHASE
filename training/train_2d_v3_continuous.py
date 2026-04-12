"""
2D V3 连续动作环境训练脚本（小目标到达 + 简化奖励）
====================================================

并行实验入口；结果输出至 low_near/ 目录。

用法示例：
    python training/train_2d_v3_continuous.py --experiment-name low_near_exp1 --total-timesteps 1000000
"""

import os
import sys
import argparse
import yaml
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_continuous import (
    LowLevelAUV2DEnvV3Continuous,
    VecLowLevelAUV2DEnvV3Continuous,
)


# =========================================================================
# 网络：与 V2 训练脚本完全相同的 GaussianActorCritic
# =========================================================================

class GaussianActorCritic(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 64)):
        super().__init__()
        actor_layers = []
        prev = obs_dim
        for h in hidden:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*actor_layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        critic_layers = []
        prev = obs_dim
        for h in hidden:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.critic = nn.Sequential(*critic_layers, nn.Linear(prev, 1))

    def forward(self, x):
        mu = self.mu_head(self.actor_body(x))
        v = self.critic(x).squeeze(-1)
        return mu, v

    def get_dist(self, x):
        mu, _ = self.forward(x)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def get_action(self, x, deterministic=False):
        dist = self.get_dist(x)
        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample()
        a = torch.tanh(z)
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return a, logp, ent

    def evaluate_actions(self, x, a):
        dist = self.get_dist(x)
        z = self._atanh(a)
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return logp, ent

    def get_value(self, x):
        return self.critic(x).squeeze(-1)


# =========================================================================
# Trainer
# =========================================================================

class Continuous2DTrainerV3:
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.experiment_name = args.experiment_name or f"low_near_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.out_dir = os.path.join(PROJECT_ROOT, "low_near", self.experiment_name)
        self.checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_environments()
        self._init_network()

        self.success_count = 0
        self.episode_count = 0
        self.history = []

    def _init_environments(self):
        training_config = self.config["training"]
        self.num_envs = int(training_config.get("n_envs", 4))

        env_kwargs = dict(
            target_distance_max=self.args.target_distance_max,
            target_distance_min=self.args.target_distance_min,
            reach_threshold=self.args.reach_threshold,
            max_episode_seconds=self.args.max_episode_seconds,
            reward_reach=self.args.reward_reach,
            reward_speed_bonus=self.args.reward_speed_bonus,
            reward_improve_scale=self.args.reward_improve_scale,
            reward_heading_scale=self.args.reward_heading_scale,
            reward_heading_terminal=self.args.reward_heading_terminal,
            reward_vel_align_scale=self.args.reward_vel_align_scale,
            reward_time_penalty=self.args.reward_time_penalty,
            init_velocity_range=self.args.init_velocity_range,
            init_yaw_rate_range=self.args.init_yaw_rate_range,
        )
        self.train_env = VecLowLevelAUV2DEnvV3Continuous(num_envs=self.num_envs, **env_kwargs)
        self.eval_env = LowLevelAUV2DEnvV3Continuous(**env_kwargs)

        self.obs_dim = self.train_env.observation_space.shape[0]
        self.act_dim = self.train_env.action_space.shape[0]

        print("2D V3 Continuous Environment initialized:")
        print(f"  obs_dim={self.obs_dim}, act_dim={self.act_dim}")
        print(f"  target range: [{self.args.target_distance_min}, {self.args.target_distance_max}] m")
        print(f"  reach_threshold={self.args.reach_threshold} m")
        print(f"  max_episode_seconds={self.args.max_episode_seconds}")
        print(f"  num_envs={self.num_envs}")

    def _init_network(self):
        training_config = self.config["training"]
        hidden = self.config.get("network", {}).get("feature_extractor", {}).get("hidden_sizes", [128, 64])
        self.model = GaussianActorCritic(self.obs_dim, self.act_dim, hidden=tuple(hidden)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(training_config.get("learning_rate", 3e-4)))

        self.gamma = float(training_config.get("gamma", 0.99))
        self.gae_lambda = float(training_config.get("gae_lambda", 0.95))
        self.clip_range = float(training_config.get("clip_range", 0.2))
        self.ent_coef = float(training_config.get("ent_coef", 0.05))
        self.vf_coef = float(training_config.get("vf_coef", 0.5))
        self.max_grad_norm = float(training_config.get("max_grad_norm", 0.5))
        self.n_steps = int(training_config.get("n_steps", 1024))
        self.batch_size = int(training_config.get("batch_size", 256))
        self.n_epochs = int(training_config.get("n_epochs", 4))

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Network parameters: {total_params:,}")

    def collect_rollouts(self):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
        obs, _ = self.train_env.reset()

        for _ in range(self.n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, logp, _ = self.model.get_action(obs_tensor, deterministic=False)
                value = self.model.get_value(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, infos = self.train_env.step(action_np)
            dones = np.logical_or(terminateds, truncateds)

            obs_buf.append(obs)
            act_buf.append(action_np)
            rew_buf.append(rewards)
            done_buf.append(dones.astype(np.float32))
            val_buf.append(value.cpu().numpy())
            logp_buf.append(logp.cpu().numpy())

            for i, info in enumerate(infos):
                if info.get("termination_reason") == "success":
                    self.success_count += 1
                if dones[i]:
                    self.episode_count += 1

            obs = next_obs

        with torch.no_grad():
            last_values = self.model.get_value(torch.tensor(obs, dtype=torch.float32, device=self.device)).cpu().numpy()

        rewards_arr = np.array(rew_buf)
        values_arr = np.array(val_buf)
        dones_arr = np.array(done_buf)

        advantages = np.zeros_like(rewards_arr, dtype=np.float32)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_values = last_values if t == self.n_steps - 1 else values_arr[t + 1]
            next_non_terminal = 1.0 - dones_arr[t]
            delta = rewards_arr[t] + self.gamma * next_values * next_non_terminal - values_arr[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values_arr

        return {
            "obs": np.array(obs_buf),
            "actions": np.array(act_buf),
            "logprobs": np.array(logp_buf),
            "advantages": advantages,
            "returns": returns,
            "mean_reward": float(np.mean(rewards_arr)),
        }

    def train_step(self, rollout_data: dict):
        obs = torch.tensor(rollout_data["obs"].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout_data["actions"].reshape(-1, self.act_dim), dtype=torch.float32, device=self.device)
        old_logprobs = torch.tensor(rollout_data["logprobs"].reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout_data["advantages"].reshape(-1), dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout_data["returns"].reshape(-1), dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.n_epochs):
            indices = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_adv = advantages[idx]
                batch_returns = returns[idx]

                new_logprobs, ent_each = self.model.evaluate_actions(batch_obs, batch_actions)
                entropy = ent_each.mean()
                values = self.model.get_value(batch_obs)

                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, batch_returns)
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += float(loss.item())

        return total_loss / max(1, self.n_epochs)

    def evaluate(self, num_episodes: int = 10):
        rewards = []
        success = 0
        final_errors = []

        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                with torch.no_grad():
                    x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, _, _ = self.model.get_action(x, deterministic=True)
                    action_np = action[0].cpu().numpy()

                obs, reward, term, trunc, info = self.eval_env.step(action_np)
                ep_reward += reward
                done = term or trunc
                if done:
                    final_errors.append(info.get("pos_error", float("nan")))
                    if info.get("termination_reason") == "success":
                        success += 1

            rewards.append(ep_reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(success / max(1, num_episodes)),
            "std_reward": float(np.std(rewards)),
            "mean_final_error": float(np.nanmean(final_errors)) if final_errors else float("nan"),
        }

    def train(self):
        total_timesteps = int(self.config["training"].get("total_timesteps", 2_000_000))
        timesteps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // timesteps_per_update

        print(f"Start training for {num_updates} updates ({total_timesteps} timesteps)")

        best_eval_reward = -1e9
        eval_freq = int(self.config["training"].get("eval_freq", 5000))
        save_freq = int(self.config["training"].get("save_freq", 20000))
        last_eval_mean_reward = np.nan
        last_eval_success_rate = np.nan

        for update in range(1, num_updates + 1):
            rollout_data = self.collect_rollouts()
            train_loss = self.train_step(rollout_data)

            current_steps = update * timesteps_per_update
            success_rate = self.success_count / max(1, self.episode_count)

            if update % 10 == 0:
                print(
                    f"Update {update}/{num_updates} | "
                    f"Steps {current_steps} | "
                    f"TrainRew {rollout_data['mean_reward']:.3f} | "
                    f"Loss {train_loss:.4f} | "
                    f"SuccRate {success_rate:.2%}",
                    flush=True,
                )

            if current_steps % eval_freq < timesteps_per_update:
                eval_metrics = self.evaluate(num_episodes=int(self.config["training"].get("eval_episodes", 10)))
                last_eval_mean_reward = float(eval_metrics["mean_reward"])
                last_eval_success_rate = float(eval_metrics["success_rate"])
                print(
                    f"  [Eval] steps={current_steps} reward={eval_metrics['mean_reward']:.3f} "
                    f"succ={eval_metrics['success_rate']:.2%} final_err={eval_metrics['mean_final_error']:.4f}",
                    flush=True,
                )
                if eval_metrics["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["mean_reward"]
                    self.save_checkpoint("best")

            if current_steps % save_freq < timesteps_per_update:
                self.save_checkpoint("latest")

            self.history.append({
                "update": int(update),
                "steps": int(current_steps),
                "train_mean_reward": float(rollout_data["mean_reward"]),
                "train_loss": float(train_loss),
                "train_success_rate": float(success_rate),
                "eval_mean_reward": float(last_eval_mean_reward),
                "eval_success_rate": float(last_eval_success_rate),
            })

            if update % 20 == 0:
                self.save_training_history()

        self.save_checkpoint("final")
        self.save_training_history()
        self.plot_training_curves()
        print("Training finished.")

    def save_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        torch.save({
            "model": self.model.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "config": self.config,
            "experiment_name": self.experiment_name,
        }, path)
        print(f"Saved checkpoint: {path}")

    def save_training_history(self):
        if not self.history:
            return
        csv_path = os.path.join(self.out_dir, "training_metrics.csv")
        json_path = os.path.join(self.out_dir, "training_metrics.json")
        fields = ["update", "steps", "train_mean_reward", "train_loss",
                   "train_success_rate", "eval_mean_reward", "eval_success_rate"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def plot_training_curves(self):
        if not self.history:
            return
        steps = np.array([h["steps"] for h in self.history], dtype=np.float64)
        train_rew = np.array([h["train_mean_reward"] for h in self.history], dtype=np.float64)
        train_loss = np.array([h["train_loss"] for h in self.history], dtype=np.float64)
        train_sr = np.array([h["train_success_rate"] for h in self.history], dtype=np.float64)
        eval_rew = np.array([h["eval_mean_reward"] for h in self.history], dtype=np.float64)
        eval_sr = np.array([h["eval_success_rate"] for h in self.history], dtype=np.float64)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        ax = axes[0, 0]
        ax.plot(steps, train_rew, lw=1.8, color="tab:blue", label="train (per-step avg)")
        ax.set_ylabel("train reward (per-step)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        valid = ~np.isnan(eval_rew)
        if np.any(valid):
            ax2 = ax.twinx()
            ax2.plot(steps[valid], eval_rew[valid], lw=1.8, color="tab:red", label="eval (episode total)")
            ax2.set_ylabel("eval reward (episode)", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        else:
            ax.legend()
        ax.set_title("Reward Curve")
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.25)

        ax = axes[0, 1]
        ax.plot(steps, train_loss, lw=1.8, color="tab:orange")
        ax.set_title("Loss Curve")
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.25)

        ax = axes[1, 0]
        ax.plot(steps, train_sr, lw=1.8, label="train_success_rate")
        valid = ~np.isnan(eval_sr)
        if np.any(valid):
            ax.plot(steps[valid], eval_sr[valid], lw=1.8, label="eval_success_rate")
        ax.set_ylim(0, 1)
        ax.set_title("Success Rate Curve")
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.25)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(np.arange(1, len(train_rew) + 1), train_rew, lw=1.8, label="reward by update")
        ax.set_title("Reward vs Update")
        ax.set_xlabel("update")
        ax.grid(True, alpha=0.25)
        ax.legend()

        plt.tight_layout()
        out_path = os.path.join(self.out_dir, "training_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training curves: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train 2D V3 continuous (small target)")
    parser.add_argument("--config", type=str, default="configs/low_level_config_2d.yaml")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # V3 环境参数
    parser.add_argument("--target-distance-max", type=float, default=1.0)
    parser.add_argument("--target-distance-min", type=float, default=0.3)
    parser.add_argument("--reach-threshold", type=float, default=0.10)
    parser.add_argument("--max-episode-seconds", type=float, default=30.0)
    # 奖励权重
    parser.add_argument("--reward-reach", type=float, default=100.0)
    parser.add_argument("--reward-speed-bonus", type=float, default=50.0)
    parser.add_argument("--reward-improve-scale", type=float, default=50.0)
    parser.add_argument("--reward-heading-scale", type=float, default=0.0)
    parser.add_argument("--reward-heading-terminal", type=float, default=0.0,
                        help="到达时航向缩放系数: 0=不缩放, 0.5=最差只拿50%%到达奖")
    parser.add_argument("--reward-vel-align-scale", type=float, default=0.0,
                        help="速度方向对齐奖励系数 (鼓励 surge 朝目标)")
    parser.add_argument("--reward-time-penalty", type=float, default=-0.2)
    # 初始速度随机化
    parser.add_argument("--init-velocity-range", type=float, default=0.0,
                        help="初始线速度随机范围 (m/s)，0=静止起步")
    parser.add_argument("--init-yaw-rate-range", type=float, default=0.0,
                        help="初始偏航角速度随机范围 (rad/s)，0=自动取线速度一半")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = int(args.total_timesteps)
    if args.n_envs is not None:
        config["training"]["n_envs"] = int(args.n_envs)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = Continuous2DTrainerV3(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
