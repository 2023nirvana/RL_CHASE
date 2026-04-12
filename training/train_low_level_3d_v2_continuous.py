"""
3D V2 连续动作低层训练脚本
============================

将当前 3D V2 低层从离散动作扩展到连续动作版本。
"""

import os
import sys
import argparse
import yaml
import csv
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_3d_v2_continuous import (
    LowLevelAUV3DEnvV2Continuous,
    VecLowLevelAUV3DEnvV2Continuous,
)


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

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def get_dist(self, x):
        mu = self.mu_head(self.actor_body(x))
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def get_action(self, x, deterministic=False):
        dist = self.get_dist(x)
        z = dist.mean if deterministic else dist.rsample()
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


def _safe_number(x):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return None
        return float(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


class Continuous3DTrainerV2:
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.experiment_name = args.experiment_name or f"3d_v2_cont_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_environments()
        self._init_network()
        self._maybe_resume()

        self.success_count = 0
        self.episode_count = 0
        self.history = []

    def _maybe_resume(self):
        ckpt_path = None
        if self.args.resume_ckpt is not None:
            ckpt_path = self.args.resume_ckpt
        elif self.args.resume_from is not None:
            candidate = os.path.join(self.checkpoint_dir, f"{self.args.resume_from}.pth")
            if os.path.exists(candidate):
                ckpt_path = candidate

        if ckpt_path is None:
            return

        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "model" not in ckpt:
            raise ValueError(f"Invalid checkpoint (missing 'model'): {ckpt_path}")
        self.model.load_state_dict(ckpt["model"], strict=False)
        print(f"Resumed model weights from: {ckpt_path}")

    def _init_environments(self):
        tr_cfg = self.config["training"]
        self.num_envs = int(tr_cfg.get("n_envs", 4))

        kwargs = dict(
            target_distance=self.args.target_distance,
            z_range=(self.args.z_min, self.args.z_max),
            success_radius=self.args.success_radius,
            z_coef=self.args.z_coef,
            z_boost_when_xy_close=self.args.z_boost_when_xy_close,
            xy_close_threshold=self.args.xy_close_threshold,
            relaxed_heading_when_xy_close=not self.args.disable_relaxed_heading_when_xy_close,
            max_steps=self.args.max_steps,
            heave_step_override=self.args.heave_step,
            yaw_delta_override=self.args.yaw_delta,
            yaw_rate_penalty_coef=self.args.yaw_rate_penalty_coef,
            action_smooth_coef=self.args.action_smooth_coef,
            z_lock_coef=self.args.z_lock_coef,
            improvement_coef=self.args.improvement_coef,
            improvement_clip=self.args.improvement_clip,
            heading_coef=self.args.heading_coef,
            direction_coef=self.args.direction_coef,
            heading_bonus_coef=self.args.heading_bonus_coef,
            direction_use_closing_speed=not self.args.disable_closing_speed,
            yaw_action_penalty_coef=self.args.yaw_action_penalty_coef,
            yaw_action_flip_penalty_coef=self.args.yaw_action_flip_penalty_coef,
            enable_pitch_control=not self.args.disable_pitch_control,
            pitch_delta_override=self.args.pitch_delta,
            pitch_limit_deg=self.args.pitch_limit_deg,
            min_target_z_gap=self.args.min_target_z_gap,
            z_progress_coef=self.args.z_progress_coef,
            z_progress_clip=self.args.z_progress_clip,
            xy_gate_strictness=self.args.xy_gate_strictness,
            spiral_penalty_coef=self.args.spiral_penalty_coef,
            enable_domain_randomization=False,
        )

        self.train_env = VecLowLevelAUV3DEnvV2Continuous(num_envs=self.num_envs, **kwargs)
        self.eval_env = LowLevelAUV3DEnvV2Continuous(**kwargs)

        self.obs_dim = self.train_env.observation_space.shape[0]
        self.act_dim = self.train_env.action_space.shape[0]

        print("3D V2 Continuous Environment initialized:")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.act_dim} (continuous)")
        print(f"  Target distance: {self.args.target_distance}m")
        print(f"  Z range: [{self.args.z_min}, {self.args.z_max}]")
        print(f"  Success radius: {self.args.success_radius}")
        print(f"  Heave step: {self.eval_env.heave_step}")
        print(f"  Yaw delta: {self.eval_env.yaw_delta}")
        print(f"  Num envs: {self.num_envs}")

    def _apply_distance_curriculum(self, current_steps: int, total_timesteps: int):
        if self.args.curriculum_start_distance is None:
            return
        start_d = float(self.args.curriculum_start_distance)
        end_d = float(self.args.target_distance)
        curriculum_steps = int(self.args.curriculum_steps) if self.args.curriculum_steps is not None else int(total_timesteps * 0.6)
        curriculum_steps = max(1, curriculum_steps)
        alpha = min(1.0, max(0.0, current_steps / curriculum_steps))
        current_d = start_d + (end_d - start_d) * alpha

        for env in self.train_env.envs:
            env.target_distance = float(current_d)
        self.eval_env.target_distance = float(current_d)

    def _init_network(self):
        tr_cfg = self.config["training"]
        hidden_sizes = self.config.get("network", {}).get("feature_extractor", {}).get("hidden_sizes", [128, 64])
        self.model = GaussianActorCritic(self.obs_dim, self.act_dim, hidden=tuple(hidden_sizes)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(tr_cfg.get("learning_rate", 3e-4)))

        self.gamma = float(tr_cfg.get("gamma", 0.99))
        self.gae_lambda = float(tr_cfg.get("gae_lambda", 0.95))
        self.clip_range = float(tr_cfg.get("clip_range", 0.2))
        self.ent_coef = float(tr_cfg.get("ent_coef", 0.05))
        self.vf_coef = float(tr_cfg.get("vf_coef", 0.5))
        self.max_grad_norm = float(tr_cfg.get("max_grad_norm", 0.5))
        self.n_steps = int(tr_cfg.get("n_steps", 1024))
        self.batch_size = int(tr_cfg.get("batch_size", 256))
        self.n_epochs = int(tr_cfg.get("n_epochs", 4))

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

        rewards = np.array(rew_buf)
        values = np.array(val_buf)
        dones = np.array(done_buf)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_values = last_values if t == self.n_steps - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        return {
            "obs": np.array(obs_buf),
            "actions": np.array(act_buf),
            "logprobs": np.array(logp_buf),
            "advantages": advantages,
            "returns": returns,
            "mean_reward": float(np.mean(rewards)),
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
        term_counts = {"success": 0, "timeout": 0, "out_of_bounds": 0, "other": 0}

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
                    reason = info.get("termination_reason", "other")
                    if reason == "success":
                        success += 1
                        term_counts["success"] += 1
                    elif reason == "timeout":
                        term_counts["timeout"] += 1
                    elif reason == "out_of_bounds":
                        term_counts["out_of_bounds"] += 1
                    else:
                        term_counts["other"] += 1

            rewards.append(ep_reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(success / max(1, num_episodes)),
            "std_reward": float(np.std(rewards)),
            "term_counts": term_counts,
        }

    def save_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        torch.save(
            {
                "model": self.model.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "config": self.config,
                "experiment_name": self.experiment_name,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    def save_training_history(self):
        if len(self.history) == 0:
            return

        csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        json_path = os.path.join(self.log_dir, "training_metrics.json")
        fields = [
            "update",
            "steps",
            "train_mean_reward",
            "train_loss",
            "train_success_rate",
            "eval_mean_reward",
            "eval_success_rate",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.history:
                writer.writerow({k: _safe_number(row.get(k)) for k in fields})

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([{k: _safe_number(v) for k, v in row.items()} for row in self.history], f, ensure_ascii=False, indent=2)

    def plot_training_curves(self):
        if len(self.history) == 0:
            return

        steps = np.array([h["steps"] for h in self.history], dtype=np.float64)
        train_rew = np.array([h["train_mean_reward"] for h in self.history], dtype=np.float64)
        train_loss = np.array([h["train_loss"] for h in self.history], dtype=np.float64)
        train_sr = np.array([h["train_success_rate"] for h in self.history], dtype=np.float64)
        eval_rew = np.array([h["eval_mean_reward"] if h["eval_mean_reward"] is not None else np.nan for h in self.history], dtype=np.float64)
        eval_sr = np.array([h["eval_success_rate"] if h["eval_success_rate"] is not None else np.nan for h in self.history], dtype=np.float64)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        ax = axes[0, 0]
        ax.plot(steps, train_rew, lw=1.8, label="train_mean_reward")
        valid = ~np.isnan(eval_rew)
        if np.any(valid):
            ax.plot(steps[valid], eval_rew[valid], lw=1.8, label="eval_mean_reward")
        ax.set_title("Reward Curve")
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.25)
        ax.legend()

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
        out_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training curves: {out_path}")

    def train(self):
        total_timesteps = int(self.config["training"].get("total_timesteps", 2_000_000))
        timesteps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // timesteps_per_update

        print(f"Start training for {num_updates} updates ({total_timesteps} timesteps)")

        best_eval_reward = -1e9
        eval_freq = int(self.config["training"].get("eval_freq", 5000))
        save_freq = int(self.config["training"].get("save_freq", 20000))
        last_eval_mean_reward = None
        last_eval_success_rate = None

        for update in range(1, num_updates + 1):
            rollout_data = self.collect_rollouts()
            train_loss = self.train_step(rollout_data)

            current_steps = update * timesteps_per_update
            self._apply_distance_curriculum(current_steps, total_timesteps)
            success_rate = self.success_count / max(1, self.episode_count)

            if update % 10 == 0:
                print(
                    f"Update {update}/{num_updates} | Steps {current_steps} | "
                    f"TrainRew {rollout_data['mean_reward']:.3f} | Loss {train_loss:.4f} | Succ {success_rate:.2%}",
                    flush=True,
                )

            if current_steps % eval_freq < timesteps_per_update:
                eval_metrics = self.evaluate(num_episodes=int(self.config["training"].get("eval_episodes", 10)))
                last_eval_mean_reward = float(eval_metrics["mean_reward"])
                last_eval_success_rate = float(eval_metrics["success_rate"])
                print(
                    f"  [Eval] steps={current_steps} mean_reward={eval_metrics['mean_reward']:.3f} "
                    f"success_rate={eval_metrics['success_rate']:.2%}",
                    flush=True,
                )
                print(f"  [EvalTerm] {eval_metrics.get('term_counts', {})}", flush=True)

                if eval_metrics["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["mean_reward"]
                    self.save_checkpoint("best")

            if current_steps % save_freq < timesteps_per_update:
                self.save_checkpoint("latest")

            self.history.append(
                {
                    "update": int(update),
                    "steps": int(current_steps),
                    "train_mean_reward": float(rollout_data["mean_reward"]),
                    "train_loss": float(train_loss),
                    "train_success_rate": float(success_rate),
                    "eval_mean_reward": None if last_eval_mean_reward is None else float(last_eval_mean_reward),
                    "eval_success_rate": None if last_eval_success_rate is None else float(last_eval_success_rate),
                }
            )

            if update % 20 == 0:
                self.save_training_history()

        self.save_checkpoint("final")
        self.save_training_history()
        self.plot_training_curves()
        print("Training finished.")


def main():
    parser = argparse.ArgumentParser(description="Train 3D V2 continuous low-level policy")
    parser.add_argument("--config", type=str, default="configs/low_level_config_2d.yaml")
    parser.add_argument("--target-distance", type=float, default=12.0)
    parser.add_argument("--curriculum-start-distance", type=float, default=None)
    parser.add_argument("--curriculum-steps", type=int, default=None)
    parser.add_argument("--z-min", type=float, default=2.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--z-coef", type=float, default=1.0)
    parser.add_argument("--z-boost-when-xy-close", type=float, default=1.5)
    parser.add_argument("--xy-close-threshold", type=float, default=1.0)
    parser.add_argument("--disable-relaxed-heading-when-xy-close", action="store_true")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--heave-step", type=float, default=None)
    parser.add_argument("--yaw-delta", type=float, default=None)
    parser.add_argument("--yaw-rate-penalty-coef", type=float, default=0.03)
    parser.add_argument("--action-smooth-coef", type=float, default=0.02)
    parser.add_argument("--z-lock-coef", type=float, default=0.6)
    parser.add_argument("--improvement-coef", type=float, default=8.0)
    parser.add_argument("--improvement-clip", type=float, default=0.12)
    parser.add_argument("--heading-coef", type=float, default=0.55)
    parser.add_argument("--direction-coef", type=float, default=0.45)
    parser.add_argument("--heading-bonus-coef", type=float, default=0.08)
    parser.add_argument("--disable-closing-speed", action="store_true")
    parser.add_argument("--yaw-action-penalty-coef", type=float, default=0.02)
    parser.add_argument("--yaw-action-flip-penalty-coef", type=float, default=0.03)
    parser.add_argument("--disable-pitch-control", action="store_true")
    parser.add_argument("--pitch-delta", type=float, default=None)
    parser.add_argument("--pitch-limit-deg", type=float, default=25.0)
    parser.add_argument("--min-target-z-gap", type=float, default=1.0)
    parser.add_argument("--z-progress-coef", type=float, default=1.2)
    parser.add_argument("--z-progress-clip", type=float, default=0.08)
    parser.add_argument("--xy-gate-strictness", type=float, default=1.0)
    parser.add_argument("--spiral-penalty-coef", type=float, default=0.25)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None, choices=["best", "latest", "final"])
    parser.add_argument("--resume-ckpt", type=str, default=None)
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

    trainer = Continuous3DTrainerV2(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
