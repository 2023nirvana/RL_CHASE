"""
2D 追逃 HRL 训练：固定低层策略，高层学习子目标
===================================================

高层动作：0=直接追逃脱者位置，1~8=以固定半径生成子目标方向。
低层策略：使用 LowLevelAUV2DEnvV2 训练好的 actor（8维观测，6动作）。
"""
import os
import sys
import argparse
import time
from typing import Dict, Optional, Tuple, List, Any

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_pursuit_evasion import (
    LowLevelAUV2DPursuitEvasionEnv,
)


def _build_actor(obs_dim: int, num_actions: int, hidden_sizes=(128, 64)) -> nn.Module:
    layers = []
    prev_dim = obs_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, num_actions))
    return nn.Sequential(*layers)


class _GaussianActorCritic(nn.Module):
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

    def get_action(self, x, deterministic=False):
        mu, _ = self.forward(x)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample()
        a = torch.tanh(z)
        return a


def _get_low_level_obs(
    pursuer_pos: np.ndarray,
    pursuer_vel: np.ndarray,
    pursuer_yaw: float,
    target_pos: np.ndarray,
) -> np.ndarray:
    pos_error_world = target_pos - pursuer_pos
    distance = np.linalg.norm(pos_error_world)
    target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
    heading_error = np.arctan2(
        np.sin(target_angle - pursuer_yaw),
        np.cos(target_angle - pursuer_yaw),
    )
    cos_yaw = np.cos(pursuer_yaw)
    sin_yaw = np.sin(pursuer_yaw)
    pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
    pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
    obs = np.array(
        [
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            pursuer_vel[0] / 2.0,
            pursuer_vel[1] / 2.0,
            0.0,  # yaw_rate unavailable in low-level obs construction
            distance / 5.0,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -10.0, 10.0)


def _continuous_to_discrete_action(action_cont: np.ndarray) -> int:
    surge = float(action_cont[0])
    sway = float(action_cont[1])
    yaw = float(action_cont[2])
    mags = np.array([abs(surge), abs(sway), abs(yaw)], dtype=np.float64)
    axis = int(np.argmax(mags))
    if axis == 0:
        return 0 if surge >= 0 else 1
    if axis == 1:
        return 2 if sway >= 0 else 3
    return 4 if yaw >= 0 else 5


def _high_action_to_waypoint(
    action: int,
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    waypoint_distance: float,
    world_size: float,
) -> np.ndarray:
    if action == 0:
        return evader_pos.copy()
    angle = (action - 1) * (2 * np.pi / 8)
    offset = waypoint_distance * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    target = pursuer_pos + offset
    return np.clip(target, -world_size, world_size)


class HRLTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.experiment_name = args.experiment_name or f"pe2d_hrl_{int(time.time())}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_envs()
        self._init_low_level()
        self._init_high_level()

        self.episode_count = 0
        self.success_count = 0
        self.history: Dict[str, List[Any]] = {
            "timesteps": [],
            "mean_reward": [],
            "eval_reward": [],
            "eval_success_rate": [],
            "train_success_cumulative": [],
            "loss": [],
        }

    def _init_envs(self):
        self.n_envs = self.args.n_envs
        env_kwargs = dict(
            init_distance_range=tuple(self.args.init_distance_range),
            world_size=self.args.world_size,
            catch_radius=self.args.catch_radius,
            max_episode_seconds=self.args.max_episode_seconds,
            curriculum_stage=self.args.curriculum_stage,
        )
        self.envs = [LowLevelAUV2DPursuitEvasionEnv(**env_kwargs) for _ in range(self.n_envs)]
        self.eval_env = LowLevelAUV2DPursuitEvasionEnv(**env_kwargs)
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.high_actions = 9
        print("HRL 2D Pursuit-Evasion:")
        print(f"  High-level obs dim: {self.obs_dim}")
        print(f"  High-level actions: {self.high_actions}")
        print(f"  Low-level K steps: {self.args.low_level_steps}")

    def _init_low_level(self):
        ckpt = torch.load(self.args.low_level_ckpt, map_location=self.device, weights_only=False)

        ll_type = self.args.low_level_policy_type
        if ll_type == "auto":
            if "actor" in ckpt:
                ll_type = "discrete"
            elif "model" in ckpt:
                ll_type = "continuous"
            else:
                raise ValueError("Cannot infer low-level policy type from checkpoint keys")

        self.low_level_policy_type = ll_type
        if ll_type == "discrete":
            self.low_actor = _build_actor(8, 6, hidden_sizes=tuple(self.args.low_level_hidden))
            self.low_actor.load_state_dict(ckpt["actor"])
            self.low_actor = self.low_actor.to(self.device)
            self.low_actor.eval()
            self.low_cont_model = None
            print("Low-level policy type: discrete")
        elif ll_type == "continuous":
            cont_hidden = tuple(self.args.low_level_cont_hidden)
            self.low_cont_model = _GaussianActorCritic(8, 3, hidden=cont_hidden).to(self.device)
            self.low_cont_model.load_state_dict(ckpt["model"])
            self.low_cont_model.eval()
            self.low_actor = None
            print("Low-level policy type: continuous (adapted to discrete env actions)")
        else:
            raise ValueError(f"Unsupported low-level policy type: {ll_type}")

    def _init_high_level(self):
        self.high_actor = _build_actor(self.obs_dim, self.high_actions, hidden_sizes=tuple(self.args.high_level_hidden)).to(self.device)
        self.high_critic = _build_actor(self.obs_dim, 1, hidden_sizes=tuple(self.args.high_level_hidden)).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.high_actor.parameters()) + list(self.high_critic.parameters()),
            lr=self.args.lr,
        )
        self.gamma = self.args.gamma
        self.gae_lambda = self.args.gae_lambda
        self.clip_range = self.args.clip_range
        self.ent_coef = self.args.ent_coef
        self.vf_coef = self.args.vf_coef
        self.max_grad_norm = self.args.max_grad_norm
        self.n_steps = self.args.n_steps
        self.batch_size = self.args.batch_size
        self.n_epochs = self.args.n_epochs

    def _high_policy(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            logits = self.high_actor(obs_t)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                actions = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
        return actions.cpu().numpy()

    def _low_policy(self, obs: np.ndarray, deterministic: bool = True):
        if self.low_level_policy_type == "discrete":
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                logits = self.low_actor(obs_t)
                probs = torch.softmax(logits, dim=-1)
                if deterministic:
                    actions = torch.argmax(probs, dim=-1)
                else:
                    dist = torch.distributions.Categorical(probs)
                    actions = dist.sample()
            return actions.cpu().numpy()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_cont = self.low_cont_model.get_action(obs_t, deterministic=deterministic).cpu().numpy()
        mapped = np.array([_continuous_to_discrete_action(a) for a in action_cont], dtype=np.int64)
        return mapped

    def collect_rollouts(self):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []

        obs_list = [env.reset()[0] for env in self.envs]
        obs = np.stack(obs_list)

        for _ in range(self.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.high_actor(obs_t)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                values = self.high_critic(obs_t).squeeze(-1)

            actions_np = actions.cpu().numpy()
            logprobs_np = logprobs.cpu().numpy()
            values_np = values.cpu().numpy()

            next_obs_list = []
            rewards = np.zeros(self.n_envs, dtype=np.float32)
            dones = np.zeros(self.n_envs, dtype=bool)

            for i, env in enumerate(self.envs):
                pursuer_pos = env.inner.dynamics.eta[:2].copy()
                pursuer_vel = env.inner.dynamics.nu[:2].copy()
                pursuer_yaw = float(env.inner.dynamics.eta[5])
                evader_pos = env.evader_pos.copy()

                subgoal = _high_action_to_waypoint(
                    int(actions_np[i]),
                    pursuer_pos,
                    evader_pos,
                    self.args.waypoint_distance,
                    env.world_size,
                )

                for _ in range(self.args.low_level_steps):
                    low_obs = _get_low_level_obs(
                        pursuer_pos,
                        pursuer_vel,
                        pursuer_yaw,
                        subgoal,
                    )
                    low_action = int(self._low_policy(low_obs[np.newaxis, :], deterministic=True)[0])
                    obs_i, reward, term, trunc, info = env.step(low_action)
                    rewards[i] += reward
                    pursuer_pos = env.inner.dynamics.eta[:2].copy()
                    pursuer_vel = env.inner.dynamics.nu[:2].copy()
                    pursuer_yaw = float(env.inner.dynamics.eta[5])
                    if term or trunc:
                        dones[i] = True
                        if info.get("termination_reason") == "success":
                            self.success_count += 1
                        self.episode_count += 1
                        obs_i, _ = env.reset()
                        break
                next_obs_list.append(obs_i)

            obs_buf.append(obs)
            act_buf.append(actions_np)
            rew_buf.append(rewards)
            done_buf.append(dones)
            val_buf.append(values_np)
            logp_buf.append(logprobs_np)
            obs = np.stack(next_obs_list)

        with torch.no_grad():
            last_values = self.high_critic(
                torch.tensor(obs, dtype=torch.float32, device=self.device)
            ).squeeze(-1).cpu().numpy()

        advantages, returns = self._compute_gae(
            np.array(rew_buf),
            np.array(val_buf),
            np.array(done_buf),
            last_values,
        )

        return {
            "obs": np.array(obs_buf),
            "actions": np.array(act_buf),
            "logprobs": np.array(logp_buf),
            "advantages": advantages,
            "returns": returns,
            "mean_reward": float(np.mean(rew_buf)),
        }

    def _compute_gae(self, rewards, values, dones, last_values):
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_values = last_values if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        returns = advantages + values
        return advantages, returns

    def train_step(self, rollout_data: Dict):
        obs = torch.tensor(rollout_data["obs"].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout_data["actions"].reshape(-1), dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(rollout_data["logprobs"].reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout_data["advantages"].reshape(-1), dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout_data["returns"].reshape(-1), dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss = 0.0

        for _ in range(self.n_epochs):
            indices = torch.randperm(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                logits = self.high_actor(batch_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                values = self.high_critic(batch_obs).squeeze(-1)

                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, batch_returns)
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.high_actor.parameters()) + list(self.high_critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / (self.n_epochs * (len(obs) // self.batch_size + 1))

    def evaluate(self, n_episodes: int = 10):
        successes = 0
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                high_action = self._high_policy(obs[np.newaxis, :], deterministic=True)[0]
                pursuer_pos = self.eval_env.inner.dynamics.eta[:2].copy()
                pursuer_vel = self.eval_env.inner.dynamics.nu[:2].copy()
                pursuer_yaw = float(self.eval_env.inner.dynamics.eta[5])
                evader_pos = self.eval_env.evader_pos.copy()
                subgoal = _high_action_to_waypoint(
                    int(high_action),
                    pursuer_pos,
                    evader_pos,
                    self.args.waypoint_distance,
                    self.eval_env.world_size,
                )
                for _ in range(self.args.low_level_steps):
                    low_obs = _get_low_level_obs(
                        pursuer_pos,
                        pursuer_vel,
                        pursuer_yaw,
                        subgoal,
                    )
                    low_action = int(self._low_policy(low_obs[np.newaxis, :], deterministic=True)[0])
                    obs, reward, term, trunc, info = self.eval_env.step(low_action)
                    ep_reward += reward
                    pursuer_pos = self.eval_env.inner.dynamics.eta[:2].copy()
                    pursuer_vel = self.eval_env.inner.dynamics.nu[:2].copy()
                    pursuer_yaw = float(self.eval_env.inner.dynamics.eta[5])
                    done = term or trunc
                    if done:
                        if info.get("termination_reason") == "success":
                            successes += 1
                        break
            rewards.append(ep_reward)
        return {
            "success_rate": successes / max(1, n_episodes),
            "mean_reward": float(np.mean(rewards)),
        }

    def save(self, path: str):
        torch.save(
            {
                "high_actor": self.high_actor.state_dict(),
                "high_critic": self.high_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def _save_history_and_plots(self):
        json_path = os.path.join(self.log_dir, "training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        ts = self.history["timesteps"]
        if not ts:
            return
        plots_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        ax.plot(ts, self.history["mean_reward"], label="Train (rollout mean)", alpha=0.8)
        ax.plot(ts, self.history["eval_reward"], label="Eval mean", alpha=0.8)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(ts, np.array(self.history["eval_success_rate"]) * 100, label="Eval Success %", alpha=0.8)
        ax.plot(ts, np.array(self.history["train_success_cumulative"]) * 100, label="Train Success %", alpha=0.8)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(ts, self.history["loss"], color="C2", alpha=0.8)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)

        axes[1, 1].axis("off")
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Training curves saved to {plot_path}", flush=True)

    def train(self):
        total_timesteps = self.args.total_timesteps
        steps_per_update = self.n_steps * self.n_envs
        num_updates = total_timesteps // steps_per_update

        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Steps per update: {steps_per_update:,}")
        print(f"Number of updates: {num_updates:,}")

        timesteps = 0
        best_success_rate = 0.0
        start_time = time.time()

        for update in range(1, num_updates + 1):
            rollout_data = self.collect_rollouts()
            timesteps += steps_per_update
            loss = self.train_step(rollout_data)

            if timesteps % self.args.eval_freq < steps_per_update:
                eval_info = self.evaluate(n_episodes=self.args.eval_episodes)
                elapsed = time.time() - start_time
                fps = timesteps / max(elapsed, 1e-6)
                train_success = self.success_count / max(1, self.episode_count)
                print(f"\nUpdate {update}/{num_updates} | Timesteps: {timesteps:,}")
                print(f"  FPS: {fps:.1f}")
                print(f"  Mean Reward: {rollout_data['mean_reward']:.3f}")
                print(f"  Eval Reward: {eval_info['mean_reward']:.3f}")
                print(f"  Eval Success Rate: {eval_info['success_rate']:.1%}")
                print(f"  Train Success (cumulative): {train_success:.1%}")
                print(f"  Loss: {loss:.4f}")

                if eval_info["success_rate"] > best_success_rate:
                    best_success_rate = eval_info["success_rate"]
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best model! Success rate: {best_success_rate:.1%}")

                self.history["timesteps"].append(timesteps)
                self.history["mean_reward"].append(float(rollout_data["mean_reward"]))
                self.history["eval_reward"].append(float(eval_info["mean_reward"]))
                self.history["eval_success_rate"].append(float(eval_info["success_rate"]))
                self.history["train_success_cumulative"].append(float(train_success))
                self.history["loss"].append(float(loss))
                self._save_history_and_plots()

            if timesteps % self.args.save_freq < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))

        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"Training completed. Best success rate: {best_success_rate:.1%}")


def _parse_init_distance(s: str) -> Tuple[float, float]:
    if not s:
        return (2.0, 6.0)
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        return (2.0, 6.0)
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return (2.0, 6.0)


def main():
    parser = argparse.ArgumentParser(description="2D PE HRL (fixed low-level, high-level PPO)")
    parser.add_argument("--low-level-ckpt", type=str, required=True, help="Checkpoint from train_low_level_2d_v2")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-distance-range", type=str, default="2,6")
    parser.add_argument("--world-size", type=float, default=10.0)
    parser.add_argument("--catch-radius", type=float, default=0.5)
    parser.add_argument("--max-episode-seconds", type=float, default=90.0)
    parser.add_argument("--curriculum-stage", type=int, default=0)
    parser.add_argument("--low-level-steps", type=int, default=5)
    parser.add_argument("--waypoint-distance", type=float, default=2.0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--low-level-hidden", type=int, nargs=2, default=[128, 64])
    parser.add_argument("--low-level-policy-type", type=str, default="auto", choices=["auto", "discrete", "continuous"])
    parser.add_argument("--low-level-cont-hidden", type=int, nargs=2, default=[128, 64])
    parser.add_argument("--high-level-hidden", type=int, nargs=3, default=[256, 128, 64])

    args = parser.parse_args()
    args.init_distance_range = _parse_init_distance(args.init_distance_range)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = HRLTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
