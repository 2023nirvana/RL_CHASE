"""
2D 追逃 HRL（连续高层 + 固定连续低层）
=========================================

设计目标：
- 高层策略（PPO, Gaussian+tanh）输出连续子目标偏移 action_h ∈ [-1,1]^2
- 低层策略固定为已训练好的连续目标到达策略（3维连续动作）
- 连接方式：每次高层决策后，低层在 K 步内追踪该子目标，环境中的逃脱者持续运动

说明：
- 本脚本强调“高层任务与低层技能解耦”：
  高层负责策略规划（子目标），低层负责控制执行（到达子目标）。
- 追逃奖励与终止按“与逃脱者关系”计算，不用子目标距离作为任务奖励。
"""

import os
import sys
import time
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

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

from core.environment.low_level_env_2d_pursuit_evasion import LowLevelAUV2DPursuitEvasionEnv


def _mlp(in_dim: int, out_dim: int, hidden: Tuple[int, ...]) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GaussianActorCritic(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 64)):
        super().__init__()
        self.actor_body = _mlp(obs_dim, hidden[-1], hidden[:-1]) if len(hidden) > 1 else nn.Identity()
        actor_in = hidden[-1] if len(hidden) > 0 else obs_dim
        if len(hidden) == 1:
            self.actor_body = nn.Sequential(nn.Linear(obs_dim, hidden[0]), nn.ReLU())
        elif len(hidden) > 1:
            self.actor_body = _mlp(obs_dim, hidden[-1], hidden[:-1])
            self.actor_body.append(nn.ReLU())
        self.mu_head = nn.Linear(actor_in, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        critic_layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def forward(self, x: torch.Tensor):
        h = self.actor_body(x)
        mu = self.mu_head(h)
        v = self.critic(x).squeeze(-1)
        return mu, v

    def get_dist(self, x: torch.Tensor):
        mu, _ = self.forward(x)
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        dist = self.get_dist(x)
        z = dist.mean if deterministic else dist.rsample()
        a = torch.tanh(z)
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return a, logp, ent

    def evaluate_actions(self, x: torch.Tensor, a: torch.Tensor):
        dist = self.get_dist(x)
        z = self._atanh(a)
        logp = (dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return logp, ent

    def get_value(self, x: torch.Tensor):
        return self.critic(x).squeeze(-1)


class LowLevelContinuousPolicy(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def __init__(self, obs_dim: int = 8, act_dim: int = 3, hidden=(128, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def get_action(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mu = self.mu_head(self.actor_body(x))
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        z = dist.mean if deterministic else dist.rsample()
        return torch.tanh(z)


def _infer_low_actor_hidden_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
    hidden = []
    idx = 0
    while True:
        k = f"actor_body.{idx}.weight"
        if k not in state_dict:
            break
        w = state_dict[k]
        hidden.append(int(w.shape[0]))
        idx += 2
    if len(hidden) == 0:
        raise ValueError("Cannot infer low-level actor hidden sizes from checkpoint")
    return tuple(hidden)


def _get_low_level_obs(pursuer_pos: np.ndarray, pursuer_vel: np.ndarray, pursuer_yaw: float,
                       pursuer_yaw_rate: float, target_pos: np.ndarray) -> np.ndarray:
    pos_error_world = target_pos - pursuer_pos
    distance = np.linalg.norm(pos_error_world)
    target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
    heading_error = np.arctan2(np.sin(target_angle - pursuer_yaw), np.cos(target_angle - pursuer_yaw))
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
            pursuer_yaw_rate / 2.0,
            distance / 5.0,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -10.0, 10.0)


def _get_high_level_obs(env: LowLevelAUV2DPursuitEvasionEnv) -> np.ndarray:
    eta = env.inner.dynamics.eta
    nu = env.inner.dynamics.nu
    pos_error_world = env.evader_pos - eta[:2]
    distance = np.linalg.norm(pos_error_world)
    yaw = eta[5]
    target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
    heading_error = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
    pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]

    pursuer_vel = nu[:2]
    evader_speed = np.linalg.norm(env.evader_vel)
    evader_heading = env.evader_heading
    if evader_speed > 0.1:
        evader_heading = np.arctan2(env.evader_vel[1], env.evader_vel[0])

    los_norm = np.linalg.norm(pos_error_world)
    if los_norm < 1e-6 or np.linalg.norm(pursuer_vel) < 1e-6:
        approach_angle = np.pi / 2
    else:
        pv_dir = pursuer_vel / (np.linalg.norm(pursuer_vel) + 1e-8)
        los_dir = pos_error_world / (los_norm + 1e-8)
        dot = float(np.clip(np.dot(pv_dir, los_dir), -1.0, 1.0))
        approach_angle = np.arccos(dot)

    obs = np.array(
        [
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            nu[0] / 2.0,
            nu[1] / 2.0,
            nu[5] / 2.0,
            distance / 5.0,
            evader_speed / max(1e-6, env.evader_v_max),
            evader_heading / np.pi,
            approach_angle / np.pi,
            1.0 if env.evader_visible else 0.0,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -10.0, 10.0)


def _high_action_to_subgoal_world(high_action: np.ndarray, env: LowLevelAUV2DPursuitEvasionEnv,
                                  radius: float) -> np.ndarray:
    eta = env.inner.dynamics.eta
    yaw = float(eta[5])
    body_offset = np.asarray(high_action, dtype=np.float64) * float(radius)
    dx_b, dy_b = body_offset[0], body_offset[1]
    dx_w = np.cos(yaw) * dx_b - np.sin(yaw) * dy_b
    dy_w = np.sin(yaw) * dx_b + np.cos(yaw) * dy_b
    subgoal = eta[:2] + np.array([dx_w, dy_w], dtype=np.float64)
    subgoal = np.clip(subgoal, -env.world_size, env.world_size)
    return subgoal


def _apply_continuous_low_level_action(env: LowLevelAUV2DPursuitEvasionEnv, action: np.ndarray, subgoal_xy: np.ndarray):
    action = np.asarray(action, dtype=np.float32).reshape(3)
    action = np.clip(action, -1.0, 1.0)

    env.inner.target_state[:3] = np.array([subgoal_xy[0], subgoal_xy[1], env.fixed_z], dtype=np.float64)

    eta = env.inner.dynamics.eta
    nu = env.inner.dynamics.nu
    dt = max(env.inner.dt, 1e-6)

    action_cfg = env.inner.config.get("action_space", {})
    distance_cfg = action_cfg.get("action_distance", {})
    angle_cfg = action_cfg.get("action_angle_delta", {})
    surge_step = float(distance_cfg.get("surge", 0.03))
    sway_step = float(distance_cfg.get("sway", 0.03))
    yaw_delta = float(angle_cfg.get("yaw", 0.0))

    linear_cmd = np.array([float(action[0]) * surge_step, float(action[1]) * sway_step, 0.0], dtype=np.float64)
    desired_vel_body = linear_cmd / dt

    slowdown_cfg = action_cfg.get("goal_slowdown", {})
    if slowdown_cfg.get("enabled", True):
        radius = float(slowdown_cfg.get("radius", 1.0))
        min_scale = float(slowdown_cfg.get("min_scale", 0.2))
        pos_error = np.linalg.norm(env.inner.target_state[:3] - eta[:3])
        if radius > 1e-6:
            scale = max(min_scale, min(1.0, pos_error / radius))
            desired_vel_body = desired_vel_body * scale

    current_vel_body = nu[:3]
    vel_error = desired_vel_body - current_vel_body
    ctl = env.inner.integrated_controller
    force_cmd = ctl.vel_kp * vel_error - ctl.vel_kd * current_vel_body
    force_cmd = np.clip(force_cmd, -ctl.max_force, ctl.max_force)

    roll_current, pitch_current, yaw_current = eta[3], eta[4], eta[5]
    p, q, r = nu[3], nu[4], nu[5]
    target_yaw = np.arctan2(np.sin(yaw_current + float(action[2]) * yaw_delta), np.cos(yaw_current + float(action[2]) * yaw_delta))
    angle_target = np.array([target_yaw, 0.0, 0.0], dtype=np.float64)
    angle_current = np.array([yaw_current, pitch_current, roll_current], dtype=np.float64)
    omega_current = np.array([r, q, p], dtype=np.float64)

    tau_attitude, _ = ctl.pid_controller.compute(angle_target, angle_current, omega_current, env.inner.dt)
    thrust_command = ctl.thrust_allocator.allocate_with_attitude(force_cmd, tau_attitude)
    env.inner.last_thrust_command = thrust_command.copy()

    external_force = env.inner._compute_external_force()
    env.inner.dynamics.step(thrust_command, external_force)
    env.inner.water_current.step(env.inner.dt)
    env.inner.current_step += 1
    env._enforce_2d_state()


def _compute_high_level_reward(env: LowLevelAUV2DPursuitEvasionEnv, prev_distance: float) -> Tuple[float, float, Dict[str, float]]:
    pursuer_xy = env.inner.dynamics.eta[:2]
    d = float(np.linalg.norm(pursuer_xy - env.evader_pos))
    progress = float(prev_distance - d)

    vel = env.inner.dynamics.nu[:2]
    los = env.evader_pos - pursuer_xy
    if np.linalg.norm(vel) < 1e-6 or np.linalg.norm(los) < 1e-6:
        approach = 0.0
    else:
        vdir = vel / (np.linalg.norm(vel) + 1e-8)
        ldir = los / (np.linalg.norm(los) + 1e-8)
        approach = float(np.clip(np.dot(vdir, ldir), -1.0, 1.0))

    reward = 8.0 * progress + 0.3 * approach - 0.02

    if d < env.catch_radius:
        reward += 100.0

    info = {
        "distance_to_evader": d,
        "progress": progress,
        "approach_alignment": approach,
    }
    return reward, d, info


def _safe_number(x: Any):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return None
        return float(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


class HighLevelContinuousHRLTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.exp_name = args.experiment_name or f"pe2d_hrl_cont_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.exp_name)
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.envs = [
            LowLevelAUV2DPursuitEvasionEnv(
                init_distance_range=tuple(args.init_distance_range),
                world_size=args.world_size,
                catch_radius=args.catch_radius,
                max_episode_seconds=args.max_episode_seconds,
                curriculum_stage=args.curriculum_stage,
            )
            for _ in range(args.n_envs)
        ]
        self.eval_env = LowLevelAUV2DPursuitEvasionEnv(
            init_distance_range=tuple(args.init_distance_range),
            world_size=args.world_size,
            catch_radius=args.catch_radius,
            max_episode_seconds=args.max_episode_seconds,
            curriculum_stage=args.curriculum_stage,
        )

        self._validate_asymmetry_constraints()

        self.high_obs_dim = 12
        self.high_act_dim = 2

        self.high_model = GaussianActorCritic(
            self.high_obs_dim,
            self.high_act_dim,
            hidden=tuple(args.high_level_hidden),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.high_model.parameters(), lr=args.lr)

        self.low_model = LowLevelContinuousPolicy(hidden=tuple(args.low_level_hidden)).to(self.device)
        ckpt = torch.load(args.low_level_ckpt, map_location=self.device, weights_only=False)
        if "model" not in ckpt:
            raise ValueError("low-level checkpoint must contain key 'model' (continuous low-level)")
        ll_state = ckpt["model"]
        inferred_hidden = _infer_low_actor_hidden_from_state_dict(ll_state)
        self.low_model = LowLevelContinuousPolicy(hidden=inferred_hidden).to(self.device)
        missing, unexpected = self.low_model.load_state_dict(ll_state, strict=False)
        if unexpected:
            print(f"[LowLevelLoad] ignored unexpected keys: {unexpected}")
        if missing:
            print(f"[LowLevelLoad] missing keys: {missing}")
        print(f"Low-level actor hidden inferred: {inferred_hidden}")
        self.low_model.eval()

        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_range = args.clip_range
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.n_steps = args.n_steps
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs

        self.episode_count = 0
        self.success_count = 0
        self.history: List[Dict[str, Any]] = []

    def _validate_asymmetry_constraints(self):
        env0 = self.envs[0]
        pv = float(env0.pursuer_v_max)
        ev = float(env0.evader_v_max)
        pa = float(env0.pursuer_a_max)
        ea = float(env0.evader_a_max)

        speed_ok = pv < ev
        accel_ok = pa > ea
        msg = (
            f"[Asymmetry] pursuer_v_max={pv:.3f}, evader_v_max={ev:.3f} (need pursuer<evader) | "
            f"pursuer_a_max={pa:.3f}, evader_a_max={ea:.3f} (need pursuer>evader)"
        )
        print(msg)

        if self.args.strict_asymmetry and (not speed_ok or not accel_ok):
            raise ValueError("Asymmetry constraints violated. Please adjust configs/robot_config.yaml")

        if (not speed_ok) or (not accel_ok):
            print("[Asymmetry][WARN] constraints not satisfied, training may not reflect asymmetric game intent.")

    def _reset_one(self, env: LowLevelAUV2DPursuitEvasionEnv):
        env.reset()
        high_obs = _get_high_level_obs(env)
        prev_d = float(np.linalg.norm(env.inner.dynamics.eta[:2] - env.evader_pos))
        return high_obs, prev_d

    def collect_rollouts(self):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []

        obs_list = []
        prev_d_list = []
        for env in self.envs:
            obs_i, prev_d = self._reset_one(env)
            obs_list.append(obs_i)
            prev_d_list.append(prev_d)
        obs = np.array(obs_list, dtype=np.float32)
        prev_distances = np.array(prev_d_list, dtype=np.float32)

        for _ in range(self.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                high_action, logp, _ = self.high_model.get_action(obs_t, deterministic=False)
                value = self.high_model.get_value(obs_t)

            high_actions = high_action.cpu().numpy()
            logp_np = logp.cpu().numpy()
            values_np = value.cpu().numpy()

            rewards = np.zeros(len(self.envs), dtype=np.float32)
            dones = np.zeros(len(self.envs), dtype=np.float32)
            next_obs = np.zeros_like(obs)

            for i, env in enumerate(self.envs):
                subgoal = _high_action_to_subgoal_world(high_actions[i], env, self.args.subgoal_radius)

                for _k in range(self.args.low_level_steps):
                    env._step_evader()

                    eta = env.inner.dynamics.eta
                    nu = env.inner.dynamics.nu
                    low_obs = _get_low_level_obs(
                        eta[:2].copy(),
                        nu[:2].copy(),
                        float(eta[5]),
                        float(nu[5]),
                        subgoal,
                    )

                    with torch.no_grad():
                        low_obs_t = torch.tensor(low_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        low_act = self.low_model.get_action(low_obs_t, deterministic=True)[0].cpu().numpy()

                    _apply_continuous_low_level_action(env, low_act, subgoal)

                    step_reward, new_d, _ = _compute_high_level_reward(env, float(prev_distances[i]))
                    rewards[i] += float(step_reward)
                    prev_distances[i] = float(new_d)

                    terminated, truncated, info_term = env._check_termination()
                    if terminated or truncated:
                        dones[i] = 1.0
                        self.episode_count += 1
                        if info_term.get("termination_reason") == "success":
                            self.success_count += 1
                        obs_i, prev_d = self._reset_one(env)
                        next_obs[i] = obs_i
                        prev_distances[i] = prev_d
                        break

                if dones[i] < 0.5:
                    next_obs[i] = _get_high_level_obs(env)

            obs_buf.append(obs)
            act_buf.append(high_actions)
            rew_buf.append(rewards)
            done_buf.append(dones)
            val_buf.append(values_np)
            logp_buf.append(logp_np)
            obs = next_obs

        with torch.no_grad():
            last_values = self.high_model.get_value(torch.tensor(obs, dtype=torch.float32, device=self.device)).cpu().numpy()

        rewards = np.array(rew_buf)
        values = np.array(val_buf)
        dones = np.array(done_buf)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(len(self.envs), dtype=np.float32)
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

    def train_step(self, rollout_data: Dict[str, np.ndarray]) -> float:
        obs = torch.tensor(rollout_data["obs"].reshape(-1, self.high_obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout_data["actions"].reshape(-1, self.high_act_dim), dtype=torch.float32, device=self.device)
        old_logprobs = torch.tensor(rollout_data["logprobs"].reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout_data["advantages"].reshape(-1), dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout_data["returns"].reshape(-1), dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss = 0.0

        for _ in range(self.n_epochs):
            idx = torch.randperm(len(obs), device=self.device)
            for s in range(0, len(obs), self.batch_size):
                b = idx[s:s + self.batch_size]
                batch_obs = obs[b]
                batch_actions = actions[b]
                batch_old_logp = old_logprobs[b]
                batch_adv = advantages[b]
                batch_ret = returns[b]

                new_logp, ent_each = self.high_model.evaluate_actions(batch_obs, batch_actions)
                entropy = ent_each.mean()
                values = self.high_model.get_value(batch_obs)

                ratio = torch.exp(new_logp - batch_old_logp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values, batch_ret)
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.high_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += float(loss.item())

        return total_loss / max(1, self.n_epochs)

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        rewards = []
        successes = 0

        for _ in range(n_episodes):
            env = self.eval_env
            env.reset()
            obs = _get_high_level_obs(env)
            prev_d = float(np.linalg.norm(env.inner.dynamics.eta[:2] - env.evader_pos))
            done = False
            ep_reward = 0.0

            while not done:
                with torch.no_grad():
                    x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    high_act, _, _ = self.high_model.get_action(x, deterministic=True)
                    high_act_np = high_act[0].cpu().numpy()

                subgoal = _high_action_to_subgoal_world(high_act_np, env, self.args.subgoal_radius)

                for _k in range(self.args.low_level_steps):
                    env._step_evader()
                    eta = env.inner.dynamics.eta
                    nu = env.inner.dynamics.nu
                    low_obs = _get_low_level_obs(eta[:2].copy(), nu[:2].copy(), float(eta[5]), float(nu[5]), subgoal)
                    with torch.no_grad():
                        low_obs_t = torch.tensor(low_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        low_act = self.low_model.get_action(low_obs_t, deterministic=True)[0].cpu().numpy()
                    _apply_continuous_low_level_action(env, low_act, subgoal)

                    r, prev_d, _ = _compute_high_level_reward(env, prev_d)
                    ep_reward += float(r)

                    term, trunc, info = env._check_termination()
                    done = term or trunc
                    if done:
                        if info.get("termination_reason") == "success":
                            successes += 1
                        break

                obs = _get_high_level_obs(env)

            rewards.append(ep_reward)

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "success_rate": float(successes / max(1, n_episodes)),
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
        }

    def save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save(
            {
                "high_model": self.high_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": vars(self.args),
                "high_obs_dim": self.high_obs_dim,
                "high_act_dim": self.high_act_dim,
                "low_level_ckpt": self.args.low_level_ckpt,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    def save_history(self):
        if not self.history:
            return

        csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        json_path = os.path.join(self.log_dir, "training_metrics.json")
        fields = [
            "update", "steps", "train_mean_reward", "train_loss", "train_success_rate",
            "eval_mean_reward", "eval_success_rate",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in self.history:
                w.writerow({k: _safe_number(row.get(k)) for k in fields})

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([{k: _safe_number(v) for k, v in row.items()} for row in self.history], f, ensure_ascii=False, indent=2)

    def plot_curves(self):
        if not self.history:
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
        ax.grid(True, alpha=0.25)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(steps, train_loss, lw=1.8, color="tab:orange")
        ax.set_title("Loss Curve")
        ax.grid(True, alpha=0.25)

        ax = axes[1, 0]
        ax.plot(steps, train_sr, lw=1.8, label="train_success_rate")
        valid = ~np.isnan(eval_sr)
        if np.any(valid):
            ax.plot(steps[valid], eval_sr[valid], lw=1.8, label="eval_success_rate")
        ax.set_ylim(0, 1)
        ax.set_title("Success Rate Curve")
        ax.grid(True, alpha=0.25)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(np.arange(1, len(train_rew) + 1), train_rew, lw=1.8, label="reward by update")
        ax.set_title("Reward vs Update")
        ax.grid(True, alpha=0.25)
        ax.legend()

        plt.tight_layout()
        out_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training curves: {out_path}")

    def train(self):
        total_steps = self.args.total_timesteps
        steps_per_update = self.n_steps * self.args.n_envs
        n_updates = max(1, total_steps // steps_per_update)

        print(f"Total timesteps: {total_steps}")
        print(f"Steps/update: {steps_per_update}")
        print(f"Num updates: {n_updates}")

        best_eval_reward = -1e9
        last_eval_rew = None
        last_eval_sr = None
        start_t = time.time()

        for update in range(1, n_updates + 1):
            rollout = self.collect_rollouts()
            loss = self.train_step(rollout)

            current_steps = update * steps_per_update
            train_sr = self.success_count / max(1, self.episode_count)

            if update % 10 == 0:
                fps = current_steps / max(1e-6, time.time() - start_t)
                print(
                    f"Update {update}/{n_updates} | Steps {current_steps} | FPS {fps:.1f} | "
                    f"TrainRew {rollout['mean_reward']:.3f} | Loss {loss:.4f} | Succ {train_sr:.2%}",
                    flush=True,
                )

            if current_steps % self.args.eval_freq < steps_per_update:
                eval_info = self.evaluate(n_episodes=self.args.eval_episodes)
                last_eval_rew = eval_info["mean_reward"]
                last_eval_sr = eval_info["success_rate"]
                print(
                    f"  [Eval] steps={current_steps} mean_reward={last_eval_rew:.3f} "
                    f"success_rate={last_eval_sr:.2%}",
                    flush=True,
                )
                if last_eval_rew > best_eval_reward:
                    best_eval_reward = last_eval_rew
                    self.save_checkpoint("best")

            if current_steps % self.args.save_freq < steps_per_update:
                self.save_checkpoint("latest")

            self.history.append(
                {
                    "update": int(update),
                    "steps": int(current_steps),
                    "train_mean_reward": float(rollout["mean_reward"]),
                    "train_loss": float(loss),
                    "train_success_rate": float(train_sr),
                    "eval_mean_reward": None if last_eval_rew is None else float(last_eval_rew),
                    "eval_success_rate": None if last_eval_sr is None else float(last_eval_sr),
                }
            )

            if update % 20 == 0:
                self.save_history()

        self.save_checkpoint("final")
        self.save_history()
        self.plot_curves()
        print("High-level continuous HRL training finished.")


def _parse_distance_range(v: str) -> Tuple[float, float]:
    parts = [x.strip() for x in v.split(",")]
    if len(parts) != 2:
        return (2.0, 6.0)
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return (2.0, 6.0)


def main():
    parser = argparse.ArgumentParser(description="Train high-level continuous PPO over fixed low-level continuous policy")
    parser.add_argument("--low-level-ckpt", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--init-distance-range", type=str, default="2,6")
    parser.add_argument("--world-size", type=float, default=10.0)
    parser.add_argument("--catch-radius", type=float, default=0.5)
    parser.add_argument("--max-episode-seconds", type=float, default=90.0)
    parser.add_argument("--curriculum-stage", type=int, default=0)
    parser.add_argument("--strict-asymmetry", action="store_true", help="Fail fast if pursuer_v_max>=evader_v_max or pursuer_a_max<=evader_a_max")

    parser.add_argument("--subgoal-radius", type=float, default=2.0)
    parser.add_argument("--low-level-steps", type=int, default=6)

    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=50000)

    parser.add_argument("--low-level-hidden", type=int, nargs=2, default=[128, 64])
    parser.add_argument("--high-level-hidden", type=int, nargs=3, default=[256, 128, 64])

    args = parser.parse_args()
    args.init_distance_range = _parse_distance_range(args.init_distance_range)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainer = HighLevelContinuousHRLTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
