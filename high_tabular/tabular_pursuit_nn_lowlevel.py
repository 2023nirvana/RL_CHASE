#!/usr/bin/env python3
"""Tabular Q-learning + Neural Network Low-Level Controller for Pursuit-Evasion.

High-level: Tabular Q-learning (8 discrete directions)
Low-level: Pre-trained NN policy from v4_tight (continuous 3D action)
"""

import argparse
import math
import random
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from high_tabular.tabular_pursuit_qlearning import (
    RectObstacle, 
    epsilon_by_episode,
    plot_training_curves,
)
from core.environment.low_level_env_2d_v3_continuous import LowLevelAUV2DEnvV3Continuous


class LowLevelNNPolicy(nn.Module):
    """低层NN策略 - 从 v4_tight checkpoint 加载"""
    
    def __init__(self, obs_dim: int = 8, act_dim: int = 3, hidden=(64, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def get_action(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mu = self.mu_head(self.actor_body(x))
        if deterministic:
            return torch.tanh(mu)
        log_std = torch.clamp(self.log_std, -5.0, 1.0)
        std = torch.exp(log_std)
        z = mu + std * torch.randn_like(mu)
        return torch.tanh(z)


def load_nn_policy(checkpoint_path: str, device: torch.device) -> LowLevelNNPolicy:
    """加载预训练的NN低层策略"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    hidden = []
    idx = 0
    while f"actor_body.{idx}.weight" in state_dict:
        hidden.append(state_dict[f"actor_body.{idx}.weight"].shape[0])
        idx += 2
    
    policy = LowLevelNNPolicy(obs_dim=8, act_dim=3, hidden=tuple(hidden))
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()
    return policy


class PursuitEvasionNNEnv:
    """追逃环境：高层tabular + 低层NN策略"""
    
    def __init__(self, nn_policy: LowLevelNNPolicy, device: torch.device, seed: int = 0, use_pd: bool = False):
        self.nn_policy = nn_policy
        self.device = device
        self.use_pd = use_pd  # 使用简单PD控制替代NN（用于调试）
        self.rng = np.random.default_rng(seed)
        
        self.L = 15.0
        self.dt = 0.02
        self.max_high_steps = 150
        self.capture_dist = 1.0
        self.escape_dist = 27.0
        
        self.pursuer_v_max = 1.5
        self.v_max_e = 1.35
        self.a_max_e = 0.55
        
        self.n_actions = 8
        self.subgoal_radius = 0.7
        self.low_k = 20
        self.boundary_margin = 3.0
        
        self.obstacles = [
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
        
        # 使用真正的AUV环境作为底层
        self.auv_env = LowLevelAUV2DEnvV3Continuous(
            target_distance_min=0.3,
            target_distance_max=1.0,
            reach_threshold=0.05,
            max_episode_seconds=30.0,
        )
        
        self.p = np.zeros(2, dtype=np.float64)
        self.p_yaw = 0.0
        self.p_vel = np.zeros(2, dtype=np.float64)
        self.p_yaw_rate = 0.0
        
        self.e = np.zeros(2, dtype=np.float64)
        self.ve = np.zeros(2, dtype=np.float64)
        
        self.prev_d = 0.0
        self.step_n = 0
        self.prev_action = 0
        self.position_history = []
        self.action_history = []

    @staticmethod
    def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= max_norm or n < 1e-12:
            return v
        return v * (max_norm / n)

    def _nearest_obstacle_dist(self, p: np.ndarray) -> float:
        return min(obs.distance(p) for obs in self.obstacles)

    def _is_near_boundary(self, p: np.ndarray) -> bool:
        return abs(p[0]) > self.L - self.boundary_margin or abs(p[1]) > self.L - self.boundary_margin

    def _boundary_dist(self, p: np.ndarray) -> float:
        return min(self.L - abs(p[0]), self.L - abs(p[1]))

    def _move_evader_with_collisions(self, p: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """逃脱者移动和碰撞处理"""
        p_new = p + v * self.dt
        
        for obs in self.obstacles:
            if obs.contains(p_new):
                d_left = abs(p_new[0] - obs.x_min)
                d_right = abs(obs.x_max - p_new[0])
                d_bottom = abs(p_new[1] - obs.y_min)
                d_top = abs(obs.y_max - p_new[1])
                which = int(np.argmin([d_left, d_right, d_bottom, d_top]))
                margin = 0.05
                if which == 0:
                    p_new[0] = obs.x_min - margin
                    v[0] *= -0.1
                elif which == 1:
                    p_new[0] = obs.x_max + margin
                    v[0] *= -0.1
                elif which == 2:
                    p_new[1] = obs.y_min - margin
                    v[1] *= -0.1
                else:
                    p_new[1] = obs.y_max + margin
                    v[1] *= -0.1
        
        p_new = np.clip(p_new, -self.L, self.L)
        return p_new, v

    def _evader_acc(self) -> np.ndarray:
        """逃脱者AI"""
        d = self.e - self.p
        n = np.linalg.norm(d)
        run = d / (n + 1e-6)

        avoid_obs = np.zeros(2, dtype=np.float64)
        for obs in self.obstacles:
            q = np.array([np.clip(self.e[0], obs.x_min, obs.x_max), 
                         np.clip(self.e[1], obs.y_min, obs.y_max)])
            away = self.e - q
            dn = np.linalg.norm(away)
            if dn < 4.0:
                avoid_obs += away / (dn + 1e-3) * (2.5 / (dn + 0.3))

        avoid_wall = np.zeros(2, dtype=np.float64)
        margin = 0.55 * self.L
        wall_strength = 3.5
        
        if self.e[0] > margin:
            dist_to_wall = self.L - self.e[0]
            avoid_wall[0] -= wall_strength * (1.0 / (dist_to_wall + 0.5))
        if self.e[0] < -margin:
            dist_to_wall = self.L + self.e[0]
            avoid_wall[0] += wall_strength * (1.0 / (dist_to_wall + 0.5))
        if self.e[1] > margin:
            dist_to_wall = self.L - self.e[1]
            avoid_wall[1] -= wall_strength * (1.0 / (dist_to_wall + 0.5))
        if self.e[1] < -margin:
            dist_to_wall = self.L + self.e[1]
            avoid_wall[1] += wall_strength * (1.0 / (dist_to_wall + 0.5))

        center_pull = -self.e * 0.03
        lateral = np.array([-run[1], run[0]], dtype=np.float64)
        cmd = 1.0 * run + 0.8 * avoid_obs + 1.0 * avoid_wall + center_pull + 0.25 * lateral
        return self._clip_norm(cmd, 1.0) * self.a_max_e

    def _get_nn_obs(self, target_pos: np.ndarray) -> np.ndarray:
        """构建NN低层策略的8维观测 - 设置AUV目标并获取观测"""
        # 设置AUV env的目标
        self.auv_env.inner.target_state[:2] = target_pos
        self.auv_env.inner.target_state[2] = self.auv_env.fixed_z
        
        # 使用AUV env的方法获取观测
        return self.auv_env._get_observation()

    def _apply_nn_action(self, action: np.ndarray):
        """执行NN策略输出的连续动作 - 使用真正的AUV动力学"""
        action = np.clip(action, -1.0, 1.0)
        
        # 使用AUV env的step执行动作
        self.auv_env.step(action)
        
        # 从AUV env获取更新后的状态
        eta = self.auv_env.inner.dynamics.eta
        nu = self.auv_env.inner.dynamics.nu
        
        new_p = eta[:2].copy()
        
        # 碰撞检测
        for obs in self.obstacles:
            if obs.contains(new_p):
                # 如果碰撞，回滚
                return
        
        new_p = np.clip(new_p, -self.L, self.L)
        self.p = new_p
        self.p_yaw = eta[5]
        self.p_vel[0] = nu[0]
        self.p_vel[1] = nu[1]
        self.p_yaw_rate = nu[5]

    def _state_discrete(self) -> tuple:
        """状态离散化"""
        rel = self.e - self.p
        d = float(np.linalg.norm(rel))

        d_edges = [1.0, 3.0, 5.0, 8.0, 11.0, 15.0, 20.0]
        d_bin = len(d_edges)
        for i, edge in enumerate(d_edges):
            if d < edge:
                d_bin = i
                break

        bearing = math.atan2(rel[1], rel[0])
        b_bin = int(((bearing + math.pi) / (2 * math.pi)) * 8.0) % 8

        vp_world = np.array([
            np.cos(self.p_yaw) * self.p_vel[0] - np.sin(self.p_yaw) * self.p_vel[1],
            np.sin(self.p_yaw) * self.p_vel[0] + np.cos(self.p_yaw) * self.p_vel[1]
        ])
        heading = math.atan2(vp_world[1] + 1e-9, vp_world[0] + 1e-9)
        heading_err = (bearing - heading + math.pi) % (2 * math.pi) - math.pi
        h_bin = int(((heading_err + math.pi) / (2 * math.pi)) * 4.0) % 4

        e_heading = math.atan2(self.ve[1] + 1e-9, self.ve[0] + 1e-9)
        flee_angle = (e_heading - bearing + math.pi) % (2 * math.pi) - math.pi
        ef_bin = int(((flee_angle + math.pi) / (2 * math.pi)) * 4.0) % 4

        d_obs_e = self._nearest_obstacle_dist(self.e)
        oe_bin = 0 if d_obs_e < 1.5 else (1 if d_obs_e < 4.0 else 2)
        
        d_obs_p = self._nearest_obstacle_dist(self.p)
        op_bin = 0 if d_obs_p < 2.0 else (1 if d_obs_p < 5.0 else 2)
        
        eb_bin = 0 if self._is_near_boundary(self.e) else 1
        prev_a_bin = self.prev_action // 2

        return d_bin, b_bin, h_bin, ef_bin, oe_bin, op_bin, eb_bin, prev_a_bin

    def reset(self) -> tuple:
        """重置环境"""
        while True:
            self.p = self.rng.uniform(-0.6 * self.L, 0.6 * self.L, size=2).astype(np.float64)
            self.e = self.rng.uniform(-0.6 * self.L, 0.6 * self.L, size=2).astype(np.float64)
            if np.linalg.norm(self.e - self.p) < 6.0:
                continue
            if any(obs.contains(self.p) or obs.contains(self.e) for obs in self.obstacles):
                continue
            break

        self.p_yaw = self.rng.uniform(-np.pi, np.pi)
        self.p_vel = np.zeros(2, dtype=np.float64)
        self.p_yaw_rate = 0.0
        
        # 同步AUV env
        self.auv_env.reset()
        self.auv_env.inner.dynamics.eta[:2] = self.p
        self.auv_env.inner.dynamics.eta[5] = self.p_yaw
        self.auv_env.inner.dynamics.nu[:] = 0.0
        
        self.ve = np.zeros(2, dtype=np.float64)
        self.step_n = 0
        self.prev_d = float(np.linalg.norm(self.e - self.p))
        
        self.prev_action = 0
        self.position_history = [self.p.copy()]
        self.action_history = []
        
        return self._state_discrete()

    def step(self, action_idx: int):
        """执行一步高层动作"""
        rel = self.e - self.p
        alpha = math.atan2(rel[1], rel[0])
        theta = alpha + action_idx * (2.0 * math.pi / self.n_actions)
        subgoal = self.p + self.subgoal_radius * np.array([math.cos(theta), math.sin(theta)])

        for _ in range(self.low_k):
            if self.use_pd:
                # 简单PD控制（与原tabular一致）
                a_p = 1.6 * (subgoal - self.p) - 0.9 * self.p_vel
                a_p = self._clip_norm(a_p, 1.8)
                self.p_vel = self._clip_norm(self.p_vel + a_p * self.dt, self.pursuer_v_max)
                new_p = self.p + self.p_vel * self.dt
                # 碰撞检测
                blocked = False
                for obs in self.obstacles:
                    if obs.contains(new_p):
                        blocked = True
                        break
                if not blocked:
                    self.p = np.clip(new_p, -self.L, self.L)
            else:
                # NN低层控制
                nn_obs = self._get_nn_obs(subgoal)
                with torch.no_grad():
                    obs_t = torch.from_numpy(nn_obs).unsqueeze(0).to(self.device)
                    action_t = self.nn_policy.get_action(obs_t, deterministic=True)
                    action = action_t.cpu().numpy().flatten()
                self._apply_nn_action(action)
            
            a_e = self._evader_acc()
            self.ve = self._clip_norm(self.ve + a_e * self.dt, self.v_max_e)
            self.e, self.ve = self._move_evader_with_collisions(self.e, self.ve)

        self.step_n += 1
        d = float(np.linalg.norm(self.e - self.p))
        done = False
        captured = False
        
        d_obs_e = self._nearest_obstacle_dist(self.e)
        evader_near_boundary = self._is_near_boundary(self.e)
        capture_pos = self.e.copy()

        reward = -0.25

        action_diff = min(abs(action_idx - self.prev_action), 
                          self.n_actions - abs(action_idx - self.prev_action))
        if action_diff >= 3:
            reward -= 0.8 * action_diff
        elif action_diff >= 2:
            reward -= 0.3 * action_diff
        
        self.position_history.append(self.p.copy())
        self.action_history.append(action_idx)
        if len(self.position_history) > 15:
            self.position_history.pop(0)
            self.action_history.pop(0)
        
        if len(self.position_history) >= 8:
            loop_penalty = 0.0
            for old_pos in self.position_history[:-4]:
                dist_to_old = float(np.linalg.norm(self.p - old_pos))
                if dist_to_old < 2.0:
                    loop_penalty += 0.5 * (2.0 - dist_to_old)
            if loop_penalty > 0:
                reward -= min(loop_penalty, 3.0)
        
        if len(self.action_history) >= 4:
            recent = self.action_history[-4:]
            if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
                reward -= 1.5
        
        self.prev_action = action_idx

        improve = self.prev_d - d
        reward += 6.0 * improve

        if d_obs_e < 3.0:
            reward += 1.2 * (3.0 - d_obs_e)

        e_dir = self.ve / (float(np.linalg.norm(self.ve)) + 1e-6)
        p_ahead = self.p - self.e
        intercept_proj = float(np.dot(p_ahead, e_dir))
        if intercept_proj > 0:
            reward += 0.6 * min(intercept_proj, 4.0)

        if d <= self.capture_dist:
            done = True
            captured = True
            reward += 100.0
            if d_obs_e < 2.0:
                reward += 80.0 * (2.0 - d_obs_e)
            elif d_obs_e < 4.0:
                reward += 15.0 * (4.0 - d_obs_e)
            
            boundary_dist_e = self._boundary_dist(self.e)
            if boundary_dist_e < self.boundary_margin:
                reward -= 60.0 * (1.0 - boundary_dist_e / self.boundary_margin)
            
            center_dist = float(np.linalg.norm(self.e))
            if center_dist < 8.0:
                reward += 40.0 * (1.0 - center_dist / 8.0)
                
        elif d >= self.escape_dist or self.step_n >= self.max_high_steps:
            reward -= 50.0
            done = True

        self.prev_d = d
        return self._state_discrete(), float(reward), done, {
            "captured": captured,
            "distance": d,
            "capture_pos": capture_pos if captured else None,
            "near_terrain": d_obs_e < 3.0 if captured else False,
            "near_boundary": evader_near_boundary if captured else False,
        }


def train_tabular_nn(
    nn_policy: LowLevelNNPolicy,
    device: torch.device,
    episodes: int = 3000,
    alpha: float = 0.15,
    gamma: float = 0.98,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 1200,
    seed: int = 0,
    use_pd: bool = False,
):
    """Tabular Q-learning训练 (使用NN低层策略或PD控制)"""
    random.seed(seed)
    np.random.seed(seed)

    env = PursuitEvasionNNEnv(nn_policy, device, seed=seed, use_pd=use_pd)
    q_table: Dict[tuple, np.ndarray] = {}

    def get_q(s):
        if s not in q_table:
            q_table[s] = np.zeros(env.n_actions, dtype=np.float32)
        return q_table[s]

    capture_history: List[float] = []
    reward_history: List[float] = []

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_ret = 0.0
        eps = epsilon_by_episode(ep, eps_start, eps_end, eps_decay)
        captured = False

        while not done:
            if random.random() < eps:
                a = random.randrange(env.n_actions)
            else:
                a = int(np.argmax(get_q(s)))

            s2, r, done, info = env.step(a)
            ep_ret += r

            q = get_q(s)
            q_next = get_q(s2)
            td_target = r if done else r + gamma * float(np.max(q_next))
            q[a] += alpha * (td_target - q[a])
            s = s2
            captured = captured or bool(info.get("captured", False))

        capture_history.append(1.0 if captured else 0.0)
        reward_history.append(ep_ret)

        if ep % 50 == 0:
            recent_cap = float(np.mean(capture_history[-50:]))
            recent_ret = float(np.mean(reward_history[-50:]))
            print(f"[NN-tabular] ep={ep:4d} eps={eps:.3f} cap={recent_cap:.2%} ret={recent_ret:.1f} states={len(q_table)}")

    return env, q_table, capture_history, reward_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="low_near/v4_tight/checkpoints/best.pth")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="high_tabular/results_nn")
    parser.add_argument("--use-pd", action="store_true", help="Use simple PD control instead of NN")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    print(f"Loading NN policy from: {ckpt_path}")
    nn_policy = load_nn_policy(ckpt_path, device)
    print("NN policy loaded successfully!")

    mode_str = "PD Control" if args.use_pd else "NN Low-Level"
    print("=" * 60)
    print(f"Tabular Q-Learning + {mode_str}")
    print(f"Episodes: {args.episodes}, Seed: {args.seed}")
    print("=" * 60)

    env, q_table, cap_hist, ret_hist = train_tabular_nn(
        nn_policy, device,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        seed=args.seed,
        use_pd=args.use_pd,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    
    final_cap = float(np.mean(cap_hist[-100:])) if len(cap_hist) >= 100 else float(np.mean(cap_hist))
    print("-" * 60)
    print(f"Training complete!")
    print(f"Final capture rate: {final_cap:.2%}")
    print(f"Q-table states: {len(q_table)}")
    print("-" * 60)

    plot_training_curves(cap_hist, ret_hist, args.save_dir)
    
    import pickle
    with open(os.path.join(args.save_dir, "q_table.pkl"), "wb") as f:
        pickle.dump({"q_table": q_table, "cap_hist": cap_hist, "ret_hist": ret_hist}, f)
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
