#!/usr/bin/env python3
"""
连续动作空间的HRL追逃
====================
高层：连续输出子目标方向（角度）
底层：V4 NN策略执行子目标到达

核心思路：
- 高层学习子目标序列规划能力
- 底层V4负责实际到达每个子目标
- 速度设置：pursuer慢但加速度大，需要利用地形
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# ============== 底层NN策略 ==============

class LowLevelNNPolicy(nn.Module):
    """底层NN策略（与V4训练结构一致）"""
    def __init__(self, obs_dim=8, act_dim=3, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.shared = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)
        
    def forward(self, obs):
        h = self.shared(obs)
        return torch.tanh(self.mu_head(h))


def load_nn_policy(ckpt_path: str, device: torch.device) -> LowLevelNNPolicy:
    """加载NN策略，自动推断hidden sizes"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
    
    hidden_sizes = []
    idx = 0
    while f"shared.{idx}.weight" in state_dict:
        hidden_sizes.append(state_dict[f"shared.{idx}.weight"].shape[0])
        idx += 2
    
    policy = LowLevelNNPolicy(obs_dim=8, act_dim=3, hidden_sizes=hidden_sizes)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()
    return policy


# ============== 简化2D物理（与原tabular一致） ==============

@dataclass
class RectObstacle:
    cx: float
    cy: float
    w: float
    h: float
    
    def contains(self, p: np.ndarray) -> bool:
        return (abs(p[0] - self.cx) <= self.w / 2 and 
                abs(p[1] - self.cy) <= self.h / 2)
    
    def dist(self, p: np.ndarray) -> float:
        dx = max(abs(p[0] - self.cx) - self.w / 2, 0)
        dy = max(abs(p[1] - self.cy) - self.h / 2, 0)
        return math.sqrt(dx * dx + dy * dy)


class SimplePursuitEnv:
    """
    简化2D追逃环境（与原tabular_pursuit_qlearning一致的物理）
    
    关键设置（来自MDP定义）：
    - pursuer: v_max=1.1, a_max=1.8 （慢但加速快）
    - evader:  v_max=1.35, a_max=0.55 （快但加速慢）
    - 追逐者需要利用地形优势
    """
    
    def __init__(self,
                 world_size: float = 15.0,
                 catch_radius: float = 1.0,
                 max_steps: int = 400,
                 dt: float = 0.1,
                 seed: int = 0):
        self.world_size = world_size
        self.half = world_size / 2
        self.catch_radius = catch_radius
        self.max_steps = max_steps
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        
        # 速度/加速度设置（与原tabular一致）
        self.v_max_p = 1.1
        self.v_max_e = 1.35
        self.a_max_p = 1.8
        self.a_max_e = 0.55
        
        self.boundary_margin = 3.0
        
        # 障碍物（与原tabular一致）
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
            RectObstacle(5.5, -5.0, 2.5, 2.5),
        ]
        
        self.reset()

    
    def _clip_norm(self, v: np.ndarray, max_n: float) -> np.ndarray:
        n = np.linalg.norm(v)
        return v if n <= max_n else v * (max_n / n)
    
    def _is_valid_pos(self, p: np.ndarray) -> bool:
        if abs(p[0]) > self.half or abs(p[1]) > self.half:
            return False
        for obs in self.obstacles:
            if obs.contains(p):
                return False
        return True
    
    def _move_with_collisions(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_pos = pos + vel * self.dt
        # 边界
        new_pos = np.clip(new_pos, -self.half, self.half)
        # 障碍物
        for obs in self.obstacles:
            if obs.contains(new_pos):
                new_pos = pos.copy()
                vel = vel * 0.0
                break
        return new_pos, vel
    
    def _nearest_obstacle_dist(self, p: np.ndarray) -> float:
        return min(obs.dist(p) for obs in self.obstacles)
    
    def _is_near_boundary(self, p: np.ndarray) -> bool:
        return (abs(p[0]) > self.half - self.boundary_margin or 
                abs(p[1]) > self.half - self.boundary_margin)
    
    def _spawn_valid_pos(self) -> np.ndarray:
        for _ in range(100):
            p = self.rng.uniform(-self.half + 1, self.half - 1, size=2).astype(np.float32)
            if self._is_valid_pos(p):
                return p
        return np.array([0.0, -5.0], dtype=np.float32)

    
    def _evader_acc(self) -> np.ndarray:
        """逃跑者AI（与原tabular一致）"""
        rel = self.p - self.e
        d = np.linalg.norm(rel)
        
        if d < 0.1:
            return self.rng.uniform(-1, 1, size=2).astype(np.float32) * self.a_max_e
        
        # 基础逃跑方向
        flee_dir = -rel / d
        
        # 避障
        obs_weight = 0.0
        obs_dir = np.zeros(2, dtype=np.float32)
        for obs in self.obstacles:
            od = obs.dist(self.e)
            if od < 2.0:
                to_e = self.e - np.array([obs.cx, obs.cy])
                obs_dir += to_e / (od + 0.1)
                obs_weight += 1.0 / (od + 0.1)
        
        if obs_weight > 0:
            obs_dir = obs_dir / obs_weight
            obs_norm = np.linalg.norm(obs_dir)
            if obs_norm > 0:
                obs_dir /= obs_norm
            flee_dir = 0.6 * flee_dir + 0.4 * obs_dir
        
        # 边界
        if self._is_near_boundary(self.e):
            center_dir = -self.e / (np.linalg.norm(self.e) + 1e-6)
            flee_dir = 0.5 * flee_dir + 0.5 * center_dir
        
        # 随机扰动
        flee_dir += self.rng.uniform(-0.2, 0.2, size=2)
        norm = np.linalg.norm(flee_dir)
        if norm > 0:
            flee_dir /= norm
        
        return flee_dir * self.a_max_e
    
    def reset(self) -> Dict:
        self.p = self._spawn_valid_pos()
        self.e = self._spawn_valid_pos()
        
        # 确保初始距离合适
        while np.linalg.norm(self.e - self.p) < 4.0:
            self.e = self._spawn_valid_pos()
        
        self.vp = np.zeros(2, dtype=np.float32)
        self.ve = np.zeros(2, dtype=np.float32)
        self.step_n = 0
        
        self.p_history = [self.p.copy()]
        self.e_history = [self.e.copy()]
        
        return self._get_obs()

    
    def _get_obs(self) -> Dict:
        """高层观察（连续值）"""
        rel = self.e - self.p
        d = np.linalg.norm(rel)
        bearing = math.atan2(rel[1], rel[0])
        
        # 追逐者速度方向
        vp_norm = np.linalg.norm(self.vp)
        heading_p = math.atan2(self.vp[1], self.vp[0]) if vp_norm > 0.01 else 0.0
        
        # 逃跑者速度方向
        ve_norm = np.linalg.norm(self.ve)
        heading_e = math.atan2(self.ve[1], self.ve[0]) if ve_norm > 0.01 else 0.0
        
        # 障碍物距离
        d_obs_p = self._nearest_obstacle_dist(self.p)
        d_obs_e = self._nearest_obstacle_dist(self.e)
        
        return {
            "distance": d,
            "bearing": bearing,
            "heading_p": heading_p,
            "heading_e": heading_e,
            "speed_p": vp_norm,
            "speed_e": ve_norm,
            "d_obs_p": d_obs_p,
            "d_obs_e": d_obs_e,
            "pos_p": self.p.copy(),
            "pos_e": self.e.copy(),
            "vel_p": self.vp.copy(),
            "vel_e": self.ve.copy(),
        }
    
    def step_low(self, subgoal: np.ndarray, low_steps: int = 8) -> Tuple[Dict, float, bool, Dict]:
        """
        执行底层控制：使用简化PD控制器向子目标移动
        （与原tabular一致，用于验证高层策略）
        """
        reward = 0.0
        for _ in range(low_steps):
            # Pursuer: PD控制向子目标
            a_p = 1.6 * (subgoal - self.p) - 0.9 * self.vp
            a_p = self._clip_norm(a_p, self.a_max_p)
            self.vp = self._clip_norm(self.vp + a_p * self.dt, self.v_max_p)
            self.p, self.vp = self._move_with_collisions(self.p, self.vp)
            
            # Evader: AI控制
            a_e = self._evader_acc()
            self.ve = self._clip_norm(self.ve + a_e * self.dt, self.v_max_e)
            self.e, self.ve = self._move_with_collisions(self.e, self.ve)
            
            self.p_history.append(self.p.copy())
            self.e_history.append(self.e.copy())
        
        self.step_n += 1
        d = np.linalg.norm(self.e - self.p)
        
        done = False
        captured = False
        info = {}
        
        # 捕获
        if d < self.catch_radius:
            done = True
            captured = True
            reward = 100.0
            info["captured"] = True
        # 超时
        elif self.step_n >= self.max_steps:
            done = True
            reward = -20.0
            info["timeout"] = True
        else:
            # 距离奖励
            reward = -0.5 - 0.1 * d
        
        info["distance"] = d
        return self._get_obs(), reward, done, info

    
    def step_continuous(self, angle: float, subgoal_dist: float = 4.5, low_steps: int = 8):
        """
        连续动作高层step
        action: 角度 [-pi, pi]，相对于pursuer→evader方向的偏移
        """
        rel = self.e - self.p
        alpha = math.atan2(rel[1], rel[0])
        theta = alpha + angle
        subgoal = self.p + subgoal_dist * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        return self.step_low(subgoal, low_steps)


# ============== 连续动作高层策略（简单Actor-Critic） ==============

class ContinuousHighLevelPolicy(nn.Module):
    """连续动作高层策略网络"""
    def __init__(self, obs_dim=8, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, 1)  # 角度均值
        self.log_std = nn.Parameter(torch.zeros(1))  # 可学习标准差
        self.value = nn.Linear(hidden, 1)
    
    def forward(self, obs):
        h = self.shared(obs)
        mu = torch.tanh(self.mu(h)) * math.pi  # [-pi, pi]
        std = torch.exp(self.log_std.clamp(-2, 1))
        v = self.value(h)
        return mu, std, v
    
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, std, v = self.forward(obs)
            if deterministic:
                return mu.item(), v.item()
            else:
                dist = torch.distributions.Normal(mu, std)
                a = dist.sample()
                return a.clamp(-math.pi, math.pi).item(), v.item()


def obs_to_tensor(obs: Dict, device) -> torch.Tensor:
    """将观察字典转为张量"""
    arr = np.array([
        obs["distance"] / 15.0,  # 归一化
        obs["bearing"] / math.pi,
        obs["heading_p"] / math.pi,
        obs["heading_e"] / math.pi,
        obs["speed_p"] / 1.5,
        obs["speed_e"] / 1.5,
        obs["d_obs_p"] / 10.0,
        obs["d_obs_e"] / 10.0,
    ], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


# ============== PPO训练 ==============

def train_ppo(
    env: SimplePursuitEnv,
    policy: ContinuousHighLevelPolicy,
    device,
    episodes: int = 1000,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    update_epochs: int = 4,
    subgoal_dist: float = 4.5,
    low_steps: int = 8,
):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    cap_history = []
    ret_history = []
    
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        
        states, actions, rewards, values, log_probs = [], [], [], [], []
        
        while not done:
            obs_t = obs_to_tensor(obs, device)
            mu, std, v = policy(obs_t)
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample()
            log_p = dist.log_prob(a).sum()
            
            a_val = a.clamp(-math.pi, math.pi).item()
            obs_next, r, done, info = env.step_continuous(a_val, subgoal_dist, low_steps)
            
            states.append(obs_t)
            actions.append(a)
            rewards.append(r)
            values.append(v.item())
            log_probs.append(log_p)
            
            obs = obs_next
            ep_ret += r
        
        # 计算GAE
        captured = info.get("captured", False)
        cap_history.append(1.0 if captured else 0.0)
        ret_history.append(ep_ret)
        
        # Bootstrap value
        if not done:
            with torch.no_grad():
                _, _, v_last = policy(obs_to_tensor(obs, device))
                values.append(v_last.item())
        else:
            values.append(0.0)
        
        # GAE计算
        advs = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] - values[t]
            gae = delta + gamma * lam * gae
            advs.insert(0, gae)
        
        returns = [advs[t] + values[t] for t in range(len(advs))]
        
        # PPO更新
        states_t = torch.cat(states, dim=0)
        actions_t = torch.cat(actions, dim=0)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=device)
        old_log_probs_t = torch.stack(log_probs)
        
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
        
        for _ in range(update_epochs):
            mu, std, v = policy(states_t)
            dist = torch.distributions.Normal(mu.squeeze(), std)
            log_probs_new = dist.log_prob(actions_t.squeeze())
            
            ratio = torch.exp(log_probs_new - old_log_probs_t.detach())
            surr1 = ratio * advs_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs_t
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * ((v.squeeze() - returns_t) ** 2).mean()
            entropy = dist.entropy().mean()
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        
        if ep % 50 == 0:
            recent_cap = np.mean(cap_history[-50:])
            recent_ret = np.mean(ret_history[-50:])
            print(f"[PPO] ep={ep:4d} cap={recent_cap:.1%} ret={recent_ret:.1f}")
    
    return cap_history, ret_history


# ============== 可视化 ==============

def plot_curves(cap_hist, ret_hist, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 捕获率
    window = 50
    smoothed_cap = np.convolve(cap_hist, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed_cap)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Capture Rate")
    axes[0].set_title("Capture Rate (smoothed)")
    axes[0].grid(True, alpha=0.3)
    
    # 回报
    smoothed_ret = np.convolve(ret_hist, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed_ret)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Return")
    axes[1].set_title("Episode Return (smoothed)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def visualize_episode(env, policy, device, save_dir, deterministic=True):
    obs = env.reset()
    done = False
    
    while not done:
        obs_t = obs_to_tensor(obs, device)
        a, _ = policy.act(obs_t, deterministic=deterministic)
        obs, r, done, info = env.step_continuous(a)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 障碍物
    for obs_rect in env.obstacles:
        rect = Rectangle(
            (obs_rect.cx - obs_rect.w/2, obs_rect.cy - obs_rect.h/2),
            obs_rect.w, obs_rect.h,
            facecolor='gray', edgecolor='black', alpha=0.5
        )
        ax.add_patch(rect)
    
    # 轨迹
    p_traj = np.array(env.p_history)
    e_traj = np.array(env.e_history)
    
    ax.plot(p_traj[:, 0], p_traj[:, 1], 'b-', lw=2, label='Pursuer', alpha=0.7)
    ax.plot(e_traj[:, 0], e_traj[:, 1], 'r-', lw=2, label='Evader', alpha=0.7)
    
    ax.scatter(p_traj[0, 0], p_traj[0, 1], c='blue', s=100, marker='o', zorder=5)
    ax.scatter(e_traj[0, 0], e_traj[0, 1], c='red', s=100, marker='o', zorder=5)
    ax.scatter(p_traj[-1, 0], p_traj[-1, 1], c='blue', s=150, marker='*', zorder=5)
    ax.scatter(e_traj[-1, 0], e_traj[-1, 1], c='red', s=150, marker='*', zorder=5)
    
    captured = info.get("captured", False)
    status = "CAPTURED" if captured else "ESCAPED"
    ax.set_title(f"Episode: {status}, Final dist={info['distance']:.2f}m")
    ax.set_xlim(-env.half - 1, env.half + 1)
    ax.set_ylim(-env.half - 1, env.half + 1)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "episode_viz.png"), dpi=150)
    plt.close()


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--subgoal-dist", type=float, default=4.5)
    parser.add_argument("--low-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="high_tabular/results_cont")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = SimplePursuitEnv(seed=args.seed)
    policy = ContinuousHighLevelPolicy().to(device)

    print("=" * 60)
    print("Continuous High-Level PPO Training")
    print(f"  Episodes: {args.episodes}")
    print(f"  Subgoal distance: {args.subgoal_dist}m")
    print(f"  Low-level steps: {args.low_steps}")
    print("=" * 60)

    cap_hist, ret_hist = train_ppo(
        env, policy, device,
        episodes=args.episodes,
        lr=args.lr,
        subgoal_dist=args.subgoal_dist,
        low_steps=args.low_steps,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    
    final_cap = np.mean(cap_hist[-100:]) if len(cap_hist) >= 100 else np.mean(cap_hist)
    print("-" * 60)
    print(f"Training complete!")
    print(f"Final capture rate: {final_cap:.1%}")
    print("-" * 60)

    plot_curves(cap_hist, ret_hist, args.save_dir)
    visualize_episode(env, policy, device, args.save_dir)
    
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "cap_hist": cap_hist,
        "ret_hist": ret_hist,
    }, os.path.join(args.save_dir, "checkpoint.pth"))
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
