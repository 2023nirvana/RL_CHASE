#!/usr/bin/env python3
"""
连续动作高层HRL v2 - 模仿tabular_pursuit_qlearning的奖励设计
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class RectObstacle:
    cx: float
    cy: float
    w: float
    h: float
    
    def contains(self, p: np.ndarray) -> bool:
        return abs(p[0] - self.cx) <= self.w/2 and abs(p[1] - self.cy) <= self.h/2
    
    def dist(self, p: np.ndarray) -> float:
        dx = max(abs(p[0] - self.cx) - self.w/2, 0)
        dy = max(abs(p[1] - self.cy) - self.h/2, 0)
        return math.sqrt(dx*dx + dy*dy)


class PursuitEnvV2:
    """
    优化版追逃环境 - 模仿tabular_pursuit_qlearning的奖励设计
    """
    def __init__(self, world_size=15.0, catch_radius=1.0, max_steps=400, dt=0.1, seed=0):
        self.world_size = world_size
        self.half = world_size / 2
        self.catch_radius = catch_radius
        self.escape_dist = 27.0
        self.max_steps = max_steps
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        
        # 速度/加速度（与tabular一致）
        self.v_max_p = 1.1
        self.v_max_e = 1.35
        self.a_max_p = 1.8
        self.a_max_e = 0.55
        
        self.boundary_margin = 3.0
        self.subgoal_len = 4.5
        self.low_k = 8
        
        # 障碍物（与tabular一致）
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
        
        # 历史记录（用于绕圈检测）
        self.history_len = 12
        self.position_history = []
        self.prev_action = 0.0
        
        self.reset()

    
    def _clip_norm(self, v, max_n):
        n = np.linalg.norm(v)
        return v if n <= max_n else v * (max_n / n)
    
    def _is_valid_pos(self, p):
        if abs(p[0]) > self.half or abs(p[1]) > self.half:
            return False
        for obs in self.obstacles:
            if obs.contains(p):
                return False
        return True
    
    def _move_with_collisions(self, pos, vel):
        new_pos = pos + vel * self.dt
        new_pos = np.clip(new_pos, -self.half, self.half)
        for obs in self.obstacles:
            if obs.contains(new_pos):
                return pos.copy(), vel * 0.0
        return new_pos, vel
    
    def _nearest_obstacle_dist(self, p):
        return min(obs.dist(p) for obs in self.obstacles)
    
    def _boundary_dist(self, p):
        return min(self.half - abs(p[0]), self.half - abs(p[1]))
    
    def _is_near_boundary(self, p):
        return self._boundary_dist(p) < self.boundary_margin
    
    def _spawn_valid_pos(self):
        for _ in range(100):
            p = self.rng.uniform(-self.half + 1, self.half - 1, size=2).astype(np.float32)
            if self._is_valid_pos(p):
                return p
        return np.array([0.0, -5.0], dtype=np.float32)
    
    def _evader_acc(self):
        """逃跑者AI（与tabular一致）"""
        rel = self.p - self.e
        d = np.linalg.norm(rel)
        if d < 0.1:
            return self.rng.uniform(-1, 1, size=2).astype(np.float32) * self.a_max_e
        
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
        
        flee_dir += self.rng.uniform(-0.2, 0.2, size=2)
        norm = np.linalg.norm(flee_dir)
        if norm > 0:
            flee_dir /= norm
        return flee_dir * self.a_max_e

    
    def reset(self):
        self.p = self._spawn_valid_pos()
        self.e = self._spawn_valid_pos()
        while np.linalg.norm(self.e - self.p) < 4.0:
            self.e = self._spawn_valid_pos()
        
        self.vp = np.zeros(2, dtype=np.float32)
        self.ve = np.zeros(2, dtype=np.float32)
        self.step_n = 0
        self.prev_d = float(np.linalg.norm(self.e - self.p))
        self.prev_action = 0.0
        self.position_history = [self.p.copy()]
        
        self.p_history = [self.p.copy()]
        self.e_history = [self.e.copy()]
        
        return self._get_obs()
    
    def _get_obs(self):
        """丰富的观察空间（模仿tabular的state信息）"""
        rel = self.e - self.p
        d = np.linalg.norm(rel)
        bearing = math.atan2(rel[1], rel[0])
        
        vp_norm = np.linalg.norm(self.vp)
        heading_p = math.atan2(self.vp[1], self.vp[0]) if vp_norm > 0.01 else 0.0
        ve_norm = np.linalg.norm(self.ve)
        heading_e = math.atan2(self.ve[1], self.ve[0]) if ve_norm > 0.01 else 0.0
        
        # 追逐者速度方向与目标方向的夹角
        heading_err = bearing - heading_p
        while heading_err > math.pi: heading_err -= 2*math.pi
        while heading_err < -math.pi: heading_err += 2*math.pi
        
        # evader逃跑方向与p->e方向的夹角
        flee_rel = heading_e - bearing
        while flee_rel > math.pi: flee_rel -= 2*math.pi
        while flee_rel < -math.pi: flee_rel += 2*math.pi
        
        d_obs_p = self._nearest_obstacle_dist(self.p)
        d_obs_e = self._nearest_obstacle_dist(self.e)
        d_bound_e = self._boundary_dist(self.e)
        
        return {
            "distance": d,
            "bearing": bearing,
            "heading_err": heading_err,
            "flee_rel": flee_rel,
            "speed_p": vp_norm,
            "speed_e": ve_norm,
            "d_obs_p": d_obs_p,
            "d_obs_e": d_obs_e,
            "d_bound_e": d_bound_e,
            "pos_p": self.p.copy(),
            "pos_e": self.e.copy(),
        }

    
    def step(self, action: float):
        """
        连续动作step，action是角度偏移 [-pi, pi]
        奖励设计模仿tabular_pursuit_qlearning
        """
        # 计算子目标
        rel = self.e - self.p
        alpha = math.atan2(rel[1], rel[0])
        theta = alpha + action
        subgoal = self.p + self.subgoal_len * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        
        # 执行低层控制
        for _ in range(self.low_k):
            a_p = 1.6 * (subgoal - self.p) - 0.9 * self.vp
            a_p = self._clip_norm(a_p, self.a_max_p)
            self.vp = self._clip_norm(self.vp + a_p * self.dt, self.v_max_p)
            self.p, self.vp = self._move_with_collisions(self.p, self.vp)
            
            a_e = self._evader_acc()
            self.ve = self._clip_norm(self.ve + a_e * self.dt, self.v_max_e)
            self.e, self.ve = self._move_with_collisions(self.e, self.ve)
            
            self.p_history.append(self.p.copy())
            self.e_history.append(self.e.copy())
        
        self.step_n += 1
        d = float(np.linalg.norm(self.e - self.p))
        d_obs_e = self._nearest_obstacle_dist(self.e)
        d_obs_p = self._nearest_obstacle_dist(self.p)
        
        # ============== 奖励计算（模仿tabular） ==============
        reward = -0.25  # 基础时间惩罚
        
        # 1. 动作变化惩罚（连续版本）
        action_diff = abs(action - self.prev_action)
        if action_diff > math.pi:
            action_diff = 2*math.pi - action_diff
        if action_diff > 2.0:  # 大于115度
            reward -= 0.5 * action_diff
        
        # 2. 绕圈检测惩罚
        self.position_history.append(self.p.copy())
        if len(self.position_history) > self.history_len:
            self.position_history.pop(0)
        
        if len(self.position_history) >= 8:
            loop_penalty = 0.0
            for old_pos in self.position_history[:-4]:
                dist_to_old = float(np.linalg.norm(self.p - old_pos))
                if dist_to_old < 2.0:
                    loop_penalty += 0.4 * (2.0 - dist_to_old)
            reward -= min(loop_penalty, 2.5)
        
        # 3. 距离改进奖励
        improve = self.prev_d - d
        reward += 6.0 * improve
        
        # 4. 地形利用奖励（逼evader向障碍物）
        if d_obs_e < 3.0:
            reward += 1.2 * (3.0 - d_obs_e)
        
        # 5. 拦截几何奖励
        ve_norm = np.linalg.norm(self.ve)
        if ve_norm > 0.01:
            e_dir = self.ve / ve_norm
            p_ahead = self.p - self.e
            intercept_proj = float(np.dot(p_ahead, e_dir))
            if intercept_proj > 0:
                reward += 0.6 * min(intercept_proj, 4.0)
        
        # 6. 追逐者靠近障碍物掩护
        if d_obs_p < 2.5 and d < 8.0:
            reward += 0.3 * (2.5 - d_obs_p)
        
        self.prev_action = action
        self.prev_d = d
        
        # ============== 终止条件 ==============
        done = False
        info = {"captured": False, "distance": d}
        
        if d <= self.catch_radius:
            done = True
            info["captured"] = True
            # 捕获奖励
            reward += 100.0
            if d_obs_e < 2.0:
                reward += 80.0 * (2.0 - d_obs_e)
            elif d_obs_e < 4.0:
                reward += 30.0 * (4.0 - d_obs_e) / 2.0
            # 边界惩罚
            bd = self._boundary_dist(self.e)
            if bd < self.boundary_margin:
                reward -= 60.0 * (1.0 - bd / self.boundary_margin)
            # 中央奖励
            center_dist = float(np.linalg.norm(self.e))
            if center_dist < 8.0:
                reward += 40.0 * (1.0 - center_dist / 8.0)
                
        elif d >= self.escape_dist or self.step_n >= self.max_steps:
            done = True
            reward -= 50.0
        
        return self._get_obs(), reward, done, info


# ============== 策略网络 ==============

class PolicyNet(nn.Module):
    def __init__(self, obs_dim=9, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.value = nn.Linear(hidden, 1)
    
    def forward(self, obs):
        h = self.shared(obs)
        mu = torch.tanh(self.mu(h)) * math.pi
        std = torch.exp(self.log_std.clamp(-2, 1))
        v = self.value(h)
        return mu, std, v
    
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, std, v = self.forward(obs)
            if deterministic:
                return mu.item(), v.item()
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample().clamp(-math.pi, math.pi)
            return a.item(), v.item()


def obs_to_tensor(obs, device):
    arr = np.array([
        obs["distance"] / 15.0,
        obs["bearing"] / math.pi,
        obs["heading_err"] / math.pi,
        obs["flee_rel"] / math.pi,
        obs["speed_p"] / 1.5,
        obs["speed_e"] / 1.5,
        obs["d_obs_p"] / 10.0,
        obs["d_obs_e"] / 10.0,
        obs["d_bound_e"] / 10.0,
    ], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


# ============== PPO训练 ==============

def train_ppo(env, policy, device, episodes=2000, gamma=0.99, lam=0.95, lr=3e-4, 
              clip_eps=0.2, update_epochs=4):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    cap_hist, ret_hist = [], []
    
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
            obs_next, r, done, info = env.step(a_val)
            
            states.append(obs_t)
            actions.append(a)
            rewards.append(r)
            values.append(v.item())
            log_probs.append(log_p)
            
            obs = obs_next
            ep_ret += r
        
        cap_hist.append(1.0 if info.get("captured") else 0.0)
        ret_hist.append(ep_ret)
        values.append(0.0)
        
        # GAE
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
            recent_cap = np.mean(cap_hist[-50:])
            recent_ret = np.mean(ret_hist[-50:])
            print(f"[PPO] ep={ep:4d} cap={recent_cap:.1%} ret={recent_ret:.1f}")
    
    return cap_hist, ret_hist


# ============== 可视化和Main ==============

def plot_curves(cap_hist, ret_hist, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    window = 50
    if len(cap_hist) > window:
        smoothed_cap = np.convolve(cap_hist, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed_cap)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Capture Rate")
    axes[0].set_title("Capture Rate (smoothed)")
    axes[0].grid(True, alpha=0.3)
    
    if len(ret_hist) > window:
        smoothed_ret = np.convolve(ret_hist, np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed_ret)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Return")
    axes[1].set_title("Episode Return (smoothed)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def visualize_episode(env, policy, device, save_dir):
    obs = env.reset()
    done = False
    while not done:
        obs_t = obs_to_tensor(obs, device)
        a, _ = policy.act(obs_t, deterministic=True)
        obs, r, done, info = env.step(a)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for obs_rect in env.obstacles:
        rect = Rectangle((obs_rect.cx - obs_rect.w/2, obs_rect.cy - obs_rect.h/2),
                         obs_rect.w, obs_rect.h, facecolor='gray', edgecolor='black', alpha=0.5)
        ax.add_patch(rect)
    
    p_traj = np.array(env.p_history)
    e_traj = np.array(env.e_history)
    ax.plot(p_traj[:, 0], p_traj[:, 1], 'b-', lw=2, label='Pursuer', alpha=0.7)
    ax.plot(e_traj[:, 0], e_traj[:, 1], 'r-', lw=2, label='Evader', alpha=0.7)
    ax.scatter(p_traj[0, 0], p_traj[0, 1], c='blue', s=100, marker='o', zorder=5)
    ax.scatter(e_traj[0, 0], e_traj[0, 1], c='red', s=100, marker='o', zorder=5)
    ax.scatter(p_traj[-1, 0], p_traj[-1, 1], c='blue', s=150, marker='*', zorder=5)
    ax.scatter(e_traj[-1, 0], e_traj[-1, 1], c='red', s=150, marker='*', zorder=5)
    
    status = "CAPTURED" if info.get("captured") else "ESCAPED"
    ax.set_title(f"Episode: {status}, Final dist={info['distance']:.2f}m")
    ax.set_xlim(-env.half - 1, env.half + 1)
    ax.set_ylim(-env.half - 1, env.half + 1)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "episode_viz.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="high_tabular/results_v2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = PursuitEnvV2(seed=args.seed)
    policy = PolicyNet().to(device)

    print("=" * 60)
    print("Continuous High-Level PPO V2 (模仿tabular奖励)")
    print(f"  Episodes: {args.episodes}")
    print(f"  Pursuer: v_max=1.1, a_max=1.8")
    print(f"  Evader:  v_max=1.35, a_max=0.55")
    print("=" * 60)

    cap_hist, ret_hist = train_ppo(env, policy, device, episodes=args.episodes, lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    final_cap = np.mean(cap_hist[-100:]) if len(cap_hist) >= 100 else np.mean(cap_hist)
    print("-" * 60)
    print(f"Training complete! Final capture rate: {final_cap:.1%}")
    print("-" * 60)

    plot_curves(cap_hist, ret_hist, args.save_dir)
    visualize_episode(env, policy, device, args.save_dir)
    
    torch.save({"policy_state_dict": policy.state_dict(), "cap_hist": cap_hist, "ret_hist": ret_hist},
               os.path.join(args.save_dir, "checkpoint.pth"))
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
