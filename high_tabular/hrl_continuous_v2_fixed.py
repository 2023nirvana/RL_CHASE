#!/usr/bin/env python3
"""
连续动作高层HRL v2 - 修复策略崩溃问题
改进:
1. 学习率cosine衰减
2. log_std最小值限制（保持探索）
3. Early stopping + 最佳模型保存
4. 更强的entropy bonus
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
    """优化版追逃环境"""
    def __init__(self, world_size=15.0, catch_radius=1.0, max_steps=400, dt=0.1, seed=0):
        self.world_size = world_size
        self.half = world_size / 2
        self.catch_radius = catch_radius
        self.escape_dist = 27.0
        self.max_steps = max_steps
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        
        self.v_max_p = 1.1
        self.v_max_e = 1.35
        self.a_max_p = 1.8
        self.a_max_e = 0.55
        
        self.boundary_margin = 3.0
        self.subgoal_len = 4.5
        self.low_k = 8
        
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
        
        self.step_count = 0
        self.pos_history = []

    def _clip_norm(self, v, max_n):
        n = np.linalg.norm(v)
        return v if n <= max_n else v * (max_n / n)

    def _is_valid(self, p):
        if abs(p[0]) > self.half or abs(p[1]) > self.half:
            return False
        for obs in self.obstacles:
            if obs.contains(p):
                return False
        return True

    def _move(self, pos, vel):
        new_p = pos + vel * self.dt
        new_p = np.clip(new_p, -self.half, self.half)
        for obs in self.obstacles:
            if obs.contains(new_p):
                return pos.copy(), vel * 0.0
        return new_p, vel

    def _nearest_obs_dist(self, p):
        return min(obs.dist(p) for obs in self.obstacles)

    def _boundary_dist(self, p):
        return min(self.half - abs(p[0]), self.half - abs(p[1]))

    def _is_near_boundary(self, p):
        return self._boundary_dist(p) < self.boundary_margin

    def _spawn(self):
        for _ in range(100):
            p = self.rng.uniform(-self.half + 1, self.half - 1, size=2).astype(np.float32)
            if self._is_valid(p):
                return p
        return np.array([0.0, -5.0], dtype=np.float32)

    def _evader_acc(self):
        rel = self.p - self.e
        d = np.linalg.norm(rel)
        if d < 0.1:
            return self.rng.uniform(-1, 1, size=2).astype(np.float32) * self.a_max_e
        
        flee_dir = -rel / d
        
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
        
        if self._is_near_boundary(self.e):
            center_dir = -self.e / (np.linalg.norm(self.e) + 1e-6)
            flee_dir = 0.5 * flee_dir + 0.5 * center_dir
        
        flee_dir += self.rng.uniform(-0.2, 0.2, size=2)
        norm = np.linalg.norm(flee_dir)
        if norm > 0:
            flee_dir /= norm
        return flee_dir * self.a_max_e

    def reset(self):
        self.step_count = 0
        self.pos_history = []
        
        angle = self.rng.uniform(0, 2*np.pi)
        r = self.rng.uniform(6, 9)
        self.p = np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
        
        self.e = self._spawn()
        while np.linalg.norm(self.e - self.p) < 4:
            self.e = self._spawn()
        
        self.vp = np.zeros(2, dtype=np.float32)
        self.ve = np.zeros(2, dtype=np.float32)
        
        return self._get_obs()

    def _get_obs(self):
        rel = self.e - self.p
        dist = np.linalg.norm(rel)
        rel_norm = rel / (dist + 1e-6)
        
        vp_n = np.linalg.norm(self.vp)
        ve_n = np.linalg.norm(self.ve)
        
        if vp_n > 0.01:
            pdir = self.vp / vp_n
            cos_angle = np.clip(np.dot(pdir, rel_norm), -1, 1)
        else:
            cos_angle = 0.0
        
        bd = self._boundary_dist(self.p) / self.half
        
        obs = np.array([
            rel[0] / self.half, rel[1] / self.half,
            self.vp[0] / self.v_max_p, self.vp[1] / self.v_max_p,
            self.ve[0] / self.v_max_e, self.ve[1] / self.v_max_e,
            dist / (2 * self.half),
            cos_angle,
            bd
        ], dtype=np.float32)
        return obs

    def step(self, angle_offset):
        """angle_offset: [-pi, pi]"""
        rel = self.e - self.p
        base_angle = math.atan2(rel[1], rel[0])
        target_angle = base_angle + angle_offset
        
        subgoal = self.p + self.subgoal_len * np.array([
            math.cos(target_angle), math.sin(target_angle)
        ], dtype=np.float32)
        
        subgoal = np.clip(subgoal, -self.half + 0.5, self.half - 0.5)
        for obs in self.obstacles:
            if obs.contains(subgoal):
                subgoal = self.p + 0.3 * rel / (np.linalg.norm(rel) + 1e-6)
        
        init_dist = np.linalg.norm(self.p - self.e)
        
        reward = 0.0
        captured = False
        timeout = False
        
        for _ in range(self.low_k):
            to_sg = subgoal - self.p
            d_sg = np.linalg.norm(to_sg)
            if d_sg < 0.3:
                break
            
            pd_a = (to_sg / d_sg) * self.a_max_p * 1.2
            pd_a = self._clip_norm(pd_a, self.a_max_p)
            
            self.vp += pd_a * self.dt
            self.vp = self._clip_norm(self.vp, self.v_max_p)
            self.p, self.vp = self._move(self.p, self.vp)
            
            acc_e = self._evader_acc()
            self.ve += acc_e * self.dt
            self.ve = self._clip_norm(self.ve, self.v_max_e)
            self.e, self.ve = self._move(self.e, self.ve)
            
            self.step_count += 1
            
            if np.linalg.norm(self.p - self.e) < self.catch_radius:
                captured = True
                break
            
            if self.step_count >= self.max_steps:
                timeout = True
                break
        
        final_dist = np.linalg.norm(self.p - self.e)
        done = captured or timeout
        
        # ========== 奖励设计 ==========
        if captured:
            time_bonus = (self.max_steps - self.step_count) / self.max_steps
            reward = 100 + 50 * time_bonus
        elif timeout:
            reward = -20
        else:
            dist_change = init_dist - final_dist
            reward += dist_change * 5.0
            
            rel_to_e = self.e - self.p
            if np.linalg.norm(self.vp) > 0.01 and np.linalg.norm(rel_to_e) > 0.01:
                pdir = self.vp / np.linalg.norm(self.vp)
                edir = rel_to_e / np.linalg.norm(rel_to_e)
                cos_a = np.dot(pdir, edir)
                reward += cos_a * 0.5
            
            if self._is_near_boundary(self.p):
                reward -= 0.5
            
            if final_dist < 3.0:
                reward += (3 - final_dist) * 0.3
            
            self.pos_history.append(self.p.copy())
            if len(self.pos_history) > 20:
                self.pos_history.pop(0)
            if len(self.pos_history) >= 10:
                recent = np.array(self.pos_history[-10:])
                spread = np.std(recent, axis=0).mean()
                if spread < 0.5:
                    reward -= 1.0
        
        return self._get_obs(), reward, done, {"captured": captured}


class PolicyNetFixed(nn.Module):
    """修复版策略网络 - 限制log_std最小值"""
    def __init__(self, obs_dim=9, hidden=128, log_std_min=-1.0):
        super().__init__()
        self.log_std_min = log_std_min
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.value_head = nn.Linear(hidden, 1)
    
    def forward(self, x):
        h = self.shared(x)
        mu = torch.tanh(self.mu_head(h)) * math.pi
        # 关键修复：限制log_std最小值，保持探索能力
        std = torch.exp(self.log_std.clamp(self.log_std_min, 0.5))
        v = self.value_head(h)
        return mu, std, v


def obs_to_tensor(obs, device):
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def cosine_lr(step, total_steps, lr_init, lr_min=1e-5):
    """Cosine学习率衰减"""
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def train_ppo_fixed(env, episodes=4000, lr=1e-4, gamma=0.99, lam=0.95, 
                    clip_eps=0.2, update_epochs=4, hidden=128,
                    save_dir=None, device='cuda', patience=1000, 
                    entropy_coef=0.02, log_std_min=-1.0):
    """
    修复版PPO训练
    - Cosine学习率衰减
    - Early stopping with patience
    - 更强的entropy bonus
    """
    policy = PolicyNetFixed(hidden=hidden, log_std_min=log_std_min).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    cap_hist = []
    ret_hist = []
    best_cap = 0.0
    best_ep = 0
    no_improve_count = 0
    
    for ep in range(1, episodes + 1):
        # Cosine学习率衰减
        current_lr = cosine_lr(ep, episodes, lr, lr_min=1e-5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
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
            
            # 更强的entropy bonus
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        
        # 统计与保存
        if ep % 50 == 0:
            recent_cap = np.mean(cap_hist[-50:])
            recent_ret = np.mean(ret_hist[-50:])
            log_std_val = policy.log_std.item()
            print(f"[PPO] ep={ep:5d} cap={recent_cap:.1%} ret={recent_ret:.1f} "
                  f"lr={current_lr:.2e} log_std={log_std_val:.2f}")
            
            # 保存最佳模型
            if recent_cap > best_cap and save_dir:
                best_cap = recent_cap
                best_ep = ep
                no_improve_count = 0
                torch.save({
                    'policy': policy.state_dict(),
                    'episode': ep,
                    'capture_rate': recent_cap,
                }, os.path.join(save_dir, 'best.pth'))
                print(f"  -> Saved best model (cap={recent_cap:.1%})")
            else:
                no_improve_count += 50
            
            # Early stopping
            if no_improve_count >= patience and ep > 2000:
                print(f"\nEarly stopping at ep={ep}, no improvement for {no_improve_count} episodes")
                print(f"Best capture rate: {best_cap:.1%} at ep={best_ep}")
                break
    
    return cap_hist, ret_hist, best_cap, best_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='high_tabular/results_v2_fixed')
    parser.add_argument('--patience', type=int, default=1000, help='Early stopping patience')
    parser.add_argument('--entropy-coef', type=float, default=0.02)
    parser.add_argument('--log-std-min', type=float, default=-1.0)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 60)
    print("Continuous High-Level PPO V2 - FIXED (防止策略崩溃)")
    print(f"  Episodes: {args.episodes}")
    print(f"  LR: {args.lr} (cosine decay)")
    print(f"  Entropy coef: {args.entropy_coef}")
    print(f"  log_std_min: {args.log_std_min}")
    print(f"  Early stopping patience: {args.patience}")
    print("=" * 60)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = PursuitEnvV2(seed=args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    cap_hist, ret_hist, best_cap, best_ep = train_ppo_fixed(
        env, 
        episodes=args.episodes,
        lr=args.lr,
        hidden=args.hidden,
        save_dir=args.save_dir,
        device=device,
        patience=args.patience,
        entropy_coef=args.entropy_coef,
        log_std_min=args.log_std_min
    )
    
    print(f"\nTraining complete!")
    print(f"Best capture rate: {best_cap:.1%} at episode {best_ep}")
    
    # 保存训练曲线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    window = 100
    cap_smooth = np.convolve(cap_hist, np.ones(window)/window, mode='valid')
    ret_smooth = np.convolve(ret_hist, np.ones(window)/window, mode='valid')
    
    ax1.plot(cap_smooth * 100)
    ax1.axhline(best_cap * 100, color='r', linestyle='--', label=f'Best: {best_cap:.1%}')
    ax1.set_ylabel('Capture Rate (%)')
    ax1.set_title('Capture Rate (smoothed)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(ret_smooth)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Return')
    ax2.set_title('Episode Return (smoothed)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curve.png'), dpi=150)
    print(f"Saved training curve to {args.save_dir}/training_curve.png")


if __name__ == "__main__":
    main()
