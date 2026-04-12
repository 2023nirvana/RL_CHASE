#!/usr/bin/env python3
"""
第三步: 真实AUV动力学 + V4 NN底层控制器
========================================

架构:
  高层策略 (PPO) → 角度偏移 → 子目标生成 → V4 NN → 真实AUV动力学

关键改进:
- 追击者使用真实AUV动力学 (LowLevelAUV2DEnvV2Continuous)
- V4 NN作为底层控制器
- 逃跑者使用简化动力学
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 导入真实AUV环境
from core.environment.low_level_env_2d_v2_continuous import LowLevelAUV2DEnvV2Continuous


# ==================== V4 NN底层控制器 ====================
class GaussianActor(nn.Module):
    """V4训练使用的Actor网络"""
    def __init__(self, obs_dim=8, act_dim=3, hidden=(64, 64)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.actor_body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, x):
        h = self.actor_body(x)
        mu = torch.tanh(self.mu_head(h))
        return mu


def load_v4_nn(checkpoint_path, device):
    """加载V4 NN底层控制器"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    obs_dim = ckpt.get('obs_dim', 8)
    act_dim = ckpt.get('act_dim', 3)
    
    # 从config获取hidden size
    config = ckpt.get('config', {})
    network_cfg = config.get('network', {})
    feature_cfg = network_cfg.get('feature_extractor', {})
    hidden = tuple(feature_cfg.get('hidden_sizes', [64, 64]))
    
    model = GaussianActor(obs_dim, act_dim, hidden)
    
    # 加载权重
    state_dict = ckpt['model']
    model_dict = {}
    for k, v in state_dict.items():
        if k.startswith('actor_body') or k.startswith('mu_head') or k == 'log_std':
            model_dict[k] = v
    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"Loaded V4 NN: obs_dim={obs_dim}, act_dim={act_dim}, hidden={hidden}")
    return model


# ==================== 障碍物定义 ====================
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


# ==================== 追逃环境 ====================
class PursuitEnvRealAUV:
    """
    使用真实AUV动力学的追逃环境
    
    - 追击者: 真实AUV动力学 + V4 NN控制器
    - 逃跑者: 简化2D积分动力学
    """
    
    def __init__(self, nn_policy, device, 
                 world_size=15.0, catch_radius=1.0, max_steps=6000,
                 subgoal_dist=0.5, low_steps=50, seed=42):
        """
        Args:
            nn_policy: V4 NN底层控制器
            device: torch device
            world_size: 世界大小
            catch_radius: 捕获半径
            max_steps: 最大底层步数（每步0.02秒，6000步=120秒）
            subgoal_dist: 子目标距离（V4训练范围~0.7m）
            low_steps: 每个高层动作执行多少底层步
        """
        self.device = device
        self.nn_policy = nn_policy
        
        self.world_size = world_size
        self.half = world_size / 2
        self.catch_radius = catch_radius
        self.max_steps = max_steps
        self.subgoal_dist = subgoal_dist
        self.low_steps = low_steps
        
        self.rng = np.random.default_rng(seed)
        
        # 逃跑者参数（简化动力学）
        # 注意: 追踪者AUV实际速度约0.2-0.4 m/s，逃跑者需要更慢才能追上
        self.v_max_e = 0.18  # 从1.35降到0.18
        self.a_max_e = 0.10  # 从0.55降到0.10
        # dt_evader与AUV环境同步！
        self.dt_evader = 0.02  # 与AUV的dt一致
        
        self.boundary_margin = 3.0
        
        # 障碍物
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
        
        # 创建真实AUV环境（用于追击者）
        self.auv_env = LowLevelAUV2DEnvV2Continuous(
            target_distance=subgoal_dist,
            seed=seed
        )
        
        # 状态变量
        self.t = 0
        self.prev_positions = []
        self.e = None  # 逃跑者位置
        self.ve = None  # 逃跑者速度
        self.last_dist = None
        self.current_subgoal = None

    
    # ========== 辅助方法 ==========
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
    
    def _move_evader(self, pos, vel):
        """移动逃跑者（简化动力学）"""
        new_pos = pos + vel * self.dt_evader
        new_pos = np.clip(new_pos, -self.half, self.half)
        for obs in self.obstacles:
            if obs.contains(new_pos):
                return pos.copy(), vel * 0.0
        return new_pos, vel
    
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
    
    def _get_pursuer_pos(self):
        """获取追击者位置（从AUV环境）"""
        return self.auv_env.inner.dynamics.eta[:2].copy()
    
    def _get_pursuer_vel(self):
        """获取追击者速度（从AUV环境）"""
        return self.auv_env.inner.dynamics.nu[:2].copy()
    
    def _get_pursuer_yaw(self):
        """获取追击者航向"""
        return self.auv_env.inner.dynamics.eta[5]
    
    def _set_auv_target(self, target_pos):
        """设置AUV的目标位置"""
        self.auv_env.inner.target_state = np.array([
            target_pos[0], target_pos[1], self.auv_env.fixed_z,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
    
    def _evader_acc(self):
        """逃跑者AI"""
        p = self._get_pursuer_pos()
        rel = p - self.e
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
        
        # 边界避让
        if self._is_near_boundary(self.e):
            center_dir = -self.e / (np.linalg.norm(self.e) + 1e-6)
            flee_dir = 0.5 * flee_dir + 0.5 * center_dir
        
        # 随机扰动
        flee_dir += self.rng.uniform(-0.2, 0.2, size=2)
        norm = np.linalg.norm(flee_dir)
        if norm > 0:
            flee_dir /= norm
        return flee_dir * self.a_max_e

    
    # ========== 环境接口 ==========
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        self.prev_positions = []
        
        # 初始化追击者（重置AUV环境）
        # 设置初始位置在世界边缘
        angle = self.rng.uniform(0, 2*np.pi)
        r = self.rng.uniform(6, 9)
        init_p = np.array([r * np.cos(angle), r * np.sin(angle)])
        
        # 重置AUV环境
        self.auv_env.reset()
        
        # 设置AUV初始位置
        self.auv_env.inner.dynamics.eta[0] = init_p[0]
        self.auv_env.inner.dynamics.eta[1] = init_p[1]
        self.auv_env.inner.dynamics.eta[2] = self.auv_env.fixed_z
        self.auv_env.inner.dynamics.nu[:] = 0.0
        
        # 初始化逃跑者
        self.e = self._spawn_valid_pos()
        while np.linalg.norm(self.e - init_p) < 4:
            self.e = self._spawn_valid_pos()
        self.ve = np.zeros(2, dtype=np.float32)
        
        # 设置追击者初始航向指向逃跑者
        rel = self.e - init_p
        init_yaw = np.arctan2(rel[1], rel[0])
        self.auv_env.inner.dynamics.eta[5] = init_yaw
        
        self.last_dist = np.linalg.norm(init_p - self.e)
        self.current_subgoal = None
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """高层策略的观测: 9D"""
        p = self._get_pursuer_pos()
        vp = self._get_pursuer_vel()
        
        rel = self.e - p
        dist = np.linalg.norm(rel)
        rel_norm = rel / (dist + 1e-6)
        
        # 速度信息
        vp_norm = np.linalg.norm(vp)
        ve_norm = np.linalg.norm(self.ve)
        
        # 追击角
        if vp_norm > 0.01:
            pursuit_dir = vp / vp_norm
            cos_angle = np.clip(np.dot(pursuit_dir, rel_norm), -1, 1)
        else:
            cos_angle = 0.0
        
        # 边界距离
        boundary_dist = self._boundary_dist(p) / self.half
        
        obs = np.array([
            rel[0] / self.half, rel[1] / self.half,
            vp[0] / 2.0, vp[1] / 2.0,  # AUV速度范围略大
            self.ve[0] / self.v_max_e, self.ve[1] / self.v_max_e,
            dist / (2 * self.half),
            cos_angle,
            boundary_dist
        ], dtype=np.float32)
        return obs
    
    def step(self, action):
        """
        高层动作: action是角度偏移 [-pi, pi]
        1. 计算子目标
        2. 用V4 NN底层控制器执行若干步
        3. 计算奖励
        """
        # 解析高层动作
        angle_offset = float(action[0]) if hasattr(action, '__len__') else float(action)
        
        p = self._get_pursuer_pos()
        rel = self.e - p
        base_angle = np.arctan2(rel[1], rel[0])
        target_angle = base_angle + angle_offset
        
        # 计算子目标
        subgoal = p + self.subgoal_dist * np.array([
            np.cos(target_angle), np.sin(target_angle)
        ], dtype=np.float32)
        
        # 确保子目标在合法区域
        subgoal = np.clip(subgoal, -self.half + 0.5, self.half - 0.5)
        for obs in self.obstacles:
            if obs.contains(subgoal):
                subgoal = p + 0.3 * rel / (np.linalg.norm(rel) + 1e-6)
        
        self.current_subgoal = subgoal.copy()
        
        # 设置AUV目标
        self._set_auv_target(subgoal)
        
        # 记录初始状态用于奖励计算
        init_dist = np.linalg.norm(p - self.e)
        
        # 使用V4 NN执行底层控制
        captured = False
        timeout = False
        
        for low_step in range(self.low_steps):
            # 获取AUV的观测（V4格式）
            auv_obs = self.auv_env._get_observation()
            
            # 检查是否已到达子目标
            subgoal_dist = np.linalg.norm(self._get_pursuer_pos() - subgoal)
            if subgoal_dist < 0.15:
                break
            
            # V4 NN推理
            with torch.no_grad():
                obs_t = torch.FloatTensor(auv_obs).unsqueeze(0).to(self.device)
                nn_action = self.nn_policy(obs_t).cpu().numpy()[0]
            
            # 执行AUV动作
            self.auv_env.step(nn_action)
            
            # 更新逃跑者
            acc_e = self._evader_acc()
            self.ve += acc_e * self.dt_evader
            self.ve = self._clip_norm(self.ve, self.v_max_e)
            self.e, self.ve = self._move_evader(self.e, self.ve)
            
            self.t += 1
            
            # 检查捕获
            p = self._get_pursuer_pos()
            if np.linalg.norm(p - self.e) < self.catch_radius:
                captured = True
                break
            
            if self.t >= self.max_steps:
                timeout = True
                break
        
        # 计算奖励
        reward = 0.0
        done = False
        
        p = self._get_pursuer_pos()
        final_dist = np.linalg.norm(p - self.e)
        
        if captured:
            time_bonus = (self.max_steps - self.t) / self.max_steps
            reward = 100.0 + 50.0 * time_bonus
            done = True
        elif timeout:
            reward = -20.0
            done = True
        else:
            # ========== 改进的奖励设计 ==========
            
            # 1. 距离变化奖励 (增大系数)
            dist_change = init_dist - final_dist
            reward += dist_change * 10.0  # 从5增到10
            
            # 2. 基于距离的势能惩罚 (鼓励保持近距离)
            # 距离越远惩罚越大，提供持续的学习信号
            gamma = 0.99
            potential_init = -init_dist
            potential_final = -final_dist
            shaping_reward = gamma * potential_final - potential_init
            reward += shaping_reward * 2.0  # 势能shaping
            
            # 3. 拦截角奖励 (增大系数)
            vp = self._get_pursuer_vel()
            rel_to_e = self.e - p
            if np.linalg.norm(vp) > 0.01 and np.linalg.norm(rel_to_e) > 0.01:
                pursuit_dir = vp / np.linalg.norm(vp)
                to_evader = rel_to_e / np.linalg.norm(rel_to_e)
                cos_angle = np.dot(pursuit_dir, to_evader)
                reward += cos_angle * 1.0  # 从0.5增到1.0
            
            # 4. 边界惩罚
            if self._is_near_boundary(p):
                reward -= 0.5
            
            # 5. 距离接近奖励 (增强)
            if final_dist < 5.0:
                # 距离越近奖励越高
                reward += (5.0 - final_dist) * 0.5
            if final_dist < 2.0:
                # 非常近时额外奖励
                reward += (2.0 - final_dist) * 1.0
            
            # 6. 循环检测惩罚
            self.prev_positions.append(p.copy())
            if len(self.prev_positions) > 20:
                self.prev_positions.pop(0)
            if len(self.prev_positions) >= 10:
                recent = np.array(self.prev_positions[-10:])
                spread = np.std(recent, axis=0).mean()
                if spread < 0.5:
                    reward -= 1.0
        
        self.last_dist = final_dist
        
        return self._get_obs(), reward, done, False, {'captured': captured, 'dist': final_dist}



# ==================== 高层策略网络 ====================
class PolicyNet(nn.Module):
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
        std = torch.exp(self.log_std.clamp(self.log_std_min, 0.5))
        v = self.value_head(h)
        return mu, std, v


def obs_to_tensor(obs, device):
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def cosine_lr(step, total_steps, lr_init, lr_min=1e-5):
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def train_ppo(env, episodes=4000, lr=1e-4, gamma=0.99, lam=0.95,
              clip_eps=0.2, update_epochs=4, hidden=128,
              save_dir=None, device='cuda', patience=1000,
              entropy_coef=0.02):
    """PPO训练（带early stopping和cosine lr decay）"""
    
    policy = PolicyNet(hidden=hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
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
        
        obs, _ = env.reset()
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
            obs_next, r, terminated, truncated, info = env.step(a_val)
            done = terminated or truncated
            
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
        
        # 跳过太短的episode
        if len(states) < 2:
            continue
        
        # PPO更新
        states_t = torch.cat(states, dim=0)
        actions_t = torch.cat(actions, dim=0)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=device)
        old_log_probs_t = torch.stack(log_probs)
        
        adv_std = advs_t.std()
        if adv_std > 1e-6:
            advs_t = (advs_t - advs_t.mean()) / (adv_std + 1e-8)
        else:
            advs_t = advs_t - advs_t.mean()
        
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
            
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        
        # 打印进度
        if ep % 50 == 0:
            recent_cap = np.mean(cap_hist[-50:])
            recent_ret = np.mean(ret_hist[-50:])
            log_std_val = policy.log_std.item()
            print(f"[PPO] ep={ep:5d} cap={recent_cap:.1%} ret={recent_ret:.1f} "
                  f"lr={current_lr:.2e} log_std={log_std_val:.2f}")
            
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
                print(f"\nEarly stopping at ep={ep}")
                print(f"Best capture rate: {best_cap:.1%} at ep={best_ep}")
                break
    
    return cap_hist, ret_hist, best_cap, best_ep, policy



# ==================== 可视化 ====================
def visualize_episode(env, policy, device, save_path=None):
    import matplotlib.patches as patches
    
    obs, _ = env.reset()
    
    pursuer_traj = [env._get_pursuer_pos().copy()]
    evader_traj = [env.e.copy()]
    subgoals = []
    
    done = False
    while not done:
        obs_t = obs_to_tensor(obs, device)
        with torch.no_grad():
            mu, _, _ = policy(obs_t)
            action = mu.cpu().numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        pursuer_traj.append(env._get_pursuer_pos().copy())
        evader_traj.append(env.e.copy())
        if env.current_subgoal is not None:
            subgoals.append(env.current_subgoal.copy())
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制边界
    boundary = patches.Rectangle((-env.half, -env.half), 2*env.half, 2*env.half,
                                   fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(boundary)
    
    # 绘制障碍物
    for obs_rect in env.obstacles:
        rect = patches.Rectangle((obs_rect.cx - obs_rect.w/2, obs_rect.cy - obs_rect.h/2), 
                                   obs_rect.w, obs_rect.h,
                                   fill=True, color='gray', alpha=0.5)
        ax.add_patch(rect)
    
    # 绘制轨迹
    p_traj = np.array(pursuer_traj)
    e_traj = np.array(evader_traj)
    
    ax.plot(p_traj[:, 0], p_traj[:, 1], 'b-', linewidth=2, label='Pursuer (Real AUV)', alpha=0.8)
    ax.plot(e_traj[:, 0], e_traj[:, 1], 'r-', linewidth=2, label='Evader', alpha=0.8)
    
    # 绘制子目标
    if subgoals:
        sg = np.array(subgoals)
        ax.scatter(sg[:, 0], sg[:, 1], c='green', s=20, alpha=0.3, label='Subgoals')
    
    # 绘制起终点
    ax.scatter(p_traj[0, 0], p_traj[0, 1], c='blue', s=100, marker='o', zorder=5)
    ax.scatter(e_traj[0, 0], e_traj[0, 1], c='red', s=100, marker='o', zorder=5)
    ax.scatter(p_traj[-1, 0], p_traj[-1, 1], c='blue', s=150, marker='*', zorder=5)
    ax.scatter(e_traj[-1, 0], e_traj[-1, 1], c='red', s=150, marker='*', zorder=5)
    
    ax.set_xlim(-env.half - 1, env.half + 1)
    ax.set_ylim(-env.half - 1, env.half + 1)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f"Pursuit with Real AUV + V4 NN (Captured: {info.get('captured', False)})")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    plt.close()


def evaluate(env, policy, device, n_episodes=100):
    """评估策略"""
    captures = 0
    total_steps = 0
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            obs_t = obs_to_tensor(obs, device)
            with torch.no_grad():
                mu, _, _ = policy(obs_t)
                action = mu.cpu().numpy()[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        if info.get('captured', False):
            captures += 1
        total_steps += steps
    
    return captures / n_episodes, total_steps / n_episodes


# ==================== Main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='checkpoints/hrl_v3_real_auv')
    parser.add_argument('--nn-checkpoint', type=str, default='low_near/v4_tight/checkpoints/best.pth')
    parser.add_argument('--subgoal-dist', type=float, default=0.5, help='Subgoal distance (within V4 range)')
    parser.add_argument('--low-steps', type=int, default=50, help='Low-level steps per high-level action')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 60)
    print("第三步: 真实AUV动力学 + V4 NN底层控制器")
    print(f"  Episodes: {args.episodes}")
    print(f"  Subgoal dist: {args.subgoal_dist}m")
    print(f"  Low steps: {args.low_steps}")
    print(f"  NN checkpoint: {args.nn_checkpoint}")
    print("=" * 60)
    
    # 加载V4 NN
    nn_policy = load_v4_nn(args.nn_checkpoint, device)
    
    # 创建环境
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = PursuitEnvRealAUV(
        nn_policy, device,
        subgoal_dist=args.subgoal_dist,
        low_steps=args.low_steps,
        seed=args.seed
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'best.pth')
    
    if args.eval_only or args.visualize:
        # 加载已训练的高层策略
        checkpoint = torch.load(save_path, map_location=device)
        policy = PolicyNet(hidden=args.hidden).to(device)
        policy.load_state_dict(checkpoint['policy'])
        policy.eval()
        
        if args.visualize:
            for i in range(5):
                viz_path = os.path.join(args.save_dir, f'viz_{i}.png')
                visualize_episode(env, policy, device, viz_path)
        else:
            rate, avg_steps = evaluate(env, policy, device, n_episodes=100)
            print(f"Evaluation: capture rate={rate:.1%}, avg steps={avg_steps:.1f}")
    else:
        # 训练
        cap_hist, ret_hist, best_cap, best_ep, policy = train_ppo(
            env,
            episodes=args.episodes,
            lr=args.lr,
            hidden=args.hidden,
            save_dir=args.save_dir,
            device=device
        )
        
        print(f"\nTraining complete!")
        print(f"Best capture rate: {best_cap:.1%} at episode {best_ep}")
        
        # 最终评估
        policy.eval()
        final_rate, avg_steps = evaluate(env, policy, device, n_episodes=100)
        print(f"Final evaluation: {final_rate:.1%}, avg steps={avg_steps:.1f}")
        
        # 保存训练曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        window = 100
        if len(cap_hist) > window:
            cap_smooth = np.convolve(cap_hist, np.ones(window)/window, mode='valid')
            ret_smooth = np.convolve(ret_hist, np.ones(window)/window, mode='valid')
            ax1.plot(cap_smooth * 100)
            ax2.plot(ret_smooth)
        else:
            ax1.plot(np.array(cap_hist) * 100)
            ax2.plot(ret_hist)
        
        ax1.axhline(best_cap * 100, color='r', linestyle='--', label=f'Best: {best_cap:.1%}')
        ax1.set_ylabel('Capture Rate (%)')
        ax1.set_title('Training Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'training_curve.png'), dpi=150)
        
        # 生成可视化
        for i in range(3):
            viz_path = os.path.join(args.save_dir, f'final_viz_{i}.png')
            visualize_episode(env, policy, device, viz_path)
