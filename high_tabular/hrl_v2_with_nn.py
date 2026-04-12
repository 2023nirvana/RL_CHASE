#!/usr/bin/env python3
"""
V2 + V4 NN底层
==============
用训练好的V4 NN替换简化PD控制器
"""
import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict

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


# ============== V4 NN 策略 ==============

class LowLevelNNPolicy(nn.Module):
    """V4底层NN策略"""
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


class PursuitEnvWithNN:
    """
    追逃环境 - 使用V4 NN作为底层控制器
    
    关键适配：
    - V4训练范围约0.7m，需要调整subgoal_dist
    - V4输入是相对子目标的位置/速度误差
    - 需要更多low_steps（NN执行较慢）
    """
    def __init__(self, nn_policy, device, world_size=15.0, catch_radius=1.0, 
                 max_steps=400, dt=0.1, seed=0,
                 subgoal_dist=0.6, low_steps=80):
        self.nn_policy = nn_policy
        self.device = device
        
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
        self.subgoal_dist = subgoal_dist  # V4训练范围内
        self.low_steps = low_steps
        
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
        """逃跑者AI"""
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

    
    def _get_nn_obs(self, subgoal):
        """
        构造V4 NN底层控制器的8D观测
        V4训练时的格式:
        [pos_error_body_x/5, pos_error_body_y/5, heading_error/pi, target_angle/pi,
         vx_body/2, vy_body/2, yaw_rate/2, distance/5]
        """
        # 位置误差(世界坐标)
        pos_error_world = subgoal - self.p
        distance = np.linalg.norm(pos_error_world)
        
        # heading (yaw)
        yaw = self.heading_p
        
        # 目标角度和航向误差
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        heading_error = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))
        
        # 位置误差转换到体坐标系
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
        pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
        
        # 速度转换到体坐标系
        vx_body = cos_yaw * self.vp[0] + sin_yaw * self.vp[1]
        vy_body = -sin_yaw * self.vp[0] + cos_yaw * self.vp[1]
        
        # 偏航角速度（用有限差分估计，初始为0）
        yaw_rate = getattr(self, 'yaw_rate_est', 0.0)
        
        obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            vx_body / 2.0,
            vy_body / 2.0,
            yaw_rate / 2.0,
            distance / 5.0,
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)
    
    def _execute_nn_action(self, nn_action):
        """将NN的3D输出转为2D加速度"""
        # V4 NN输出: [surge, sway, yaw_rate] 均为[-1,1]范围
        # surge控制前进/后退推力，sway控制侧向推力
        surge = nn_action[0]  # 直接使用，稍后转换
        sway = nn_action[1]
        yaw_cmd = nn_action[2]
        
        # 保存旧heading
        old_heading = self.heading_p
        
        # 更新heading (yaw_cmd控制角速度)
        yaw_rate = yaw_cmd * 1.0  # 最大角速度1 rad/s
        self.heading_p += yaw_rate * self.dt
        self.heading_p = np.arctan2(np.sin(self.heading_p), np.cos(self.heading_p))
        
        # 估计yaw_rate用于下一步观测
        self.yaw_rate_est = (self.heading_p - old_heading) / self.dt
        
        # 推力转为世界坐标系加速度
        # surge沿heading方向，sway垂直于heading
        cos_h = np.cos(self.heading_p)
        sin_h = np.sin(self.heading_p)
        
        ax = surge * self.a_max_p * cos_h - sway * self.a_max_p * 0.3 * sin_h
        ay = surge * self.a_max_p * sin_h + sway * self.a_max_p * 0.3 * cos_h
        
        return np.array([ax, ay], dtype=np.float32)
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        self.prev_positions = []
        
        # 初始化位置
        angle = self.rng.uniform(0, 2*np.pi)
        r = self.rng.uniform(6, 9)
        self.p = np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
        
        self.e = self._spawn_valid_pos()
        while np.linalg.norm(self.e - self.p) < 4:
            self.e = self._spawn_valid_pos()
        
        self.vp = np.zeros(2, dtype=np.float32)
        self.ve = np.zeros(2, dtype=np.float32)
        
        # 初始heading指向逃跑者
        rel = self.e - self.p
        self.heading_p = np.arctan2(rel[1], rel[0])
        self.yaw_rate_est = 0.0  # 初始yaw_rate为0
        
        self.last_dist = np.linalg.norm(self.p - self.e)
        self.current_subgoal = None
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """高层策略的观测: 9D"""
        rel = self.e - self.p
        dist = np.linalg.norm(rel)
        rel_norm = rel / (dist + 1e-6)
        
        # 速度信息
        vp_norm = np.linalg.norm(self.vp)
        ve_norm = np.linalg.norm(self.ve)
        
        # 追击角
        if vp_norm > 0.01:
            pursuit_dir = self.vp / vp_norm
            cos_angle = np.clip(np.dot(pursuit_dir, rel_norm), -1, 1)
        else:
            cos_angle = 0.0
        
        # 边界距离
        boundary_dist = self._boundary_dist(self.p) / self.half
        
        obs = np.array([
            rel[0] / self.half, rel[1] / self.half,
            self.vp[0] / self.v_max_p, self.vp[1] / self.v_max_p,
            self.ve[0] / self.v_max_e, self.ve[1] / self.v_max_e,
            dist / (2 * self.half),
            cos_angle,
            boundary_dist
        ], dtype=np.float32)
        return obs

    
    def step(self, action):
        """
        高层动作: action是一个角度偏移 [-pi, pi]
        1. 计算子目标
        2. 用NN底层控制器执行若干步到达/接近子目标
        3. 计算奖励
        """
        # 解析高层动作并计算子目标
        angle_offset = float(action[0]) if hasattr(action, '__len__') else float(action)
        
        rel = self.e - self.p
        base_angle = np.arctan2(rel[1], rel[0])
        target_angle = base_angle + angle_offset
        
        subgoal = self.p + self.subgoal_dist * np.array([
            np.cos(target_angle), np.sin(target_angle)
        ], dtype=np.float32)
        
        # 确保子目标在合法区域
        subgoal = np.clip(subgoal, -self.half + 0.5, self.half - 0.5)
        for obs in self.obstacles:
            if obs.contains(subgoal):
                subgoal = self.p + 0.3 * (self.e - self.p) / (np.linalg.norm(self.e - self.p) + 1e-6)
        
        self.current_subgoal = subgoal.copy()
        
        # 记录初始状态用于奖励计算
        init_dist = np.linalg.norm(self.p - self.e)
        init_pos = self.p.copy()
        
        # 使用NN底层控制器执行
        captured = False
        collision = False
        timeout = False
        
        for low_step in range(self.low_steps):
            # 检查是否已接近子目标
            dist_to_subgoal = np.linalg.norm(self.p - subgoal)
            if dist_to_subgoal < 0.15:
                break
            
            # 构造NN观测
            nn_obs = self._get_nn_obs(subgoal)
            
            # NN推理
            with torch.no_grad():
                nn_obs_tensor = torch.FloatTensor(nn_obs).unsqueeze(0).to(self.device)
                nn_action = self.nn_policy(nn_obs_tensor).cpu().numpy()[0]
            
            # 将NN输出转为加速度
            acc = self._execute_nn_action(nn_action)
            acc = self._clip_norm(acc, self.a_max_p)
            
            # 更新追击者
            self.vp += acc * self.dt
            self.vp = self._clip_norm(self.vp, self.v_max_p)
            self.p, self.vp = self._move_with_collisions(self.p, self.vp)
            
            # 更新逃跑者
            acc_e = self._evader_acc()
            self.ve += acc_e * self.dt
            self.ve = self._clip_norm(self.ve, self.v_max_e)
            self.e, self.ve = self._move_with_collisions(self.e, self.ve)
            
            self.t += 1
            
            # 检查捕获
            if np.linalg.norm(self.p - self.e) < self.catch_radius:
                captured = True
                break
            
            if self.t >= self.max_steps:
                timeout = True
                break
        
        # 计算奖励（与V2相同的奖励设计）
        reward = 0.0
        done = False
        
        final_dist = np.linalg.norm(self.p - self.e)
        
        if captured:
            reward = 100.0
            done = True
        elif timeout:
            reward = -20.0
            done = True
        else:
            # 距离变化奖励
            dist_change = init_dist - final_dist
            reward += dist_change * 5.0
            
            # 拦截角奖励
            rel_to_e = self.e - self.p
            if np.linalg.norm(self.vp) > 0.01 and np.linalg.norm(rel_to_e) > 0.01:
                pursuit_dir = self.vp / np.linalg.norm(self.vp)
                to_evader = rel_to_e / np.linalg.norm(rel_to_e)
                cos_angle = np.dot(pursuit_dir, to_evader)
                reward += cos_angle * 0.5
            
            # 边界惩罚
            if self._is_near_boundary(self.p):
                reward -= 0.5
            
            # 距离接近奖励
            if final_dist < 3.0:
                reward += (3.0 - final_dist) * 0.3
            
            # 循环检测惩罚
            self.prev_positions.append(self.p.copy())
            if len(self.prev_positions) > 20:
                self.prev_positions.pop(0)
            if len(self.prev_positions) >= 10:
                recent = np.array(self.prev_positions[-10:])
                spread = np.std(recent, axis=0).mean()
                if spread < 0.5:
                    reward -= 1.0
        
        self.last_dist = final_dist
        
        return self._get_obs(), reward, done, False, {'captured': captured, 'dist': final_dist}



# ==================== 策略网络 ====================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=9, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        h = self.fc(x)
        mu = torch.tanh(self.mu(h)) * np.pi
        std = torch.exp(torch.clamp(self.log_std, -2, 0.5))
        return mu, std.expand_as(mu)


class ValueNet(nn.Module):
    def __init__(self, obs_dim=9, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x):
        return self.fc(x)


def collect_episode(env, policy, device):
    obs, _ = env.reset()
    states, actions, rewards, log_probs = [], [], [], []
    done = False
    
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, std = policy(obs_t)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        
        action_np = action.cpu().numpy()[0]
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        
        states.append(obs)
        actions.append(action_np)
        rewards.append(reward)
        log_probs.append(log_prob.item())
        
        obs = next_obs
    
    return states, actions, rewards, log_probs, info.get('captured', False)


def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_ppo(env, episodes=2000, lr=1e-4, gamma=0.99, clip_eps=0.15, 
              epochs=6, hidden=128, eval_interval=50, save_path=None, device='cuda'):
    
    policy = PolicyNet(hidden=hidden).to(device)
    value = ValueNet(hidden=hidden).to(device)
    
    opt_policy = optim.Adam(policy.parameters(), lr=lr)
    opt_value = optim.Adam(value.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt_policy, step_size=500, gamma=0.8)
    
    best_rate = 0
    capture_history = []
    
    for ep in range(episodes):
        # 收集数据
        states, actions, rewards, old_log_probs, captured = collect_episode(env, policy, device)
        returns = compute_returns(rewards, gamma)
        
        capture_history.append(1 if captured else 0)
        if len(capture_history) > 100:
            capture_history.pop(0)
        
        # 跳过太短的episode（至少需要2步）
        if len(states) < 2:
            continue
        
        # PPO更新
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.FloatTensor(np.array(actions)).to(device)
        if len(actions_t.shape) == 1:
            actions_t = actions_t.unsqueeze(-1)
        returns_t = torch.FloatTensor(returns).to(device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(device)
        
        with torch.no_grad():
            values = value(states_t).squeeze(-1)
            if values.dim() == 0:
                values = values.unsqueeze(0)
            advantages = returns_t - values
            adv_std = advantages.std()
            if adv_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            else:
                advantages = advantages - advantages.mean()
        
        for _ in range(epochs):
            mu, std = policy(states_t)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(actions_t).sum(-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            entropy = dist.entropy().mean()
            policy_loss = policy_loss - 0.01 * entropy
            
            opt_policy.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt_policy.step()
            
            value_pred = value(states_t).squeeze(-1)
            if value_pred.dim() == 0:
                value_pred = value_pred.unsqueeze(0)
            value_loss = F.mse_loss(value_pred, returns_t)
            
            opt_value.zero_grad()
            value_loss.backward()
            opt_value.step()
        
        scheduler.step()
        
        # 打印
        if (ep + 1) % eval_interval == 0:
            rate = sum(capture_history) / len(capture_history)
            print(f"Episode {ep+1}/{episodes}, Capture Rate: {rate:.1%}, "
                  f"Reward: {sum(rewards):.1f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
            if rate > best_rate and save_path:
                best_rate = rate
                torch.save({
                    'policy': policy.state_dict(),
                    'value': value.state_dict(),
                    'rate': rate,
                    'episode': ep+1
                }, save_path)
                print(f"  -> Saved best model (rate={rate:.1%})")
    
    return policy, value, best_rate



# ==================== 可视化 ====================
def visualize_episode(env, policy, device, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    obs, _ = env.reset()
    
    pursuer_traj = [env.p.copy()]
    evader_traj = [env.e.copy()]
    subgoals = []
    
    done = False
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = policy(obs_t)
            action = mu.cpu().numpy()[0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        pursuer_traj.append(env.p.copy())
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
    for obs in env.obstacles:
        rect = patches.Rectangle((obs.cx - obs.w/2, obs.cy - obs.h/2), 
                                   obs.w, obs.h,
                                   fill=True, color='gray', alpha=0.5)
        ax.add_patch(rect)
    
    # 绘制轨迹
    p_traj = np.array(pursuer_traj)
    e_traj = np.array(evader_traj)
    
    ax.plot(p_traj[:, 0], p_traj[:, 1], 'b-', linewidth=2, label='Pursuer', alpha=0.8)
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
    ax.set_title(f"Pursuit with NN Low-Level (Captured: {info.get('captured', False)})")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    plt.close()


def evaluate(env, policy, device, n_episodes=100):
    captures = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = policy(obs_t)
                action = mu.cpu().numpy()[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get('captured', False):
            captures += 1
    return captures / n_episodes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='checkpoints/hrl_nn')
    parser.add_argument('--nn-checkpoint', type=str, default='low_near/v4_tight/checkpoints/best.pth')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载NN底层策略
    nn_checkpoint = args.nn_checkpoint
    nn_policy = load_nn_policy(nn_checkpoint, device)
    print(f"Loaded NN policy from {nn_checkpoint}")
    
    # 创建环境
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    env = PursuitEnvWithNN(nn_policy, device=device, seed=args.seed)
    
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
            rate = evaluate(env, policy, device, n_episodes=100)
            print(f"Evaluation capture rate: {rate:.1%}")
    else:
        # 训练
        print(f"Training with NN low-level controller...")
        print(f"Episodes: {args.episodes}, LR: {args.lr}, Hidden: {args.hidden}")
        
        policy, value, best_rate = train_ppo(
            env, 
            episodes=args.episodes,
            lr=args.lr,
            hidden=args.hidden,
            save_path=save_path,
            device=device
        )
        
        print(f"\nTraining complete. Best capture rate: {best_rate:.1%}")
        
        # 最终评估
        policy.eval()
        final_rate = evaluate(env, policy, device, n_episodes=100)
        print(f"Final evaluation: {final_rate:.1%}")
        
        # 生成可视化
        for i in range(3):
            viz_path = os.path.join(args.save_dir, f'final_viz_{i}.png')
            visualize_episode(env, policy, device, viz_path)
