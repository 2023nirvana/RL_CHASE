#!/usr/bin/env python3
"""
HRL Pursuit-Evasion: Tabular High-Level + NN Low-Level on Real AUV Dynamics

架构:
- 高层: Tabular Q-learning, 8个离散方向动作
- 低层: 预训练NN策略 (v4_tight)
- 环境: 真实AUV动力学 (LowLevelAUV2DPursuitEvasionEnv)
"""

import argparse
import math
import random
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_pursuit_evasion import LowLevelAUV2DPursuitEvasionEnv


# ============== NN Low-Level Policy ==============

class LowLevelNNPolicy(nn.Module):
    """低层NN策略"""
    
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
    """加载预训练NN策略"""
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



# ============== Obstacle Definition ==============

@dataclass
class RectObstacle:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def x_min(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def x_max(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def y_min(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def y_max(self) -> float:
        return self.cy + self.h / 2.0

    def contains(self, p: np.ndarray) -> bool:
        return self.x_min <= p[0] <= self.x_max and self.y_min <= p[1] <= self.y_max

    def distance(self, p: np.ndarray) -> float:
        dx = max(self.x_min - p[0], 0.0, p[0] - self.x_max)
        dy = max(self.y_min - p[1], 0.0, p[1] - self.y_max)
        if dx == 0.0 and dy == 0.0:
            return -min(p[0] - self.x_min, self.x_max - p[0], p[1] - self.y_min, self.y_max - p[1])
        return math.sqrt(dx * dx + dy * dy)



# ============== HRL Environment ==============

class HRLPursuitEnv:
    """
    层次强化学习追逃环境
    
    - 基于LowLevelAUV2DPursuitEvasionEnv (真实AUV动力学)
    - 高层每K步输出子目标方向
    - 低层NN策略执行子目标追踪
    """
    
    def __init__(
        self,
        nn_policy: LowLevelNNPolicy,
        device: torch.device,
        seed: int = 0,
        # 高层参数
        subgoal_radius: float = 0.7,  # 子目标距离 (m)
        low_steps_per_high: int = 50,  # 每个高层动作的低层步数
        n_high_actions: int = 8,       # 高层动作数
        # 追逃环境参数
        world_size: float = 15.0,
        catch_radius: float = 1.0,
        init_distance_range: Tuple[float, float] = (6.0, 12.0),
        max_episode_seconds: float = 60.0,
    ):
        self.nn_policy = nn_policy
        self.device = device
        
        # 高层参数
        self.subgoal_radius = subgoal_radius
        self.low_steps_per_high = low_steps_per_high
        self.n_actions = n_high_actions
        
        # 创建底层AUV追逃环境
        self.env = LowLevelAUV2DPursuitEvasionEnv(
            init_distance_range=init_distance_range,
            world_size=world_size,
            catch_radius=catch_radius,
            max_episode_seconds=max_episode_seconds,
            curriculum_stage=0,  # 使用配置中可用的stage
        )
        self.env.seed(seed)
        
        self.world_size = world_size
        self.catch_radius = catch_radius
        self.boundary_margin = 3.0
        
        # 障碍物 (与tabular环境一致)
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
        
        # Episode状态
        self.high_step = 0
        self.max_high_steps = int(max_episode_seconds * 10 / low_steps_per_high)
        self.prev_distance = 0.0
        self.prev_action = 0
        self.position_history = []
        self.action_history = []
        
        # 轨迹记录 (用于可视化)
        self.pursuer_trajectory = []
        self.evader_trajectory = []
        self.subgoal_history = []


    # ========== 辅助方法 ==========
    
    def _get_pursuer_state(self) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """获取追逐者状态: (position, yaw, velocity, yaw_rate)"""
        eta = self.env.inner.dynamics.eta
        nu = self.env.inner.dynamics.nu
        return eta[:2].copy(), float(eta[5]), nu[:2].copy(), float(nu[5])
    
    def _get_evader_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取逃脱者状态: (position, velocity)"""
        return self.env.evader_pos.copy(), self.env.evader_vel.copy()

    def _nearest_obstacle_dist(self, p: np.ndarray) -> float:
        return min(obs.distance(p) for obs in self.obstacles)

    def _is_near_boundary(self, p: np.ndarray) -> bool:
        return abs(p[0]) > self.world_size - self.boundary_margin or \
               abs(p[1]) > self.world_size - self.boundary_margin

    def _boundary_dist(self, p: np.ndarray) -> float:
        return min(self.world_size - abs(p[0]), self.world_size - abs(p[1]))

    def _get_nn_obs(self, pursuer_pos: np.ndarray, pursuer_yaw: float,
                   pursuer_vel: np.ndarray, pursuer_yaw_rate: float,
                   target_pos: np.ndarray) -> np.ndarray:
        """构建NN策略的8维观测"""
        pos_error_world = target_pos - pursuer_pos
        distance = np.linalg.norm(pos_error_world)
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        heading_error = np.arctan2(np.sin(target_angle - pursuer_yaw),
                                   np.cos(target_angle - pursuer_yaw))
        
        cos_yaw = np.cos(pursuer_yaw)
        sin_yaw = np.sin(pursuer_yaw)
        pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
        pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
        
        obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            pursuer_vel[0] / 2.0,
            pursuer_vel[1] / 2.0,
            pursuer_yaw_rate / 2.0,
            distance / 5.0,
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)


    def _state_discrete(self) -> tuple:
        """高层状态离散化"""
        p_pos, p_yaw, p_vel, _ = self._get_pursuer_state()
        e_pos, e_vel = self._get_evader_state()
        
        rel = e_pos - p_pos
        d = float(np.linalg.norm(rel))

        # 8 distance bins
        d_edges = [1.0, 3.0, 5.0, 8.0, 11.0, 15.0, 20.0]
        d_bin = len(d_edges)
        for i, edge in enumerate(d_edges):
            if d < edge:
                d_bin = i
                break

        # 8-bin bearing to evader
        bearing = math.atan2(rel[1], rel[0])
        b_bin = int(((bearing + math.pi) / (2 * math.pi)) * 8.0) % 8

        # 4-bin heading error
        heading = math.atan2(p_vel[1] + 1e-9, p_vel[0] + 1e-9)
        heading_err = (bearing - heading + math.pi) % (2 * math.pi) - math.pi
        h_bin = int(((heading_err + math.pi) / (2 * math.pi)) * 4.0) % 4

        # 4-bin evader flee direction
        e_heading = math.atan2(e_vel[1] + 1e-9, e_vel[0] + 1e-9)
        flee_angle = (e_heading - bearing + math.pi) % (2 * math.pi) - math.pi
        ef_bin = int(((flee_angle + math.pi) / (2 * math.pi)) * 4.0) % 4

        # 3-bin obstacle proximities
        d_obs_e = self._nearest_obstacle_dist(e_pos)
        oe_bin = 0 if d_obs_e < 1.5 else (1 if d_obs_e < 4.0 else 2)
        
        d_obs_p = self._nearest_obstacle_dist(p_pos)
        op_bin = 0 if d_obs_p < 2.0 else (1 if d_obs_p < 5.0 else 2)
        
        # 2-bin evader near boundary
        eb_bin = 0 if self._is_near_boundary(e_pos) else 1
        
        # 4-bin previous action
        prev_a_bin = self.prev_action // 2

        return d_bin, b_bin, h_bin, ef_bin, oe_bin, op_bin, eb_bin, prev_a_bin


    # ========== Gym接口 ==========

    def reset(self) -> tuple:
        """重置环境"""
        self.env.reset()
        
        self.high_step = 0
        p_pos, _, _, _ = self._get_pursuer_state()
        e_pos, _ = self._get_evader_state()
        self.prev_distance = float(np.linalg.norm(e_pos - p_pos))
        
        self.prev_action = 0
        self.position_history = [p_pos.copy()]
        self.action_history = []
        
        # 清空轨迹记录
        self.pursuer_trajectory = [p_pos.copy()]
        self.evader_trajectory = [e_pos.copy()]
        self.subgoal_history = []
        
        return self._state_discrete()


    def step(self, action_idx: int):
        """执行一个高层动作"""
        # 获取当前状态
        p_pos, p_yaw, p_vel, p_yaw_rate = self._get_pursuer_state()
        e_pos, e_vel = self._get_evader_state()
        
        # 计算子目标方向 (相对于到逃脱者的方位角)
        rel = e_pos - p_pos
        bearing = math.atan2(rel[1], rel[0])
        theta = bearing + action_idx * (2.0 * math.pi / self.n_actions)
        subgoal = p_pos + self.subgoal_radius * np.array([math.cos(theta), math.sin(theta)])
        subgoal = np.clip(subgoal, -self.world_size + 0.5, self.world_size - 0.5)
        
        self.subgoal_history.append(subgoal.copy())
        
        # 执行低层NN控制
        captured = False
        for _ in range(self.low_steps_per_high):
            # 获取最新状态
            p_pos, p_yaw, p_vel, p_yaw_rate = self._get_pursuer_state()
            
            # 构建NN观测并执行动作
            nn_obs = self._get_nn_obs(p_pos, p_yaw, p_vel, p_yaw_rate, subgoal)
            with torch.no_grad():
                obs_t = torch.from_numpy(nn_obs).unsqueeze(0).to(self.device)
                action_t = self.nn_policy.get_action(obs_t, deterministic=True)
                nn_action = action_t.cpu().numpy().flatten()
            
            # 通过底层环境执行动作
            # 注意: 底层环境使用离散动作，需要映射连续动作
            # 这里直接调用底层的_apply_continuous_action逻辑
            self._apply_continuous_action(nn_action, subgoal)
            
            # 记录轨迹
            p_pos_new, _, _, _ = self._get_pursuer_state()
            e_pos_new, _ = self._get_evader_state()
            self.pursuer_trajectory.append(p_pos_new.copy())
            self.evader_trajectory.append(e_pos_new.copy())
            
            # 检查是否捕获
            dist = np.linalg.norm(p_pos_new - e_pos_new)
            if dist < self.catch_radius:
                captured = True
                break

        self.high_step += 1
        
        # 计算奖励
        p_pos, _, p_vel, _ = self._get_pursuer_state()
        e_pos, e_vel = self._get_evader_state()
        d = float(np.linalg.norm(e_pos - p_pos))
        d_obs_e = self._nearest_obstacle_dist(e_pos)
        evader_near_boundary = self._is_near_boundary(e_pos)
        
        reward = self._compute_reward(action_idx, d, d_obs_e, captured)
        
        # 更新历史
        self.position_history.append(p_pos.copy())
        self.action_history.append(action_idx)
        if len(self.position_history) > 15:
            self.position_history.pop(0)
            self.action_history.pop(0)
        self.prev_action = action_idx
        self.prev_distance = d
        
        # 检查终止
        done = False
        if captured:
            done = True
        elif d > 27.0 or self.high_step >= self.max_high_steps:
            done = True
        
        info = {
            "captured": captured,
            "distance": d,
            "capture_pos": e_pos.copy() if captured else None,
            "near_terrain": d_obs_e < 3.0 if captured else False,
            "near_boundary": evader_near_boundary if captured else False,
            "high_step": self.high_step,
        }
        
        return self._state_discrete(), float(reward), done, info


    def _apply_continuous_action(self, action: np.ndarray, subgoal_xy: np.ndarray):
        """执行连续动作并步进环境 (基于train_pe_high_level_continuous_hrl.py)"""
        action = np.clip(action, -1.0, 1.0)
        
        # 设置底层环境的目标
        self.env.inner.target_state[:3] = np.array([subgoal_xy[0], subgoal_xy[1], self.env.fixed_z])
        
        eta = self.env.inner.dynamics.eta
        nu = self.env.inner.dynamics.nu
        dt = self.env.inner.dt
        
        # 从配置读取动作参数
        action_cfg = self.env.inner.config.get("action_space", {})
        distance_cfg = action_cfg.get("action_distance", {})
        angle_cfg = action_cfg.get("action_angle_delta", {})
        surge_step = float(distance_cfg.get("surge", 0.03))
        sway_step = float(distance_cfg.get("sway", 0.03))
        yaw_delta = float(angle_cfg.get("yaw", 0.0))
        
        # 动作映射到期望位移
        linear_cmd = np.array([
            float(action[0]) * surge_step,
            float(action[1]) * sway_step,
            0.0
        ])
        desired_vel_body = linear_cmd / max(dt, 1e-6)
        
        # 目标附近减速
        slowdown_cfg = action_cfg.get("goal_slowdown", {})
        if slowdown_cfg.get("enabled", True):
            radius = float(slowdown_cfg.get("radius", 1.0))
            min_scale = float(slowdown_cfg.get("min_scale", 0.2))
            pos_error = np.linalg.norm(subgoal_xy - eta[:2])
            if radius > 1e-6:
                scale = max(min_scale, min(1.0, pos_error / radius))
                desired_vel_body = desired_vel_body * scale
        
        # 速度环控制
        current_vel_body = nu[:3]
        vel_error = desired_vel_body - current_vel_body
        ctl = self.env.inner.integrated_controller
        force_cmd = ctl.vel_kp * vel_error - ctl.vel_kd * current_vel_body
        force_cmd = np.clip(force_cmd, -ctl.max_force, ctl.max_force)
        
        # 姿态环控制
        yaw_current = eta[5]
        target_yaw = np.arctan2(np.sin(yaw_current + float(action[2]) * yaw_delta),
                                np.cos(yaw_current + float(action[2]) * yaw_delta))
        angle_target = np.array([target_yaw, 0.0, 0.0])
        angle_current = np.array([yaw_current, eta[4], eta[3]])
        omega_current = np.array([nu[5], nu[4], nu[3]])
        
        tau_attitude, _ = ctl.pid_controller.compute(angle_target, angle_current, omega_current, dt)
        thrust_command = ctl.thrust_allocator.allocate_with_attitude(force_cmd, tau_attitude)
        
        # 物理步进
        external_force = self.env.inner._compute_external_force()
        self.env.inner.dynamics.step(thrust_command, external_force)
        self.env.inner.water_current.step(dt)
        self.env.inner.current_step += 1
        self.env._enforce_2d_state()
        
        # 步进逃脱者
        self.env._step_evader()


    def _compute_reward(self, action_idx: int, d: float, d_obs_e: float, captured: bool) -> float:
        """计算奖励"""
        reward = -0.25  # 时间惩罚
        
        # 动作变化惩罚
        action_diff = min(abs(action_idx - self.prev_action),
                          self.n_actions - abs(action_idx - self.prev_action))
        if action_diff >= 3:
            reward -= 0.8 * action_diff
        elif action_diff >= 2:
            reward -= 0.3 * action_diff
        
        # 绕圈检测惩罚
        if len(self.position_history) >= 8:
            p_pos = self.position_history[-1] if self.position_history else np.zeros(2)
            loop_penalty = 0.0
            for old_pos in self.position_history[:-4]:
                dist_to_old = float(np.linalg.norm(p_pos - old_pos))
                if dist_to_old < 2.0:
                    loop_penalty += 0.5 * (2.0 - dist_to_old)
            if loop_penalty > 0:
                reward -= min(loop_penalty, 3.0)
        
        # 动作震荡惩罚
        if len(self.action_history) >= 4:
            recent = self.action_history[-4:]
            if recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]:
                reward -= 1.5
        
        # 距离改善奖励
        improve = self.prev_distance - d
        reward += 6.0 * improve
        
        # 地形利用奖励
        if d_obs_e < 3.0:
            reward += 1.2 * (3.0 - d_obs_e)
        
        # 捕获奖励
        if captured:
            reward += 100.0
            if d_obs_e < 2.0:
                reward += 80.0 * (2.0 - d_obs_e)
            elif d_obs_e < 4.0:
                reward += 15.0 * (4.0 - d_obs_e)
            
            e_pos, _ = self._get_evader_state()
            boundary_dist_e = self._boundary_dist(e_pos)
            if boundary_dist_e < self.boundary_margin:
                reward -= 60.0 * (1.0 - boundary_dist_e / self.boundary_margin)
            
            center_dist = float(np.linalg.norm(e_pos))
            if center_dist < 8.0:
                reward += 40.0 * (1.0 - center_dist / 8.0)
        
        elif d > 27.0 or self.high_step >= self.max_high_steps:
            reward -= 50.0
        
        return reward



# ============== Training ==============

def epsilon_by_episode(ep: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    return eps_end + (eps_start - eps_end) * math.exp(-ep / max(1, eps_decay))


def train_hrl(
    nn_policy: LowLevelNNPolicy,
    device: torch.device,
    episodes: int = 2000,
    alpha: float = 0.15,
    gamma: float = 0.98,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 800,
    seed: int = 0,
    subgoal_radius: float = 0.7,
    low_steps_per_high: int = 50,
):
    """Tabular Q-learning训练"""
    random.seed(seed)
    np.random.seed(seed)
    
    env = HRLPursuitEnv(
        nn_policy, device, seed=seed,
        subgoal_radius=subgoal_radius,
        low_steps_per_high=low_steps_per_high,
    )
    
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

        if ep % 20 == 0:
            recent_cap = float(np.mean(capture_history[-20:]))
            recent_ret = float(np.mean(reward_history[-20:]))
            print(f"[HRL] ep={ep:4d} eps={eps:.3f} cap={recent_cap:.2%} ret={recent_ret:.1f} states={len(q_table)}")

    return env, q_table, capture_history, reward_history



# ============== Visualization ==============

def plot_training_curves(cap_hist, ret_hist, save_dir, window=50):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = np.arange(1, len(cap_hist) + 1)
    
    def rolling_mean(arr, w):
        arr = np.array(arr)
        result = np.zeros(len(arr))
        for i in range(len(arr)):
            start = max(0, i - w + 1)
            result[i] = np.mean(arr[start:i+1])
        return result
    
    cap_smooth = rolling_mean(cap_hist, window)
    ret_smooth = rolling_mean(ret_hist, window)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.plot(episodes, cap_hist, alpha=0.3, color='blue')
    ax.plot(episodes, cap_smooth, color='blue', linewidth=2, label=f'{window}-ep avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Capture Rate')
    ax.set_title('Capture Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    ax = axes[1]
    ax.plot(episodes, ret_hist, alpha=0.3, color='green')
    ax.plot(episodes, ret_smooth, color='green', linewidth=2, label=f'{window}-ep avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('Episode Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def visualize_episode(env: HRLPursuitEnv, q_table: Dict, save_dir: str, seed: int = 999):
    """可视化一个episode"""
    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    
    def get_q(s):
        if s not in q_table:
            return np.zeros(env.n_actions, dtype=np.float32)
        return q_table[s]
    
    env.env.seed(seed)
    s = env.reset()
    done = False
    captured = False
    
    while not done:
        a = int(np.argmax(get_q(s)))
        s, _, done, info = env.step(a)
        captured = captured or bool(info.get("captured", False))
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制障碍物
    for obs in env.obstacles:
        rect = plt.Rectangle((obs.x_min, obs.y_min), obs.w, obs.h, color='gray', alpha=0.5)
        ax.add_patch(rect)
    
    # 绘制边界区域
    L = env.world_size
    margin = env.boundary_margin
    ax.fill_between([-L, L], [L-margin, L-margin], [L, L], color='red', alpha=0.1)
    ax.fill_between([-L, L], [-L, -L], [-L+margin, -L+margin], color='red', alpha=0.1)
    
    # 绘制轨迹
    p_traj = np.array(env.pursuer_trajectory)
    e_traj = np.array(env.evader_trajectory)
    
    ax.plot(p_traj[:, 0], p_traj[:, 1], 'b-', linewidth=2, label='Pursuer', alpha=0.7)
    ax.plot(e_traj[:, 0], e_traj[:, 1], 'r-', linewidth=2, label='Evader', alpha=0.7)
    
    # 起点和终点
    ax.scatter(p_traj[0, 0], p_traj[0, 1], c='blue', s=100, marker='o', zorder=5)
    ax.scatter(e_traj[0, 0], e_traj[0, 1], c='red', s=100, marker='o', zorder=5)
    ax.scatter(p_traj[-1, 0], p_traj[-1, 1], c='blue', s=150, marker='*', zorder=5)
    ax.scatter(e_traj[-1, 0], e_traj[-1, 1], c='red', s=150, marker='*', zorder=5)
    
    # 绘制子目标
    if env.subgoal_history:
        subgoals = np.array(env.subgoal_history)
        ax.scatter(subgoals[:, 0], subgoals[:, 1], c='green', s=30, marker='x', alpha=0.5, label='Subgoals')
    
    ax.set_xlim(-L - 1, L + 1)
    ax.set_ylim(-L - 1, L + 1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'{"CAPTURED" if captured else "ESCAPED"}', fontweight='bold',
                 color='green' if captured else 'red')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'episode_visualization.png'), dpi=150)
    plt.close()



# ============== Main ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="low_near/v4_tight/checkpoints/best.pth")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="high_tabular/results_hrl")
    parser.add_argument("--subgoal-radius", type=float, default=0.7)
    parser.add_argument("--low-steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载NN策略
    ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    print(f"Loading NN policy from: {ckpt_path}")
    nn_policy = load_nn_policy(ckpt_path, device)
    print("NN policy loaded!")

    print("=" * 60)
    print("HRL Pursuit-Evasion Training")
    print(f"  Episodes: {args.episodes}")
    print(f"  Subgoal radius: {args.subgoal_radius}m")
    print(f"  Low-level steps per high: {args.low_steps}")
    print("=" * 60)

    env, q_table, cap_hist, ret_hist = train_hrl(
        nn_policy, device,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        seed=args.seed,
        subgoal_radius=args.subgoal_radius,
        low_steps_per_high=args.low_steps,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    
    final_cap = float(np.mean(cap_hist[-50:])) if len(cap_hist) >= 50 else float(np.mean(cap_hist))
    print("-" * 60)
    print(f"Training complete!")
    print(f"Final capture rate: {final_cap:.2%}")
    print(f"Q-table states: {len(q_table)}")
    print("-" * 60)

    # 保存结果
    plot_training_curves(cap_hist, ret_hist, args.save_dir)
    visualize_episode(env, q_table, args.save_dir)
    
    import pickle
    with open(os.path.join(args.save_dir, "q_table.pkl"), "wb") as f:
        pickle.dump({"q_table": q_table, "cap_hist": cap_hist, "ret_hist": ret_hist}, f)
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
