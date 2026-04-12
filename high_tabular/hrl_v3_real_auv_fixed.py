#!/usr/bin/env python3
"""
第三步修正版: 真实AUV动力学 + V4 NN底层控制器
============================================

修正点(对齐2D_MDP_CN.pdf):
1. 高层动作: 输出子目标相对位移 (Δx, Δy)，不是角度
2. 高层观测: 添加加速度信息
3. 保持奖励设计与文档一致
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

from core.environment.low_level_env_2d_v2_continuous import LowLevelAUV2DEnvV2Continuous


# ==================== V4 NN底层控制器 ====================
class GaussianActor(nn.Module):
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
    ckpt = torch.load(checkpoint_path, map_location=device)
    obs_dim = ckpt.get('obs_dim', 8)
    act_dim = ckpt.get('act_dim', 3)
    
    config = ckpt.get('config', {})
    network_cfg = config.get('network', {})
    feature_cfg = network_cfg.get('feature_extractor', {})
    hidden = tuple(feature_cfg.get('hidden_sizes', [64, 64]))
    
    model = GaussianActor(obs_dim, act_dim, hidden)
    
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
        return abs(p[0] - self.cx) <= self.w/2 and abs(p[1] - self.cy) <= self.h/2
    
    def dist(self, p: np.ndarray) -> float:
        dx = max(abs(p[0] - self.cx) - self.w/2, 0)
        dy = max(abs(p[1] - self.cy) - self.h/2, 0)
        return math.sqrt(dx*dx + dy*dy)

    def line_intersects(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[bool, float, str]:
        """检测线段 p1->p2 是否与矩形相交。
        返回: (是否相交, 最小t参数, 碰撞边)"""
        d = p2 - p1
        t_min, t_max = 0.0, 1.0
        hit_edge = None

        if abs(d[0]) < 1e-10:
            if p1[0] < self.x_min or p1[0] > self.x_max:
                return False, 1.0, None
        else:
            t1 = (self.x_min - p1[0]) / d[0]
            t2 = (self.x_max - p1[0]) / d[0]
            if t1 > t2:
                t1, t2 = t2, t1
                edge_enter, edge_exit = 'right', 'left'
            else:
                edge_enter, edge_exit = 'left', 'right'
            if t1 > t_min:
                t_min = t1
                hit_edge = edge_enter
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False, 1.0, None

        if abs(d[1]) < 1e-10:
            if p1[1] < self.y_min or p1[1] > self.y_max:
                return False, 1.0, None
        else:
            t1 = (self.y_min - p1[1]) / d[1]
            t2 = (self.y_max - p1[1]) / d[1]
            if t1 > t2:
                t1, t2 = t2, t1
                edge_enter, edge_exit = 'top', 'bottom'
            else:
                edge_enter, edge_exit = 'bottom', 'top'
            if t1 > t_min:
                t_min = t1
                hit_edge = edge_enter
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False, 1.0, None

        if t_min > 0 and t_min <= 1.0:
            return True, t_min, hit_edge
        return False, 1.0, None

    def resolve(self, p_old: np.ndarray, p_new: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """碰撞解决：线段穿过障碍物则停在边缘。"""
        intersects, t_hit, hit_edge = self.line_intersects(p_old, p_new)
        margin = 0.05
        if intersects:
            p_fix = p_old + (t_hit - 0.01) * (p_new - p_old)
            v_fix = v.copy()
            if hit_edge == 'left':
                p_fix[0] = self.x_min - margin
                v_fix[0] = min(0, v_fix[0]) * -0.1
            elif hit_edge == 'right':
                p_fix[0] = self.x_max + margin
                v_fix[0] = max(0, v_fix[0]) * -0.1
            elif hit_edge == 'bottom':
                p_fix[1] = self.y_min - margin
                v_fix[1] = min(0, v_fix[1]) * -0.1
            elif hit_edge == 'top':
                p_fix[1] = self.y_max + margin
                v_fix[1] = max(0, v_fix[1]) * -0.1
            return p_fix, v_fix
        if self.contains(p_new):
            d_left = abs(p_new[0] - self.x_min)
            d_right = abs(self.x_max - p_new[0])
            d_bottom = abs(p_new[1] - self.y_min)
            d_top = abs(self.y_max - p_new[1])
            which = int(np.argmin([d_left, d_right, d_bottom, d_top]))
            p_fix = p_new.copy()
            v_fix = v.copy()
            if which == 0:
                p_fix[0] = self.x_min - margin
                v_fix[0] *= -0.1
            elif which == 1:
                p_fix[0] = self.x_max + margin
                v_fix[0] *= -0.1
            elif which == 2:
                p_fix[1] = self.y_min - margin
                v_fix[1] *= -0.1
            else:
                p_fix[1] = self.y_max + margin
                v_fix[1] *= -0.1
            return p_fix, v_fix
        return p_new, v


# ==================== 追逃环境（修正版） ====================
class PursuitEnvRealAUVFixed:
    """
    修正版环境 - 对齐2D_MDP_CN.pdf
    
    高层动作: 2D相对位移 (Δx, Δy) ∈ [-subgoal_range, subgoal_range]^2
    高层观测: 包含加速度信息
    """
    
    def __init__(self, nn_policy, device, 
                 world_size=30.0, catch_radius=1.0, max_steps=6000,
                 subgoal_range=0.5, low_steps=50, seed=42,
                 evader_mode="medium"):
        """
        Args:
            subgoal_range: 子目标最大位移范围 (m)
        """
        self.device = device
        self.nn_policy = nn_policy
        
        self.world_size = world_size
        self.half = world_size / 2
        self.catch_radius = catch_radius
        self.max_steps = max_steps
        self.subgoal_range = subgoal_range  # 子目标位移范围
        self.low_steps = low_steps
        self.evader_mode = evader_mode
        
        self.rng = np.random.default_rng(seed)

        # 逃避者策略类型: "potential" (势场法) 或 "sampling" (采样前瞻法) 或 "raycast" (射线投射间隙导航法)
        # evader_mode 同时控制难度和策略:
        #   simple/medium/hard → 势场法 (reactive)
        #   sampling_easy/sampling_medium/sampling_hard → 采样前瞻法 (predictive)
        #   raycast_easy/raycast_medium/raycast_hard → 射线间隙导航法 (geometric)
        #   unicycle_easy/unicycle_medium/unicycle_hard → unicycle原生采样法 (Plan C)
        if self.evader_mode.startswith("unicycle_"):
            self._evader_strategy = "unicycle_sampling"
            self._unicycle_sampling_difficulty = self.evader_mode.split("_", 1)[1]
        elif self.evader_mode.startswith("sampling_"):
            self._evader_strategy = "sampling"
            self._sampling_difficulty = self.evader_mode.split("_", 1)[1]  # easy/medium/hard
        elif self.evader_mode.startswith("raycast_"):
            self._evader_strategy = "raycast"
            self._raycast_difficulty = self.evader_mode.split("_", 1)[1]  # easy/medium/hard
        else:
            self._evader_strategy = "potential"
            if self.evader_mode not in ["simple", "medium", "hard"]:
                raise ValueError(f"Unsupported evader_mode={self.evader_mode}")
        
        # 逃跑者参数 - 满足不对称约束:
        # 实测 AUV 追击者有效速度: ~0.55-0.68 m/s (取决于 subgoal_range/low_steps)
        # 核心约束: v_max_evader > v_max_pursuer, a_max_evader < a_max_pursuer
        # 追击者实测加速度: ~1.73 m/s² (峰值), ~0.58 m/s² (均值)
        self.v_max_e = 0.72   # 逃跑者最大速度 (> 追击者有效速度 ~0.55, 比值 ≈1.31)
        self.a_max_e = 0.15   # 逃跑者最大纵向加速度
        self.omega_max_e = 1.2  # 逃跑者最大转向角速率 (rad/s)，对应最小转弯半径 0.6m@v=0.72
        self.dt_evader = 0.02 # 与AUV底层仿真步长一致
        
        # 逃脱者朝向角（unicycle model）
        self.psi_e = 0.0      # 逃跑者朝向角 (rad)
        self.speed_e = 0.0    # 逃跑者当前纵向速度
        
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
            RectObstacle(5.5, 5.0, 2.5, 2.5),
        ]
        
        # AUV环境
        self.auv_env = LowLevelAUV2DEnvV2Continuous(
            target_distance=subgoal_range,
            seed=seed
        )
        
        # 状态变量
        self.t = 0
        self.prev_positions = []
        self.e = None
        self.ve = None
        self.ae = None  # 逃跑者加速度
        self.psi_e = 0.0     # 逃跑者朝向
        self.speed_e = 0.0   # 逃跑者速度标量
        self.last_dist = None
        self.current_subgoal = None
        
        # Unicycle sampling决策缓存（每~0.25s重规划一次，不是每0.02s）
        self._cached_command = None
        self._command_age = 0
        self._command_replan_steps = int(0.25 / self.dt_evader)  # ~12步
        
        # 用于计算加速度的历史速度
        self.prev_vp = None
        self.prev_ve = None
        self.evader_collision_hits = 0

    def _evader_profile(self):
        """固定逃跑策略分级：simple/medium/hard。"""
        profiles = {
            "simple": {
                "run_w": 1.0,
                "obs_w": 0.45,
                "wall_w": 0.55,
                "center_w": 0.06,
                "lateral_w": 0.08,
                "obs_radius": 3.0,
                "obs_gain": 1.8,
                "wall_margin_ratio": 0.50,
                "wall_strength": 3.5,
                "risk_lookahead": 0.35,
                "risk_push": 0.6,
            },
            "medium": {
                # 基本对齐 tabular 默认
                "run_w": 1.0,
                "obs_w": 0.8,
                "wall_w": 1.0,
                "center_w": 0.08,
                "lateral_w": 0.25,
                "obs_radius": 4.0,
                "obs_gain": 2.5,
                "wall_margin_ratio": 0.40,
                "wall_strength": 5.0,
                "risk_lookahead": 0.45,
                "risk_push": 1.0,
            },
            "hard": {
                "run_w": 1.05,
                "obs_w": 1.2,
                "wall_w": 1.25,
                "center_w": 0.10,
                "lateral_w": 0.42,
                "obs_radius": 4.8,
                "obs_gain": 3.4,
                "wall_margin_ratio": 0.38,
                "wall_strength": 5.5,
                "risk_lookahead": 0.6,
                "risk_push": 1.4,
            },
        }
        return profiles[self.evader_mode]

    def _sampling_profile(self):
        """采样前瞻法逃避者参数配置。"""
        profiles = {
            "easy": {
                "n_samples": 12,
                "horizon_steps": 20,
                "sim_dt": 0.1,             # 粗模拟时间步 (实际dt=0.02, 这里用更大步长让前瞻更远)
                "w_dist_pursuer": 1.0,
                "pursuit_safe_dist": 6.0,
                "w_obs_clearance": 5.0,
                "w_wall_clearance": 5.0,
                "w_center_pref": 0.14,
                "w_smoothness": 0.3,
                "obs_danger_dist": 2.5,
                "wall_safe_margin": 9.0,
            },
            "medium": {
                "n_samples": 24,
                "horizon_steps": 30,       # 30 × 0.1s = 3s 前瞻
                "sim_dt": 0.1,
                "w_dist_pursuer": 1.0,
                "pursuit_safe_dist": 6.0,
                "w_obs_clearance": 6.0,
                "w_wall_clearance": 6.0,
                "w_center_pref": 0.18,
                "w_smoothness": 0.2,
                "obs_danger_dist": 3.0,
                "wall_safe_margin": 9.0,
            },
            "hard": {
                "n_samples": 36,
                "horizon_steps": 40,       # 40 × 0.1s = 4s 前瞻
                "sim_dt": 0.1,
                "w_dist_pursuer": 1.2,
                "pursuit_safe_dist": 7.0,
                "w_obs_clearance": 7.0,
                "w_wall_clearance": 7.0,
                "w_center_pref": 0.20,
                "w_smoothness": 0.15,
                "obs_danger_dist": 3.5,
                "wall_safe_margin": 9.0,
            },
        }
        return profiles[self._sampling_difficulty]

    def _evader_acceleration_sampling(self, p):
        """
        采样前瞻法 (Receding Horizon / Monte Carlo Sampling):
        1. 生成 N 个候选加速度方向（均匀 + 特殊方向）
        2. 对每个候选，向前模拟 H 步
        3. 评估每条轨迹的综合得分（远离追击者 + 远离障碍/边界 + 平滑性）
        4. 选得分最高的方向
        
        与势场法的本质区别：
        - 势场法: 基于当前状态的梯度反应
        - 采样法: 基于未来轨迹的全局评估
        """
        e = self.e
        ve = self.ve
        cfg = self._sampling_profile()
        
        n = cfg["n_samples"]
        H = cfg["horizon_steps"]
        sim_dt = cfg["sim_dt"]  # 粗模拟时间步，让前瞻距离更远
        
        # 1. 生成候选方向：均匀角度 + 几个特殊方向
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        candidates = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # 添加特殊方向：纯逃离、侧向逃离
        d = e - p
        dist_ep = np.linalg.norm(d)
        if dist_ep > 1e-6:
            flee_dir = d / dist_ep
            lateral_l = np.array([-flee_dir[1], flee_dir[0]])
            lateral_r = np.array([flee_dir[1], -flee_dir[0]])
            candidates = np.vstack([candidates, flee_dir, lateral_l, lateral_r])
        
        # 也添加"维持当前速度方向"
        ve_spd = np.linalg.norm(ve)
        if ve_spd > 0.05:
            candidates = np.vstack([candidates, ve / ve_spd])
        
        best_score = -np.inf
        best_dir = candidates[0]
        vp_now = self._get_pursuer_vel()
        
        # 2. 对每个候选方向模拟前瞻
        for acc_dir in candidates:
            sim_e = e.copy()
            sim_ve = ve.copy()
            score = 0.0
            invalid = False
            
            for step in range(H):
                # 施加加速度（用粗时间步）
                sim_ve = sim_ve + acc_dir * self.a_max_e * sim_dt
                spd = np.linalg.norm(sim_ve)
                if spd > self.v_max_e:
                    sim_ve = sim_ve * (self.v_max_e / spd)
                sim_e = sim_e + sim_ve * sim_dt
                
                # 时间衰减因子
                gamma = 0.95 ** step
                
                # (à) 远离追击者得分（带饱和：距离超过安全值后不再额外奖励）
                p_est = p + vp_now * (step + 1) * sim_dt
                dist_to_p = np.linalg.norm(sim_e - p_est)
                capped_dist = min(dist_to_p, cfg["pursuit_safe_dist"])
                score += cfg["w_dist_pursuer"] * capped_dist * gamma
                
                # (b) 障碍物安全距离
                for obs in self.obstacles:
                    obs_dist = obs.dist(sim_e)
                    if obs_dist < 0.1:
                        score -= 500.0
                        invalid = True
                        break
                    elif obs_dist < cfg["obs_danger_dist"]:
                        score -= cfg["w_obs_clearance"] / (obs_dist + 0.2) * gamma
                
                if invalid:
                    break
                
                # (c) 边界安全距离
                wall_clearance = min(
                    self.half - abs(sim_e[0]),
                    self.half - abs(sim_e[1])
                )
                if wall_clearance < 0.3:
                    score -= 500.0
                    invalid = True
                    break
                elif wall_clearance < cfg["wall_safe_margin"]:
                    score -= cfg["w_wall_clearance"] / (wall_clearance + 0.3) * gamma
                
                # (d) 中心偏好：二次方惩罚，离中心越远惩罚增长越快
                center_dist_sq = sim_e[0]**2 + sim_e[1]**2
                score -= cfg["w_center_pref"] * center_dist_sq * gamma
            
            if invalid:
                score -= 1000.0
            
            # (d) 速度平滑性：惩罚与当前速度方向差异大的候选
            if ve_spd > 0.05:
                cos_sim = np.dot(acc_dir, ve / ve_spd)
                score += cfg["w_smoothness"] * cos_sim * H
            
            if score > best_score:
                best_score = score
                best_dir = acc_dir
        
        return best_dir * self.a_max_e

    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        self.prev_positions = []
        self.prev_vp = None
        self.prev_ve = None
        self.evader_collision_hits = 0
        
        # 随机初始位置
        for _ in range(500):
            p = self.rng.uniform(-self.half + 2, self.half - 2, 2)
            e = self.rng.uniform(-self.half + 2, self.half - 2, 2)
            d = np.linalg.norm(p - e)
            if d >= 5.0 and d <= 10.0:
                if not self._in_obstacle(p) and not self._in_obstacle(e):
                    break
        
        # 追踪者朝向逃跑者
        direction = e - p
        psi = math.atan2(direction[1], direction[0])
        
        # 计算初始子目标
        self.current_subgoal = np.array([p[0], p[1]])
        
        # 重置并初始化AUV
        self.auv_env.reset()
        # 直接操作内部dynamics设置状态
        self.auv_env.inner.dynamics.eta[0] = p[0]
        self.auv_env.inner.dynamics.eta[1] = p[1]
        self.auv_env.inner.dynamics.eta[2] = self.auv_env.fixed_z
        self.auv_env.inner.dynamics.eta[5] = psi
        self.auv_env.inner.dynamics.nu[:] = 0.0
        
        # 设置AUV目标
        self.auv_env.inner.target_state = np.array([
            self.current_subgoal[0], self.current_subgoal[1], self.auv_env.fixed_z,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        
        # 逃跑者状态
        self.e = e.copy()
        self.ve = np.zeros(2)
        self.ae = np.zeros(2)
        # 逃脱者初始朝向：背对追击者
        flee_dir = e - p
        self.psi_e = math.atan2(flee_dir[1], flee_dir[0])
        self.speed_e = 0.0
        
        # 重置决策缓存
        self._cached_command = None
        self._command_age = 0
        
        d = np.linalg.norm(p - e)
        self.last_dist = d
        
        return self._high_obs()
    
    def _in_obstacle(self, p):
        for obs in self.obstacles:
            if obs.contains(p):
                return True
        return False
    
    def _subgoal_blocked(self, pursuer_pos, subgoal):
        """检查 subgoal 是否在障碍物内部，或路径是否穿过障碍物。"""
        for obs in self.obstacles:
            if obs.contains(subgoal):
                return True
            hit, t, _ = obs.line_intersects(pursuer_pos, subgoal)
            if hit and 0.01 < t < 0.99:
                return True
        return False

    def _push_subgoal_out_of_obstacles(self, pursuer_pos, subgoal, margin=0.3):
        """确保 subgoal 不在障碍物内部，且路径不穿过障碍物。
        
        核心策略：方向扫描绕行法
        当直接路径被障碍物阻挡时，不是简单缩回 subgoal，而是以最小偏转角
        找到一个能绕过障碍物的替代方向，让追击者实际绕过障碍物去追击。
        
        算法：
        1. 若当前 subgoal 路径畅通 → 直接返回
        2. 否则，从原始方向开始，按 5° 步长左右交替扫描
        3. 对每个候选方向，检查该方向上 subgoal_range 距离处是否安全
        4. 返回第一个安全的方向（最小偏转角）
        """
        original_dir = subgoal - pursuer_pos
        sg_dist = np.linalg.norm(original_dir)
        if sg_dist < 1e-6:
            return subgoal
        
        # 快速检查：当前 subgoal 已经安全则直接返回
        if not self._subgoal_blocked(pursuer_pos, subgoal):
            return subgoal
        
        # 方向扫描：从原始方向开始，逐步增大偏转角
        base_angle = math.atan2(original_dir[1], original_dir[0])
        
        for deg in range(5, 180, 5):
            rad = math.radians(deg)
            for sign in (1, -1):
                angle = base_angle + sign * rad
                direction = np.array([math.cos(angle), math.sin(angle)])
                candidate = pursuer_pos + direction * sg_dist
                
                # 边界裁剪
                candidate = np.clip(candidate, -self.half + 0.5, self.half - 0.5)
                
                if not self._subgoal_blocked(pursuer_pos, candidate):
                    return candidate
        
        # 所有方向都被阻挡（极端情况）— 尝试更短距离
        for shorter_ratio in (0.5, 0.3):
            short_dist = sg_dist * shorter_ratio
            for deg in range(0, 180, 10):
                rad = math.radians(deg)
                for sign in (1, -1) if deg > 0 else (1,):
                    angle = base_angle + sign * rad
                    direction = np.array([math.cos(angle), math.sin(angle)])
                    candidate = pursuer_pos + direction * short_dist
                    candidate = np.clip(candidate, -self.half + 0.5, self.half - 0.5)
                    if not self._subgoal_blocked(pursuer_pos, candidate):
                        return candidate
        
        return pursuer_pos.copy()
    
    def _get_pursuer_state(self):
        """获取追踪者完整状态"""
        eta = self.auv_env.inner.dynamics.eta
        nu = self.auv_env.inner.dynamics.nu
        return np.array([
            eta[0],  # x
            eta[1],  # y
            eta[5],  # psi (yaw)
            nu[0],   # u (surge)
            nu[1],   # v (sway)
            nu[5]    # r (yaw rate)
        ])
    
    def _get_pursuer_vel(self):
        """追踪者速度 (世界坐标下)"""
        nu = self.auv_env.inner.dynamics.nu
        eta = self.auv_env.inner.dynamics.eta
        u = nu[0]  # surge
        v = nu[1]  # sway
        psi = eta[5]  # yaw
        # 从体坐标转世界坐标
        vx = u * math.cos(psi) - v * math.sin(psi)
        vy = u * math.sin(psi) + v * math.cos(psi)
        return np.array([vx, vy])
    
    def _get_pursuer_acc(self):
        """追踪者加速度（世界坐标下，近似计算）"""
        if self.prev_vp is None:
            return np.zeros(2)
        else:
            vp = self._get_pursuer_vel()
            acc = (vp - self.prev_vp) / self.dt_evader
            return acc
    
    def _evader_acceleration(self, p):
        """逃跑者加速度计算：根据策略类型分派。"""
        if self._evader_strategy == "sampling":
            return self._evader_acceleration_sampling(p)
        elif self._evader_strategy == "raycast":
            return self._evader_acceleration_raycast(p)
        return self._evader_acceleration_potential(p)

    def _evader_command(self, p):
        """Unicycle接口：将策略层的加速度向量转换为 (target_heading, target_speed)。
        
        ==== Plan B: 可达性感知转换 ====
        核心改进：策略层假设点质量动力学（加速度瞬间改变方向），
        但执行层是unicycle（转弯受omega_max限制）。
        本方法将策略期望映射到unicycle实际可达的动作空间。
        
        转换逻辑：
        1. 用加速度向量推演"期望速度方向"
        2. 计算期望朝向与当前朝向的偏差
        3. 若偏差超出短期可达范围，将目标朝向钳位到可达边界
        4. 大角度转弯时主动降速（防止转弯时侧漂撞墙）
        """
        # 对unicycle_sampling策略，使用缓存机制（每~0.25s重规划一次）
        if self._evader_strategy == "unicycle_sampling":
            self._command_age += 1
            if self._cached_command is None or self._command_age >= self._command_replan_steps:
                self._cached_command = self._evader_command_unicycle_sampling(p)
                self._command_age = 0
            return self._cached_command
        
        acc = self._evader_acceleration(p)
        
        # 用前瞻推演期望速度方向
        lookahead_dt = 0.3
        desired_vel = self.ve + acc * lookahead_dt
        
        desired_speed = np.linalg.norm(desired_vel)
        if desired_speed < 1e-4:
            return self.psi_e, 0.0
        
        raw_target = math.atan2(desired_vel[1], desired_vel[0])
        
        # ==== 可达性钳位 ====
        heading_err = math.atan2(
            math.sin(raw_target - self.psi_e),
            math.cos(raw_target - self.psi_e)
        )
        # 在一个planning horizon内（0.5s）能转多少度
        reachable_range = self.omega_max_e * 0.5
        
        if abs(heading_err) > reachable_range:
            # 钳位到可达边界，但稍微超出一点（让控制器持续转向）
            clamped_err = math.copysign(reachable_range * 1.2, heading_err)
            target_heading = self.psi_e + clamped_err
        else:
            target_heading = raw_target
        
        target_heading = math.atan2(math.sin(target_heading), math.cos(target_heading))
        
        # ==== 大角度转弯时主动降速 ====
        abs_err = abs(heading_err)
        if abs_err > math.radians(90):
            # >90°偏差：大幅降速，让转弯尽快完成
            speed_scale = 0.2
        elif abs_err > math.radians(45):
            # 45°-90°：线性降速
            blend = (abs_err - math.radians(45)) / math.radians(45)
            speed_scale = 1.0 - 0.8 * blend  # 1.0 → 0.2
        else:
            speed_scale = 1.0
        
        target_speed = min(desired_speed, self.v_max_e) * speed_scale
        
        return target_heading, target_speed

    # ==================== Plan C: Unicycle-Native 采样策略 ====================
    def _unicycle_sampling_profile(self):
        """Unicycle原生采样策略参数配置。"""
        profiles = {
            "easy": {
                "n_omega": 7, "n_speed": 3, "horizon": 2.0, "sim_dt": 0.05,
                "w_flee": 1.0, "w_obs": 6.0, "w_wall": 5.0, "w_center": 0.15,
                "w_smooth": 0.4, "obs_danger": 2.0, "wall_margin": 3.0,
            },
            "medium": {
                "n_omega": 9, "n_speed": 3, "horizon": 2.0, "sim_dt": 0.08,
                "w_flee": 1.0, "w_obs": 8.0, "w_wall": 6.0, "w_center": 0.18,
                "w_smooth": 0.3, "obs_danger": 2.5, "wall_margin": 3.5,
            },
            "hard": {
                "n_omega": 15, "n_speed": 5, "horizon": 3.0, "sim_dt": 0.05,
                "w_flee": 1.2, "w_obs": 10.0, "w_wall": 7.0, "w_center": 0.20,
                "w_smooth": 0.2, "obs_danger": 3.0, "wall_margin": 4.0,
            },
        }
        return profiles[self._unicycle_sampling_difficulty]

    def _evader_command_unicycle_sampling(self, p):
        """
        Plan C: Unicycle原生采样策略 — 直接在 (omega, speed) 空间采样+前瞻。
        
        与旧采样法的本质区别：
        - 旧采样法: 采样加速度方向 → 点质量前瞻 → 再转换为unicycle指令
          → 前瞻轨迹和实际执行不一致
        - 本方法: 直接采样unicycle可执行的 (omega, speed) 组合 → unicycle前瞻
          → 前瞻轨迹 = 实际将执行的轨迹（真实可达）
        
        算法步骤：
        1. 在 omega ∈ [-omega_max, +omega_max] 和 speed ∈ [0, v_max] 上网格采样
        2. 对每个 (omega, speed) 对，用unicycle动力学前瞻模拟 horizon 秒
        3. 综合评分：远离追击者 + 避障 + 避墙 + 中心偏好 + 平滑性
        4. 选最高分的 (omega, speed)，返回对应的 (target_heading, target_speed)
        """
        cfg = self._unicycle_sampling_profile()
        e = self.e.copy()
        psi = self.psi_e
        spd = self.speed_e
        ve = self.ve.copy()
        
        n_omega = cfg["n_omega"]
        n_speed = cfg["n_speed"]
        horizon = cfg["horizon"]
        sim_dt = cfg["sim_dt"]
        H = int(horizon / sim_dt)
        
        # 采样网格
        omegas = np.linspace(-self.omega_max_e, self.omega_max_e, n_omega)
        speeds = np.linspace(0.1 * self.v_max_e, self.v_max_e, n_speed)
        
        # 追击者速度，用于估算追击者未来位置
        vp_now = self._get_pursuer_vel()
        
        best_score = -np.inf
        best_omega = 0.0
        best_speed = self.v_max_e * 0.5
        
        for omega_cmd in omegas:
            for speed_cmd in speeds:
                sim_psi = psi
                sim_spd = spd
                sim_x, sim_y = e[0], e[1]
                score = 0.0
                invalid = False
                
                for step in range(H):
                    # Unicycle积分（与_step_evader完全一致）
                    # 转向：P控制 omega_cmd 只是目标omega，这里简化为直接用
                    sim_psi += omega_cmd * sim_dt
                    sim_psi = math.atan2(math.sin(sim_psi), math.cos(sim_psi))
                    
                    # 速度逼近
                    speed_err = speed_cmd - sim_spd
                    a_lon = max(min(speed_err * 2.0, self.a_max_e), -self.a_max_e)
                    sim_spd += a_lon * sim_dt
                    sim_spd = max(min(sim_spd, self.v_max_e), 0.0)
                    
                    # 位置更新
                    sim_x += sim_spd * math.cos(sim_psi) * sim_dt
                    sim_y += sim_spd * math.sin(sim_psi) * sim_dt
                    
                    gamma = 0.95 ** step
                    sim_e = np.array([sim_x, sim_y])
                    
                    # (a) 远离追击者（带位置预测）
                    p_est = p + vp_now * (step + 1) * sim_dt
                    dist_to_p = np.linalg.norm(sim_e - p_est)
                    score += cfg["w_flee"] * min(dist_to_p, 8.0) * gamma
                    
                    # (b) 障碍物安全
                    for obs in self.obstacles:
                        od = obs.dist(sim_e)
                        if od < 0.15:
                            score -= 500.0
                            invalid = True
                            break
                        elif od < cfg["obs_danger"]:
                            score -= cfg["w_obs"] / (od + 0.2) * gamma
                    if invalid:
                        break
                    
                    # (c) 边界安全
                    wall_clear = min(
                        self.half - abs(sim_x),
                        self.half - abs(sim_y)
                    )
                    if wall_clear < 0.3:
                        score -= 500.0
                        invalid = True
                        break
                    elif wall_clear < cfg["wall_margin"]:
                        score -= cfg["w_wall"] / (wall_clear + 0.3) * gamma
                    
                    # (d) 中心偏好
                    center_dist_sq = sim_x**2 + sim_y**2
                    score -= cfg["w_center"] * center_dist_sq * gamma
                
                if invalid:
                    score -= 1000.0
                
                # (e) 平滑性：偏好与当前运动方向一致的omega
                # omega_cmd 接近0 = 保持直行 = 最平滑
                score -= cfg["w_smooth"] * abs(omega_cmd) * H
                
                if score > best_score:
                    best_score = score
                    best_omega = omega_cmd
                    best_speed = speed_cmd
        
        # 将最优omega转换为target_heading（前瞻一小段时间）
        target_heading = psi + best_omega * 0.3  # 用0.3s前瞻确定目标朝向
        target_heading = math.atan2(math.sin(target_heading), math.cos(target_heading))
        
        return target_heading, best_speed

    # ==================== 射线投射间隙导航策略 ====================
    def _evader_acceleration_raycast(self, p):
        """
        射线投射间隙导航法 (Ray-Cast Gap Navigation):
        
        与势场法的本质区别：
        - 势场法: 所有力叠加，容易产生局部极小值和力的冲突
        - 射线投射法: 先构建环境的自由空间感知，再从中选择最优逃跑方向
        
        算法步骤：
        1. 向 N 个方向发射射线，计算每个方向的自由距离
        2. 在自由距离图上寻找"间隙"(连续的开阔区域)
        3. 用逃离评分（远离追击者 + 自由距离 + 速度平滑）选择最佳间隙
        4. 计算目标速度并输出加速度
        
        核心优势：自然避免撞墙/障碍物，因为方向选择就排除了被阻挡的路径。
        """
        e = self.e
        ve = self.ve
        cfg = self._raycast_profile()
        
        n_rays = cfg["n_rays"]
        max_range = cfg["max_range"]
        
        # ===== 1. 射线投射：计算每个方向的自由距离 =====
        ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        free_dists = np.full(n_rays, max_range)
        
        for i, angle in enumerate(ray_angles):
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            ray_end = e + ray_dir * max_range
            
            # 检测射线与障碍物的交点
            for obs in self.obstacles:
                intersects, t_hit, _ = obs.line_intersects(e, ray_end)
                if intersects:
                    hit_dist = t_hit * max_range
                    free_dists[i] = min(free_dists[i], hit_dist)
            
            # 检测射线与边界的交点
            for axis_val, boundary in [(0, self.half), (0, -self.half),
                                        (1, self.half), (1, -self.half)]:
                d_comp = ray_dir[axis_val]
                if abs(d_comp) < 1e-10:
                    continue
                t_wall = (boundary - e[axis_val]) / d_comp
                if 0 < t_wall < max_range:
                    # 验证交点在地图范围内
                    other_axis = 1 - axis_val
                    cross_pos = e[other_axis] + ray_dir[other_axis] * t_wall
                    if abs(cross_pos) <= self.half:
                        free_dists[i] = min(free_dists[i], t_wall)
        
        # 减去安全余量
        safe_margin = cfg["safe_margin"]
        effective_dists = np.maximum(free_dists - safe_margin, 0.0)
        
        # ===== 2. 寻找间隙区域 =====
        # 间隙 = 连续的自由距离 > 阈值的角度区域
        min_gap_dist = cfg["min_gap_dist"]
        open_mask = effective_dists > min_gap_dist
        
        # 找到所有连续开放区间 (处理环形wrap)
        gaps = []
        if np.any(open_mask):
            # 使用差分找区间边界
            extended = np.concatenate([open_mask, open_mask[:1]])  # 处理wrap
            changes = np.diff(extended.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1
            
            if open_mask[0]:
                starts = np.concatenate([[0], starts])
            if open_mask[-1] and not open_mask[0]:
                ends = np.append(ends, n_rays)
            
            # 匹配starts和ends
            if len(starts) > 0 and len(ends) > 0:
                # 处理wrap-around情况
                if open_mask[0] and open_mask[-1] and len(starts) > 1:
                    # 首尾连通，合并
                    last_start = starts[-1]
                    first_end = ends[0]
                    starts = starts[:-1]
                    ends = ends[1:] if len(ends) > 1 else np.array([first_end + n_rays - last_start])
                    # 合并的间隙
                    merged_indices = list(range(last_start, n_rays)) + list(range(0, first_end))
                    if len(starts) == len(ends):
                        gaps.append(merged_indices)
                
                for s, nd in zip(starts[:min(len(starts), len(ends))],
                                  ends[:min(len(starts), len(ends))]):
                    gap_indices = list(range(s, min(nd, n_rays)))
                    if gap_indices:
                        gaps.append(gap_indices)
        
        # 如果没有找到间隙，退化为选择最大自由距离方向
        if not gaps:
            gaps = [[np.argmax(effective_dists)]]
        
        # ===== 3. 对每个间隙评分 =====
        flee_dir = e - p
        flee_dist = np.linalg.norm(flee_dir)
        if flee_dist > 1e-6:
            flee_dir = flee_dir / flee_dist
        else:
            flee_dir = np.array([1.0, 0.0])
        
        ve_speed = np.linalg.norm(ve)
        ve_dir = ve / ve_speed if ve_speed > 0.05 else flee_dir
        
        best_score = -np.inf
        best_angle = 0.0
        
        for gap in gaps:
            # 间隙中心角和平均自由距离
            gap_angles = ray_angles[gap]
            gap_dists = effective_dists[gap]
            
            # 自由距离加权的中心角
            weights = gap_dists / (np.sum(gap_dists) + 1e-6)
            
            # 用加权平均计算间隙中心方向（在单位圆上）
            cx = np.sum(weights * np.cos(gap_angles))
            cy = np.sum(weights * np.sin(gap_angles))
            center_angle = math.atan2(cy, cx)
            center_dir = np.array([math.cos(center_angle), math.sin(center_angle)])
            
            # 评分项
            # (a) 逃离追击者得分
            flee_score = np.dot(center_dir, flee_dir) * cfg["w_flee"]
            
            # (b) 自由空间得分
            avg_free = np.mean(gap_dists)
            space_score = min(avg_free / max_range, 1.0) * cfg["w_space"]
            
            # (c) 间隙宽度得分（更宽的间隙=更多机动空间）
            gap_width = len(gap) / n_rays  # 归一化到 [0,1]
            width_score = gap_width * cfg["w_width"]
            
            # (d) 速度平滑得分
            smooth_score = np.dot(center_dir, ve_dir) * cfg["w_smooth"]
            
            # (e) 中心偏好（靠近中心的方向加分）
            test_pos = e + center_dir * 2.0
            center_dist = np.linalg.norm(test_pos) / self.half
            center_score = (1.0 - center_dist) * cfg["w_center"]
            
            total_score = flee_score + space_score + width_score + smooth_score + center_score
            
            if total_score > best_score:
                best_score = total_score
                best_angle = center_angle
        
        # ===== 4. 计算目标速度和加速度 =====
        target_dir = np.array([math.cos(best_angle), math.sin(best_angle)])
        
        # 目标速度取决于前方自由距离
        # 查找目标方向的自由距离
        angle_idx = int(round(best_angle / (2 * math.pi) * n_rays)) % n_rays
        forward_free = effective_dists[angle_idx]
        
        # 前方空间不足时减速
        if forward_free < cfg["slow_dist"]:
            speed_ratio = max(forward_free / cfg["slow_dist"], 0.2)
        else:
            speed_ratio = 1.0
        
        target_vel = target_dir * self.v_max_e * speed_ratio
        
        # P控制器计算加速度
        acc = (target_vel - ve) * cfg["vel_kp"]
        acc_mag = np.linalg.norm(acc)
        if acc_mag > self.a_max_e:
            acc = acc / acc_mag * self.a_max_e
        
        return acc

    def _raycast_profile(self):
        """射线投射间隙导航法参数配置。"""
        profiles = {
            "easy": {
                "n_rays": 36,
                "max_range": 8.0,
                "safe_margin": 0.5,
                "min_gap_dist": 1.5,
                "slow_dist": 3.0,
                "vel_kp": 0.8,
                "w_flee": 1.0,
                "w_space": 0.5,
                "w_width": 0.3,
                "w_smooth": 0.2,
                "w_center": 0.15,
            },
            "medium": {
                "n_rays": 48,
                "max_range": 10.0,
                "safe_margin": 0.6,
                "min_gap_dist": 2.0,
                "slow_dist": 3.5,
                "vel_kp": 1.0,
                "w_flee": 1.0,
                "w_space": 0.7,
                "w_width": 0.4,
                "w_smooth": 0.15,
                "w_center": 0.20,
            },
            "hard": {
                "n_rays": 72,
                "max_range": 12.0,
                "safe_margin": 0.7,
                "min_gap_dist": 2.5,
                "slow_dist": 4.0,
                "vel_kp": 1.2,
                "w_flee": 1.0,
                "w_space": 0.8,
                "w_width": 0.5,
                "w_smooth": 0.1,
                "w_center": 0.25,
            },
        }
        return profiles[self._raycast_difficulty]

    def _ray_free_dist(self, origin, direction, max_range):
        """从 origin 沿 direction 射线，返回最近碰撞距离（障碍物+边界）。"""
        ray_end = origin + direction * max_range
        best = max_range
        for obs in self.obstacles:
            intersects, t_hit, _ = obs.line_intersects(origin, ray_end)
            if intersects:
                best = min(best, t_hit * max_range)
        # 边界
        for axis in range(2):
            d_comp = direction[axis]
            if abs(d_comp) < 1e-10:
                continue
            for boundary in (self.half, -self.half):
                t_wall = (boundary - origin[axis]) / d_comp
                if 0 < t_wall < best:
                    other = 1 - axis
                    cross = origin[other] + direction[other] * t_wall
                    if abs(cross) <= self.half:
                        best = t_wall
        return best

    def _nearest_obs_dist(self, e):
        """返回离 e 最近障碍物的距离"""
        return min((obs.dist(e) for obs in self.obstacles), default=999.0)

    def _find_best_escape_direction(self, e, run_dir):
        """
        最小偏转角法 (Minimum Deflection) + 锥形安全检查（速度感知版）：
        
        从 run_dir 开始，同时向左/右扫描，找到第一个"锥形安全"的方向。
        锥形安全 = 中心射线和两侧 ±15° 射线都有足够间距。
        
        速度感知：min_free 根据当前速度动态调整，高速时需要更多前方空间。
        扫描偏好：优先选择与当前速度方向偏差小的一侧，减少急转弯。
        """
        ray_range = 10.0
        ve_spd = np.linalg.norm(self.ve)
        # 高速时需更多前方空间：停车距 v²/(2a) + 安全余量
        stopping_dist = ve_spd ** 2 / (2.0 * self.a_max_e + 1e-6)
        min_free = max(3.5, stopping_dist + 1.5)
        min_free = min(min_free, ray_range - 1.0)  # 不超过射线范围
        
        cone_half = math.radians(15)  # 锥半角 15°（比原来12°更宽，更安全）
        step_deg = 8     # 每步扫描 8°
        max_deflect = 150  # 最大偏转 150°

        def cone_safe(direction):
            """检查 direction ± cone_half 三根射线是否都 >= min_free"""
            a0 = math.atan2(direction[1], direction[0])
            for da in (0, cone_half, -cone_half):
                a = a0 + da
                d = np.array([math.cos(a), math.sin(a)])
                if self._ray_free_dist(e, d, ray_range) < min_free:
                    return False
            return True

        base_angle = math.atan2(run_dir[1], run_dir[0])
        
        # 先检查原始方向
        if cone_safe(run_dir):
            return run_dir
        
        # 确定扫描优先侧：优先向当前速度方向偏转（减少急转弯）
        if ve_spd > 0.1:
            ve_angle = math.atan2(self.ve[1], self.ve[0])
            angle_diff = math.atan2(math.sin(ve_angle - base_angle),
                                    math.cos(ve_angle - base_angle))
            preferred_sign = 1 if angle_diff >= 0 else -1
        else:
            preferred_sign = 1
        other_sign = -preferred_sign
        
        # 左右同时扫描，优先速度方向侧
        for deg in range(step_deg, max_deflect + 1, step_deg):
            rad = math.radians(deg)
            for sign in (preferred_sign, other_sign):
                angle = base_angle + sign * rad
                cand = np.array([math.cos(angle), math.sin(angle)])
                if cone_safe(cand):
                    return cand
        
        # 极端情况：选自由距离最长的
        best_d = 0
        best_dir = run_dir
        for deg in range(0, 360, 15):
            angle = math.radians(deg)
            cand = np.array([math.cos(angle), math.sin(angle)])
            free_d = self._ray_free_dist(e, cand, ray_range)
            if free_d > best_d:
                best_d = free_d
                best_dir = cand
        return best_dir

    def _evader_acceleration_potential(self, p):
        """
        势场法逃避者加速度（V3c）：最小偏转 + 距离自适应平滑 + 短距安全推力 + 速度预测制动。
        """
        e = self.e
        ve = self.ve
        cfg = self._evader_profile()
        d = e - p
        dist = np.linalg.norm(d)
        
        if dist < 1e-6:
            return np.zeros(2)
        
        # 原始逃离方向
        raw_run = d / dist
        
        # ===== 核心：最小偏转方向选择 =====
        run = self._find_best_escape_direction(e, raw_run)
        
        # ===== 速度方向平滑：根据最近障碍物距离自适应权重 =====
        ve_spd = np.linalg.norm(ve)
        ve_dir = ve / ve_spd if ve_spd > 0.05 else run  # 预先定义，避免后续引用错误
        if ve_spd > 0.15:
            ve_dir = ve / ve_spd
            obs_d = self._nearest_obs_dist(e)
            # 距离远 → 多保持惯性；距离近 → 跟随逃跑方向（但不过于激进）
            if obs_d > 4.0:
                vel_w, esc_w = 0.6, 0.4
            elif obs_d > 2.0:
                t = (obs_d - 2.0) / 2.0  # [0, 1]
                vel_w = 0.25 + 0.35 * t   # 0.25 ~ 0.6
                esc_w = 1.0 - vel_w
            else:
                vel_w, esc_w = 0.2, 0.8   # 近距也保留更多惯性，避免急转
            blended = esc_w * run + vel_w * ve_dir
            bn = np.linalg.norm(blended)
            if bn > 1e-6:
                run = blended / bn
        
        # ===== 短距安全推力：距障碍物 < 1.2m 时推离 =====
        safety_push = np.zeros(2)
        safety_range = 1.2
        for obs in self.obstacles:
            od = obs.dist(e)
            if od < safety_range and od > 1e-6:
                # 最近点 = 坐标 clamped 到矩形内
                nx = np.clip(e[0], obs.x_min, obs.x_max)
                ny = np.clip(e[1], obs.y_min, obs.y_max)
                push = e - np.array([nx, ny])
                pn = np.linalg.norm(push)
                if pn > 1e-6:
                    push = push / pn
                    strength = (safety_range - od) / safety_range  # 0→1
                    # 近距(< 0.3m)急推，远距温和
                    gain = 3.0 if od < 0.3 else 2.0
                    safety_push += push * strength * gain

        # (速度预测制动已移至执行层 _step_evader，避免双重制动导致过度减速)

        # ===== 边界回避 — smoothstep（更早起效）=====
        avoid_wall = np.zeros(2)
        margin = cfg["wall_margin_ratio"] * self.half
        wall_strength = cfg["wall_strength"]
        soft_zone = 3.0  # 扩大软区域（原2.0）
        for axis in range(2):
            pos = e[axis]
            onset = margin - soft_zone
            if pos > onset:
                dist_to_wall = self.half - pos
                t = np.clip((pos - onset) / soft_zone, 0.0, 1.0)
                ramp = t * t * (3.0 - 2.0 * t)
                avoid_wall[axis] -= wall_strength * ramp / (dist_to_wall + 0.5)
            if pos < -onset:
                dist_to_wall = self.half + pos
                t = np.clip((-pos - onset) / soft_zone, 0.0, 1.0)
                ramp = t * t * (3.0 - 2.0 * t)
                avoid_wall[axis] += wall_strength * ramp / (dist_to_wall + 0.5)
        
        # 向中心的微弱吸引力
        center_pull = -e * cfg["center_w"]
        
        # 侧向机动
        lateral = np.array([-run[1], run[0]])
        if self.evader_mode == "hard":
            side = 1.0 if ((self.t // 8) % 2 == 0) else -1.0
            lateral = lateral * side
        
        # ===== 合并 =====
        cmd = (
            cfg["run_w"] * run
            + cfg["wall_w"] * avoid_wall
            + center_pull
            + cfg["lateral_w"] * lateral
            + safety_push
        )
        
        # 限制加速度
        cmd_norm = np.linalg.norm(cmd)
        if cmd_norm > 1.0:
            cmd = cmd / cmd_norm
        
        return cmd * self.a_max_e


    def _resolve_collisions(self, p_old, p_new, v):
        """对所有障碍物做线段碰撞检测与解决"""
        for _ in range(3):  # 最多3次迭代处理连续碰撞
            collision_found = False
            for obs in self.obstacles:
                p_prev = p_new.copy()
                p_new, v = obs.resolve(p_old, p_new, v)
                if not np.allclose(p_prev, p_new):
                    collision_found = True
            if not collision_found:
                break
        return p_new, v

    def _step_evader(self, num_steps=1):
        """更新逃跑者状态 — Unicycle动力学 + 碰撞处理。
        
        动力学模型：
          psi_e' = omega           (转向角速率，受 omega_max_e 限制)
          speed_e' = a_lon         (纵向加速度，受 a_max_e 限制)
          x' = speed_e * cos(psi_e)
          y' = speed_e * sin(psi_e)
        
        策略层输出 (target_heading, target_speed)，由P控制器计算 omega 和 a_lon。
        """
        eta = self.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        
        for _ in range(num_steps):
            self.prev_ve = self.ve.copy()
            
            # ==== 策略层：计算目标朝向和目标速度 ====
            target_heading, target_speed = self._evader_command(p)
            
            # ==== 转向控制器 ====
            heading_err = math.atan2(
                math.sin(target_heading - self.psi_e),
                math.cos(target_heading - self.psi_e)
            )
            # P控制 + 速度依赖增益：低速时转弯更灵活
            speed_ratio = self.speed_e / (self.v_max_e + 1e-6)
            turn_gain = 3.0 - 1.5 * speed_ratio  # 低速3.0, 高速1.5
            omega = turn_gain * heading_err
            omega = np.clip(omega, -self.omega_max_e, self.omega_max_e)
            
            # ==== 纵向速度控制器 ====
            # 前方有障碍物/墙壁时自动减速
            fwd_dir = np.array([math.cos(self.psi_e), math.sin(self.psi_e)])
            fwd_free = self._ray_free_dist(self.e, fwd_dir, 8.0)
            brake_dist = self.speed_e ** 2 / (2.0 * self.a_max_e + 1e-6) + 0.8
            
            if fwd_free < brake_dist:
                # 紧急减速：前方空间不足
                safe_speed = math.sqrt(max(2.0 * self.a_max_e * (fwd_free - 0.3), 0.0))
                target_speed = min(target_speed, safe_speed)
            
            # （大角度转弯减速已在 _evader_command 中处理，此处仅作安全兜底）
            if abs(heading_err) > math.radians(90):
                target_speed = min(target_speed, self.v_max_e * 0.15)
            
            speed_err = target_speed - self.speed_e
            a_lon = np.clip(speed_err * 2.0, -self.a_max_e, self.a_max_e)
            
            # ==== Unicycle 积分 ====
            self.psi_e += omega * self.dt_evader
            self.psi_e = math.atan2(math.sin(self.psi_e), math.cos(self.psi_e))  # wrap
            
            self.speed_e += a_lon * self.dt_evader
            self.speed_e = np.clip(self.speed_e, 0.0, self.v_max_e)  # 不允许倒退
            
            # 更新速度向量（供外部使用和观测）
            self.ve = np.array([
                self.speed_e * math.cos(self.psi_e),
                self.speed_e * math.sin(self.psi_e)
            ])
            self.ae = np.array([a_lon * math.cos(self.psi_e), a_lon * math.sin(self.psi_e)])
            
            # ==== 位置更新 + 碰撞处理 ====
            e_old = self.e.copy()
            e_new = self.e + self.ve * self.dt_evader
            
            # 障碍物碰撞检测
            e_resolved, v_resolved = self._resolve_collisions(e_old, e_new, self.ve)
            if not np.allclose(e_new, e_resolved):
                self.evader_collision_hits += 1
                self.speed_e *= 0.5  # 碰撞后减速（从0.3改为0.5，恢复更快）
                # 根据碰撞修正速度方向更新朝向
                v_spd = np.linalg.norm(v_resolved)
                if v_spd > 0.01:
                    self.psi_e = math.atan2(v_resolved[1], v_resolved[0])
                self.ve = v_resolved
            e_new = e_resolved
            self.e = e_new
            
            # 边界处理（平滑减速 + 朝向偏转）
            for axis in range(2):
                soft_start = self.half - 3.0
                hard_limit = self.half - 0.2
                pos = self.e[axis]
                
                # 正方向边界
                if pos > soft_start:
                    ratio = max((hard_limit - pos) / (hard_limit - soft_start), 0.0)
                    # 减速
                    self.speed_e *= (0.3 + 0.7 * ratio)
                    if pos > hard_limit:
                        self.e[axis] = hard_limit
                        self.speed_e *= 0.2
                        # 偏转朝向：远离墙壁
                        if axis == 0 and math.cos(self.psi_e) > 0:
                            self.psi_e = math.pi - self.psi_e  # 反射x
                        elif axis == 1 and math.sin(self.psi_e) > 0:
                            self.psi_e = -self.psi_e  # 反射y
                
                # 负方向边界
                if pos < -soft_start:
                    ratio = max((hard_limit + pos) / (hard_limit - soft_start), 0.0)
                    self.speed_e *= (0.3 + 0.7 * ratio)
                    if pos < -hard_limit:
                        self.e[axis] = -hard_limit
                        self.speed_e *= 0.2
                        if axis == 0 and math.cos(self.psi_e) < 0:
                            self.psi_e = math.pi - self.psi_e
                        elif axis == 1 and math.sin(self.psi_e) < 0:
                            self.psi_e = -self.psi_e
            
            # 同步速度向量
            self.speed_e = max(self.speed_e, 0.0)
            self.ve = np.array([
                self.speed_e * math.cos(self.psi_e),
                self.speed_e * math.sin(self.psi_e)
            ])
    
    def _high_obs(self):
        """
        高层观测向量（包含加速度）
        
        观测内容:
        - rel_pos: 相对位置 (2D)
        - vp: 追踪者速度 (2D)
        - ve: 逃跑者速度 (2D)
        - ap: 追踪者加速度 (2D)
        - ae: 逃跑者加速度 (2D)
        - dist: 距离 (1D)
        - cos_angle: 速度对齐角 (1D)
        - boundary_dist: 边界距离 (1D)
        
        - ray_dists: 8方向射线距离 (8D, 归一化到0-1)
        
        总计: 21维
        """
        eta = self.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        e = self.e
        
        rel = e - p
        rel_norm = rel / max(np.linalg.norm(rel), 1e-6)
        
        vp = self._get_pursuer_vel()
        ve = self.ve
        
        ap = self._get_pursuer_acc()
        ae = self.ae
        
        dist = np.linalg.norm(rel)
        
        vp_norm = vp / max(np.linalg.norm(vp), 1e-6)
        cos_angle = float(np.dot(vp_norm, rel_norm))
        
        boundary_dists = [
            p[0] + self.half, self.half - p[0],
            p[1] + self.half, self.half - p[1]
        ]
        boundary_dist = min(boundary_dists) / self.half
        
        # 8方向射线距离 (追踪者位置出发的LIDAR)
        ray_range = 8.0
        n_rays = 8
        ray_dists = np.zeros(n_rays, dtype=np.float32)
        for i in range(n_rays):
            angle = i * (2 * math.pi / n_rays)
            direction = np.array([math.cos(angle), math.sin(angle)])
            ray_dists[i] = self._ray_free_dist(p, direction, ray_range) / ray_range
        
        obs = np.concatenate([
            np.array([
                rel[0] / self.world_size,    # 相对位置 x
                rel[1] / self.world_size,    # 相对位置 y
                vp[0] / self.v_max_e,        # 追踪者速度 x
                vp[1] / self.v_max_e,        # 追踪者速度 y
                ve[0] / self.v_max_e,        # 逃跑者速度 x
                ve[1] / self.v_max_e,        # 逃跑者速度 y
                ap[0] / self.a_max_e,        # 追踪者加速度 x
                ap[1] / self.a_max_e,        # 追踪者加速度 y
                ae[0] / self.a_max_e,        # 逃跑者加速度 x
                ae[1] / self.a_max_e,        # 逃跑者加速度 y
                dist / self.world_size,      # 距离
                cos_angle,                    # 速度对齐角
                boundary_dist                 # 边界距离归一化
            ], dtype=np.float32),
            ray_dists                         # 8射线距离 (归一化)
        ])
        
        return obs
    
    def step(self, action):
        """
        高层动作: (Δx, Δy) ∈ [-1, 1]^2 → 实际位移 ∈ [-range, range]^2
        
        动作被解释为子目标相对于当前位置的位移
        """
        self.t += 1
        
        # 保存当前速度用于下一步加速度计算
        self.prev_vp = self._get_pursuer_vel()
        
        eta = self.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        init_dist = np.linalg.norm(p - self.e)
        
        # ==== 解析2D动作：相对位移 ====
        # action = [dx_norm, dy_norm] ∈ [-1, 1]^2
        dx = float(action[0]) * self.subgoal_range  # 实际x位移
        dy = float(action[1]) * self.subgoal_range  # 实际y位移
        
        # 子目标 = 当前位置 + 位移
        subgoal = np.array([p[0] + dx, p[1] + dy])
        
        # 边界限制
        subgoal = np.clip(subgoal, -self.half + 0.5, self.half - 0.5)
        
        # 注意: 不再对子目标做避障修正 (_push_subgoal_out_of_obstacles 已移除)
        # 避障通过奖励函数中的 r_sg_obs 惩罚项学习
        
        self.current_subgoal = subgoal
        # 设置AUV目标
        self.auv_env.inner.target_state = np.array([
            subgoal[0], subgoal[1], self.auv_env.fixed_z,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        
        # ==== 执行low_steps步底层控制 ====
        captured = False
        out_of_bounds = False
        collision = False
        sub_traj_p = []
        sub_traj_e = []
        sub_vp = []
        sub_ve = []
        
        for _ in range(self.low_steps):
            # 底层观测
            low_obs = self.auv_env._get_observation()
            low_obs_t = torch.tensor(low_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                low_action = self.nn_policy(low_obs_t).cpu().numpy().flatten()
            
            # 记录追踪者移动前的位置
            eta_before = self.auv_env.inner.dynamics.eta
            p_before = np.array([eta_before[0], eta_before[1]])
            
            # 执行底层动作
            self.auv_env.step(low_action)
            
            # 追踪者碰撞检测：防止穿过障碍物
            eta = self.auv_env.inner.dynamics.eta
            p_after = np.array([eta[0], eta[1]])
            nu = self.auv_env.inner.dynamics.nu
            psi = eta[5]
            vp_world = np.array([nu[0]*math.cos(psi) - nu[1]*math.sin(psi),
                                 nu[0]*math.sin(psi) + nu[1]*math.cos(psi)])
            p_resolved, vp_resolved = self._resolve_collisions(p_before, p_after, vp_world)
            if not np.allclose(p_after, p_resolved):
                # 碰撞发生，更新AUV状态（不终止episode，只停在墙边继续）
                eta[0] = p_resolved[0]
                eta[1] = p_resolved[1]
                # 将世界坐标速度转回体坐标
                nu[0] = vp_resolved[0]*math.cos(psi) + vp_resolved[1]*math.sin(psi)
                nu[1] = -vp_resolved[0]*math.sin(psi) + vp_resolved[1]*math.cos(psi)
                collision = True
            
            # 同步更新逃跑者
            self._step_evader(num_steps=1)
            
            # 记录子步轨迹
            eta = self.auv_env.inner.dynamics.eta
            p = np.array([eta[0], eta[1]])
            e = self.e
            sub_traj_p.append(p.copy())
            sub_traj_e.append(e.copy())
            sub_vp.append(self._get_pursuer_vel().copy())
            sub_ve.append(self.ve.copy())
            
            # 检查捕获
            dist = np.linalg.norm(p - e)
            
            if dist < self.catch_radius:
                captured = True
                break
            
            # 检查边界
            if abs(p[0]) > self.half or abs(p[1]) > self.half:
                out_of_bounds = True
                break
        
        # ==== 计算奖励 ====
        eta = self.auv_env.inner.dynamics.eta
        p = np.array([eta[0], eta[1]])
        e = self.e
        final_dist = np.linalg.norm(p - e)
        
        reward = self._compute_reward(
            captured, out_of_bounds, collision,
            init_dist, final_dist, p, e
        )
        
        self.last_dist = final_dist
        self.prev_positions.append(p.copy())
        if len(self.prev_positions) > 20:
            self.prev_positions.pop(0)
        
        done = captured or out_of_bounds or self.t >= self.max_steps // self.low_steps
        
        info = {
            'captured': captured,
            'out_of_bounds': out_of_bounds,
            'collision': collision,
            'evader_collision_hits': self.evader_collision_hits,
            'evader_mode': self.evader_mode,
            'steps': self.t,
            'final_dist': final_dist,
            'corrected_subgoal': self.current_subgoal.copy(),
            'sub_traj_p': sub_traj_p,
            'sub_traj_e': sub_traj_e,
            'sub_vp': sub_vp,
            'sub_ve': sub_ve,
        }
        
        return self._high_obs(), reward, done, info
    
    def _compute_reward(self, captured, out_of_bounds, collision,
                        init_dist, final_dist, p, e):
        """
        奖励设计 - 对齐文档
        
        主要成分:
        1. 捕获奖励: +100 + time_bonus
        2. 距离变化奖励: (init_dist - final_dist) * 距离系数
        3. 速度方向惩罚: 鼓励速度指向目标
        4. 边界惩罚
        5. 出界/碰撞惩罚
        """
        reward = 0.0
        
        # 捕获奖励
        if captured:
            max_high_steps = self.max_steps // self.low_steps
            time_bonus = max(0, (max_high_steps - self.t) / max_high_steps)
            reward = 100.0 + 50.0 * time_bonus
            return reward
        
        # 出界惩罚
        if out_of_bounds:
            reward = -50.0
            return reward
        
        # 碰撞惩罚（非终止性，小惩罚鼓励绕开障碍物）
        if collision:
            reward -= 2.0
        
        # ==== 改进的过程奖励 ====
        
        # 1. 距离变化奖励 (增大系数)
        dist_change = init_dist - final_dist
        reward += dist_change * 10.0  # 增大系数
        
        # 2. 基于距离的势能惩罚 (鼓励保持近距离)
        gamma = 0.99
        potential_init = -init_dist
        potential_final = -final_dist
        shaping_reward = gamma * potential_final - potential_init
        reward += shaping_reward * 2.0  # 势能shaping
        
        # 3. 拦截角奖励（速度方向指向目标）
        vp = self._get_pursuer_vel()
        rel = e - p
        rel_norm = rel / max(np.linalg.norm(rel), 1e-6)
        vp_norm = vp / max(np.linalg.norm(vp), 1e-6)
        cos_angle = float(np.dot(vp_norm, rel_norm))
        reward += cos_angle * 1.0  # 增大系数
        
        # 4. 边界惩罚
        boundary_dists = [
            p[0] + self.half, self.half - p[0],
            p[1] + self.half, self.half - p[1]
        ]
        min_boundary = min(boundary_dists)
        if min_boundary < self.boundary_margin:
            boundary_penalty = 0.5 * (1 - min_boundary / self.boundary_margin)
            reward -= boundary_penalty
        
        # 5. 接近奖励（终盘阶段，增强）
        if final_dist < 5.0:
            reward += (5.0 - final_dist) * 0.5
        if final_dist < 2.0:
            reward += (2.0 - final_dist) * 1.0
        
        # 6. 循环惩罚
        if len(self.prev_positions) >= 10:
            recent = np.array(self.prev_positions[-10:])
            movement = np.linalg.norm(recent[-1] - recent[0])
            if movement < 0.5:
                reward -= 1.0
        
        return reward
    
    @property
    def high_obs_dim(self):
        """高层观测维度: 13(原始) + 8(射线LIDAR)"""
        return 21
    
    @property
    def high_act_dim(self):
        """高层动作维度（2D位移）"""
        return 2


# ==================== PPO策略（2D动作输出） ====================
class HighLevelPolicy2D(nn.Module):
    """
    高层策略网络 - 输出2D子目标位移
    
    输出: (Δx, Δy) ∈ [-1, 1]^2 (归一化)
    """
    
    def __init__(self, obs_dim=13, act_dim=2, hidden=(128, 128)):
        super().__init__()
        
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.body = nn.Sequential(*layers)
        
        self.mu = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        self.log_std_min = -1.0
        self.log_std_max = 1.0
    
    def forward(self, x):
        h = self.body(x)
        mu = torch.tanh(self.mu(h))  # [-1, 1]
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std


class HighLevelCritic(nn.Module):
    def __init__(self, obs_dim=13, hidden=(128, 128)):
        super().__init__()
        
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = 0
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages


def ppo_update(actor, critic, actor_opt, critic_opt, 
               obs_batch, act_batch, ret_batch, adv_batch, old_logp_batch,
               clip_ratio=0.2, entropy_coef=0.02):
    """PPO更新（2D动作）"""
    
    # 确保batch size >= 2
    if obs_batch.shape[0] < 2:
        return 0.0, 0.0, 0.0
    
    # 优势归一化
    adv_std = adv_batch.std()
    if adv_std > 1e-8:
        adv_batch = (adv_batch - adv_batch.mean()) / (adv_std + 1e-8)
    
    mu, std = actor(obs_batch)
    
    # 检查NaN
    if torch.isnan(mu).any() or torch.isnan(std).any():
        return 0.0, 0.0, 0.0
    
    dist = torch.distributions.Normal(mu, std)
    log_prob = dist.log_prob(act_batch).sum(-1)  # 2D动作求和
    entropy = dist.entropy().sum(-1).mean()
    
    ratio = (log_prob - old_logp_batch).exp()
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_batch
    actor_loss = -torch.min(ratio * adv_batch, clip_adv).mean() - entropy_coef * entropy
    
    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
    actor_opt.step()
    
    values = critic(obs_batch).view(-1)  # 使用view确保维度正确
    critic_loss = F.mse_loss(values, ret_batch)
    
    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
    critic_opt.step()
    
    return actor_loss.item(), critic_loss.item(), entropy.item()


def collect_episode(env, actor, device):
    """收集单个episode的数据（2D动作）"""
    obs = env.reset()
    
    obs_list, act_list, rew_list, val_list, logp_list, done_list = [], [], [], [], [], []
    
    done = False
    
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, std = actor(obs_t)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()  # (1, 2)
            log_prob = dist.log_prob(action).sum(-1)  # 2D求和
            
            action_np = action.cpu().numpy().flatten()
            action_np = np.clip(action_np, -1, 1)
            
            next_obs, reward, done, info = env.step(action_np)
            
            obs_list.append(obs)
            act_list.append(action_np)
            rew_list.append(reward)
            logp_list.append(log_prob.item())
            done_list.append(float(done))
            
            obs = next_obs
    
    return obs_list, act_list, rew_list, logp_list, done_list, info


# ==================== 可视化 ====================
def visualize_episode(env, actor, device, save_path, obs_dim=13):
    """可视化一个episode"""
    obs = env.reset()
    
    eta = env.auv_env.inner.dynamics.eta
    traj_p = [np.array([eta[0], eta[1]])]
    traj_e = [env.e.copy()]
    subgoals = []
    
    done = False
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, _ = actor(obs_t)
            action = mu.cpu().numpy().flatten()
            action = np.clip(action, -1, 1)
            
            # 记录子目标
            eta = env.auv_env.inner.dynamics.eta
            p = np.array([eta[0], eta[1]])
            dx = action[0] * env.subgoal_range
            dy = action[1] * env.subgoal_range
            subgoal = np.array([p[0] + dx, p[1] + dy])
            subgoals.append(subgoal.copy())
            
            obs, reward, done, info = env.step(action)
            
            eta = env.auv_env.inner.dynamics.eta
            traj_p.append(np.array([eta[0], eta[1]]))
            traj_e.append(env.e.copy())
    
    traj_p = np.array(traj_p)
    traj_e = np.array(traj_e)
    subgoals = np.array(subgoals) if subgoals else np.zeros((0, 2))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    half = env.half
    ax.set_xlim(-half - 1, half + 1)
    ax.set_ylim(-half - 1, half + 1)
    ax.set_aspect('equal')
    
    ax.axhline(-half, color='black', linewidth=2)
    ax.axhline(half, color='black', linewidth=2)
    ax.axvline(-half, color='black', linewidth=2)
    ax.axvline(half, color='black', linewidth=2)
    
    # 边界惩罚区域 (淡红色)
    margin = env.boundary_margin
    ax.fill_between([-half, half], [half-margin, half-margin], [half, half], color='red', alpha=0.1)
    ax.fill_between([-half, half], [-half, -half], [-half+margin, -half+margin], color='red', alpha=0.1)
    ax.fill_betweenx([-half+margin, half-margin], [-half, -half], [-half+margin, -half+margin], color='red', alpha=0.1)
    ax.fill_betweenx([-half+margin, half-margin], [half-margin, half-margin], [half, half], color='red', alpha=0.1)
    
    for obs in env.obstacles:
        rect = Rectangle(
            (obs.cx - obs.w/2, obs.cy - obs.h/2),
            obs.w, obs.h, facecolor='gray', edgecolor='black', alpha=0.5
        )
        ax.add_patch(rect)
    
    ax.plot(traj_p[:, 0], traj_p[:, 1], 'b-', linewidth=1.5, label='Pursuer')
    ax.plot(traj_e[:, 0], traj_e[:, 1], 'r-', linewidth=1.5, label='Evader')
    
    if len(subgoals) > 0:
        ax.scatter(subgoals[:, 0], subgoals[:, 1], c='green', s=20, alpha=0.5, label='Subgoals')
    
    ax.plot(traj_p[0, 0], traj_p[0, 1], 'bo', markersize=10, label='P Start')
    ax.plot(traj_e[0, 0], traj_e[0, 1], 'ro', markersize=10, label='E Start')
    ax.plot(traj_p[-1, 0], traj_p[-1, 1], 'bs', markersize=10)
    ax.plot(traj_e[-1, 0], traj_e[-1, 1], 'rs', markersize=10)
    
    catch_circle = plt.Circle(
        (traj_p[-1, 0], traj_p[-1, 1]), env.catch_radius,
        fill=False, color='green', linewidth=2, linestyle='--', label='Catch radius'
    )
    ax.add_patch(catch_circle)
    
    ax.legend(loc='upper right')
    
    title = f"Steps: {info['steps']}"
    if info['captured']:
        title += " | CAPTURED"
    elif info['out_of_bounds']:
        title += " | OUT OF BOUNDS"
    elif info['collision']:
        title += " | COLLISION"
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return info


# ==================== 训练主函数 ====================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载V4 NN
    nn_path = os.path.join(PROJECT_ROOT, "low_near/v4_tight_vel10_v2/checkpoints/best.pth")
    nn_policy = load_v4_nn(nn_path, device)
    
    # 创建环境
    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy,
        device=device,
        world_size=30.0,
        catch_radius=1.0,
        max_steps=6000,
        subgoal_range=args.subgoal_range,
        low_steps=args.low_steps,
        seed=args.seed,
        evader_mode=args.evader_mode,
    )
    
    obs_dim = env.high_obs_dim
    act_dim = env.high_act_dim
    print(f"High-level: obs_dim={obs_dim}, act_dim={act_dim}")
    
    # 创建策略
    actor = HighLevelPolicy2D(obs_dim, act_dim, hidden=(256, 128)).to(device)
    critic = HighLevelCritic(obs_dim, hidden=(256, 128)).to(device)
    
    actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr)
    
    # LR调度器（余弦退火，按更新次数）
    total_episodes = args.episodes
    total_updates = max(total_episodes // 4, 1)  # 每4 episode一次更新
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_opt, T_max=total_updates, eta_min=args.lr * 0.01
    )
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        critic_opt, T_max=total_updates, eta_min=args.lr * 0.01
    )
    
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    log_file = os.path.join(save_dir, "training_log.txt")
    
    gamma = args.gamma
    lam = 0.95
    
    best_capture_rate = 0.0
    capture_history = []
    reward_history = []
    
    # Early stopping
    no_improve_count = 0
    patience = 500
    
    with open(log_file, 'w') as f:
        f.write("Episode,Captured,Steps,Reward,CaptureRate,LR\n")
    
    # === 多episode rollout + mini-batch PPO ===
    eps_per_update = 4   # 每次收集4个episode再更新
    mini_batch_size = 64  # mini-batch大小
    
    ep_count = 0
    update_count = 0
    total_updates = total_episodes // eps_per_update
    
    while ep_count < total_episodes:
        # 收集多个episode
        all_obs, all_act, all_rew, all_logp, all_done = [], [], [], [], []
        ep_infos = []
        
        for _ in range(eps_per_update):
            ep_count += 1
            obs_list, act_list, rew_list, logp_list, done_list, info = collect_episode(env, actor, device)
            
            captured = info['captured']
            capture_history.append(1 if captured else 0)
            ep_reward = sum(rew_list)
            reward_history.append(ep_reward)
            ep_infos.append(info)
            
            if len(capture_history) > 200:
                capture_history.pop(0)
                reward_history.pop(0)
            
            if len(obs_list) >= 2:
                all_obs.extend(obs_list)
                all_act.extend(act_list)
                all_rew.extend(rew_list)
                all_logp.extend(logp_list)
                all_done.extend(done_list)
        
        capture_rate = np.mean(capture_history[-200:]) * 100
        avg_reward = np.mean(reward_history[-200:])
        
        # 跳过空数据
        if len(all_obs) < 4:
            continue
        
        # GAE计算
        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
        with torch.no_grad():
            values = critic(obs_t).squeeze().cpu().numpy()
        values = np.atleast_1d(values)
        
        advantages = compute_gae(all_rew, values.tolist(), all_done, gamma, lam)
        returns = [adv + val for adv, val in zip(advantages, values.tolist())]
        
        act_t = torch.tensor(np.array(all_act), dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        logp_t = torch.tensor(all_logp, dtype=torch.float32, device=device)
        
        # Mini-batch PPO 更新
        n = obs_t.shape[0]
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(n, device=device)
            for mb_start in range(0, n, mini_batch_size):
                mb_end = min(mb_start + mini_batch_size, n)
                idx = perm[mb_start:mb_end]
                if len(idx) < 2:
                    continue
                ppo_update(
                    actor, critic, actor_opt, critic_opt,
                    obs_t[idx], act_t[idx], ret_t[idx], adv_t[idx], logp_t[idx],
                    clip_ratio=0.2, entropy_coef=args.entropy_coef
                )
        
        update_count += 1
        actor_scheduler.step()
        critic_scheduler.step()
        current_lr = actor_scheduler.get_last_lr()[0]
        
        # 日志
        if ep_count % 20 == 0 or ep_count <= 20:
            last_info = ep_infos[-1]
            status = "CAUGHT" if last_info['captured'] else "MISSED"
            print(f"[{ep_count:5d}] {status} | Steps={last_info['steps']:4d} | "
                  f"Reward={reward_history[-1]:7.2f} | Rate={capture_rate:5.1f}% | "
                  f"AvgR={avg_reward:7.2f} | LR={current_lr:.2e} | buf={n}")
        
        with open(log_file, 'a') as f:
            for i, info in enumerate(ep_infos):
                ep_num = ep_count - eps_per_update + i + 1
                cap = 1 if info['captured'] else 0
                f.write(f"{ep_num},{cap},{info['steps']},{sum(all_rew)/eps_per_update:.2f},{capture_rate:.1f},{current_lr:.2e}\n")
        
        # 保存最优
        if capture_rate > best_capture_rate and ep_count >= 100:
            best_capture_rate = capture_rate
            no_improve_count = 0
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'episode': ep_count,
                'capture_rate': capture_rate,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
            }, os.path.join(save_dir, "best.pth"))
            print(f"  -> Best model saved: {capture_rate:.1f}%")
        else:
            no_improve_count += 1
        
        # 定期保存
        if ep_count % args.save_freq == 0:
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'episode': ep_count,
                'capture_rate': capture_rate,
            }, os.path.join(save_dir, f"ckpt_ep{ep_count}.pth"))
            
            vis_path = os.path.join(save_dir, f"traj_ep{ep_count}.png")
            visualize_episode(env, actor, device, vis_path, obs_dim)
    
    print(f"\nTraining completed. Best capture rate: {best_capture_rate:.1f}%")
    
    final_vis_path = os.path.join(save_dir, "final_trajectory.png")
    visualize_episode(env, actor, device, final_vis_path, obs_dim)


# ==================== 测试 ====================
def test_direct_pursuit(args):
    """测试直接追踪（动作始终朝向目标）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nn_path = os.path.join(PROJECT_ROOT, "low_near/v4_tight_vel10_v2/checkpoints/best.pth")
    nn_policy = load_v4_nn(nn_path, device)
    
    env = PursuitEnvRealAUVFixed(
        nn_policy=nn_policy,
        device=device,
        world_size=30.0,
        catch_radius=1.0,
        max_steps=6000,
        subgoal_range=args.subgoal_range,
        low_steps=args.low_steps,
        seed=args.seed,
        evader_mode=args.evader_mode,
    )
    
    num_tests = 10
    results = []
    
    print("=== Testing Direct Pursuit (action toward evader) ===")
    for i in range(num_tests):
        env.rng = np.random.default_rng(args.seed + i)
        obs = env.reset()
        
        done = False
        while not done:
            # 动作直接指向逃跑者
            eta = env.auv_env.inner.dynamics.eta
            p = np.array([eta[0], eta[1]])
            e = env.e
            direction = e - p
            dist = np.linalg.norm(direction)
            
            if dist > 1e-6:
                # 归一化位移
                dx = direction[0] / max(dist, env.subgoal_range) * env.subgoal_range
                dy = direction[1] / max(dist, env.subgoal_range) * env.subgoal_range
                action = np.array([dx / env.subgoal_range, dy / env.subgoal_range])
            else:
                action = np.array([0.0, 0.0])
            
            action = np.clip(action, -1, 1)
            obs, reward, done, info = env.step(action)
        
        results.append(info)
        status = "CAUGHT" if info['captured'] else "MISSED"
        print(f"  Test {i+1}: {status}, steps={info['steps']}, final_dist={info['final_dist']:.2f}")
    
    capture_count = sum(1 for r in results if r['captured'])
    print(f"\nDirect pursuit: {capture_count}/{num_tests} captured ({capture_count/num_tests*100:.0f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--run_name', type=str, default='3d_v3_fixed_smoke')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy_coef', type=float, default=0.02)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--subgoal_range', type=float, default=2.0, 
                        help='Max subgoal displacement (m)')
    parser.add_argument('--evader_mode', type=str, default='medium',
                        choices=['simple', 'medium', 'hard'],
                        help='Fixed evader policy difficulty')
    parser.add_argument('--low_steps', type=int, default=50,
                        help='Low-level steps per high-level action')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        test_direct_pursuit(args)


if __name__ == '__main__':
    main()
