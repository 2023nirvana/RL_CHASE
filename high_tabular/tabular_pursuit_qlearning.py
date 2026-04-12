#!/usr/bin/env python3
"""Tabular Q-learning pursuit-evasion with rectangular obstacles.

This script is intentionally simple and interpretable:
- State is discretized geometry (distance, bearing, heading error, obstacle proximity).
- Action is one of 8 relative steering directions for the pursuer.
- Q-table is a Python dict mapping discrete state -> action values.

It is meant to quickly test whether terrain-aware pursuit can emerge with tabular RL.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import os


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

    def line_intersects(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[bool, float, str]:
        """
        检测线段 p1->p2 是否与矩形相交。
        返回: (是否相交, 最小t参数, 碰撞边) 其中 t 表示碰撞点 = p1 + t*(p2-p1)
        """
        d = p2 - p1
        t_min, t_max = 0.0, 1.0
        hit_edge = None
        
        # 检查X轴方向
        if abs(d[0]) < 1e-10:
            # 线段与X轴平行
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
        
        # 检查Y轴方向
        if abs(d[1]) < 1e-10:
            # 线段与Y轴平行
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
        
        # 只有 t_min 在 (0, 1] 区间内才算有效碰撞
        if t_min > 0 and t_min <= 1.0:
            return True, t_min, hit_edge
        return False, 1.0, None

    def resolve(self, p_old: np.ndarray, p_new: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        碰撞解决：检查线段是否穿过障碍物，如果是则停在障碍物边缘。
        """
        # 首先检查线段是否与矩形相交
        intersects, t_hit, hit_edge = self.line_intersects(p_old, p_new)
        
        if intersects:
            # 碰撞发生，停在碰撞点之前（保留小偏移防止卡住）
            margin = 0.05
            p_fix = p_old + (t_hit - 0.01) * (p_new - p_old)
            v_fix = v.copy()
            
            # 根据碰撞边调整位置和速度
            if hit_edge == 'left':
                p_fix[0] = self.x_min - margin
                v_fix[0] = min(0, v_fix[0]) * -0.1  # 反弹并减速
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
        
        # 没有穿过，但检查终点是否在内部（边界情况）
        if self.contains(p_new):
            d_left = abs(p_new[0] - self.x_min)
            d_right = abs(self.x_max - p_new[0])
            d_bottom = abs(p_new[1] - self.y_min)
            d_top = abs(self.y_max - p_new[1])
            which = int(np.argmin([d_left, d_right, d_bottom, d_top]))
            p_fix = p_new.copy()
            v_fix = v.copy()
            margin = 0.05
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


class PursuitEvasionTabularEnv:
    """
    重新设计的追逐环境，鼓励利用内部地形而非边界围堵。
    
    核心改动：
    1. 更丰富的障碍物布局 - 创造死角、走廊、掩体
    2. 边界惩罚 - 在边界附近捕获会降低奖励
    3. 地形利用奖励 - 在障碍物附近捕获获得额外奖励
    4. 更强的逃跑者边界回避 - 让逃跑者更倾向于在内部活动
    5. 状态空间加入追逐者与障碍物关系
    """
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.L = 15.0
        self.dt = 0.1
        self.max_steps = 150  # 增加步数，允许更复杂的策略
        self.capture_dist = 1.0
        self.escape_dist = 27.0

        # 缩小速度差距，让地形策略更有价值
        self.v_max_p = 1.1
        self.v_max_e = 1.35  # 从1.5降低到1.35
        self.a_max_p = 1.8
        self.a_max_e = 0.55  # 稍微提高逃跑者机动性

        self.n_actions = 8
        self.subgoal_len = 4.5
        self.low_k = 8
        
        # 边界惩罚区域 (距边界多近算"边界区域")
        self.boundary_margin = 3.0
        
        # 更丰富的障碍物布局 - 创造多个死角和走廊
        self.obstacles = [
            # 中央区域 - T型障碍物
            RectObstacle(0.0, 2.0, 8.0, 2.0),      # 横向主体
            RectObstacle(0.0, -2.0, 2.5, 6.0),     # 纵向延伸
            
            # 四个角落附近的障碍物 - 创造L型死角
            RectObstacle(-9.0, 9.0, 4.0, 2.0),     # 左上横
            RectObstacle(-10.0, 7.0, 2.0, 4.0),    # 左上纵
            
            RectObstacle(9.0, 9.0, 4.0, 2.0),      # 右上横
            RectObstacle(10.0, 7.0, 2.0, 4.0),     # 右上纵
            
            RectObstacle(-9.0, -9.0, 4.0, 2.0),    # 左下横
            RectObstacle(-10.0, -7.0, 2.0, 4.0),   # 左下纵
            
            RectObstacle(9.0, -9.0, 4.0, 2.0),     # 右下横
            RectObstacle(10.0, -7.0, 2.0, 4.0),    # 右下纵
            
            # 中间区域的小障碍物 - 创造掩体
            RectObstacle(-5.5, -5.0, 2.5, 2.5),
            RectObstacle(5.5, 5.0, 2.5, 2.5),
        ]

        self.p = np.zeros(2, dtype=np.float32)
        self.e = np.zeros(2, dtype=np.float32)
        self.vp = np.zeros(2, dtype=np.float32)
        self.ve = np.zeros(2, dtype=np.float32)
        self.prev_d = 0.0
        self.step_n = 0
        
        # 防止绕圈的状态变量
        self.prev_action = 0  # 上一个动作
        self.position_history = []  # 位置历史，用于检测绕圈
        self.history_len = 15  # 历史长度
        self.action_history = []  # 动作历史

    @staticmethod
    def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= max_norm or n < 1e-12:
            return v
        return v * (max_norm / n)

    def _inside_world(self, p: np.ndarray) -> bool:
        return abs(p[0]) <= self.L and abs(p[1]) <= self.L

    def _nearest_obstacle_dist(self, p: np.ndarray) -> float:
        return min(obs.distance(p) for obs in self.obstacles)

    def _move_with_collisions(self, p: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """处理移动和碰撞，确保不能穿过障碍物"""
        p_new = p + v * self.dt
        
        # 对每个障碍物检查碰撞，可能需要多次迭代处理连续碰撞
        for _ in range(3):  # 最多3次迭代防止无限循环
            collision_found = False
            for obs in self.obstacles:
                p_old = p_new.copy()
                p_new, v = obs.resolve(p, p_new, v)
                if not np.allclose(p_old, p_new):
                    collision_found = True
            if not collision_found:
                break
        
        # 边界裁剪
        p_new = np.clip(p_new, -self.L, self.L)
        return p_new, v

    def _evader_acc(self) -> np.ndarray:
        """
        更强的逃跑者AI - 极力避免边界，更倾向于在中央区域机动。
        """
        d = self.e - self.p
        n = np.linalg.norm(d)
        run = d / (n + 1e-6)

        avoid_obs = np.zeros(2, dtype=np.float32)
        for obs in self.obstacles:
            q = np.array([np.clip(self.e[0], obs.x_min, obs.x_max), np.clip(self.e[1], obs.y_min, obs.y_max)])
            away = self.e - q
            dn = np.linalg.norm(away)
            if dn < 4.0:
                avoid_obs += away / (dn + 1e-3) * (2.5 / (dn + 0.3))

        # 大幅增强边界回避 - 让逃跑者极力避免靠近边界
        avoid_wall = np.zeros(2, dtype=np.float32)
        margin = 0.55 * self.L  # 更早开始回避
        wall_strength = 3.5  # 大幅增加回避力度
        
        # 距离边界越近，回避力越强 (非线性)
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

        # 向中心的微弱吸引力 (让逃跑者倾向于在中央活动)
        center_pull = -self.e * 0.03

        lateral = np.array([-run[1], run[0]], dtype=np.float32)
        cmd = 1.0 * run + 0.8 * avoid_obs + 1.0 * avoid_wall + center_pull + 0.25 * lateral
        return self._clip_norm(cmd, 1.0) * self.a_max_e
    
    def _is_near_boundary(self, p: np.ndarray) -> bool:
        """检查点是否在边界附近"""
        return (abs(p[0]) > self.L - self.boundary_margin or 
                abs(p[1]) > self.L - self.boundary_margin)
    
    def _boundary_dist(self, p: np.ndarray) -> float:
        """计算到最近边界的距离"""
        return min(self.L - abs(p[0]), self.L - abs(p[1]))

    def _state_discrete(self) -> tuple:
        rel = self.e - self.p
        d = float(np.linalg.norm(rel))

        # 8 distance bins (finer near capture zone)
        d_edges = [1.0, 3.0, 5.0, 8.0, 11.0, 15.0, 20.0]
        d_bin = len(d_edges)
        for i, edge in enumerate(d_edges):
            if d < edge:
                d_bin = i
                break

        # 8-bin bearing to evader
        bearing = math.atan2(rel[1], rel[0])
        b_bin = int(((bearing + math.pi) / (2 * math.pi)) * 8.0) % 8

        # 4-bin heading error (pursuer velocity vs bearing to evader)
        heading = math.atan2(self.vp[1] + 1e-9, self.vp[0] + 1e-9)
        heading_err = (bearing - heading + math.pi) % (2 * math.pi) - math.pi
        h_bin = int(((heading_err + math.pi) / (2 * math.pi)) * 4.0) % 4

        # 4-bin evader flee direction relative to pursuer-evader line
        e_heading = math.atan2(self.ve[1] + 1e-9, self.ve[0] + 1e-9)
        flee_angle = (e_heading - bearing + math.pi) % (2 * math.pi) - math.pi
        ef_bin = int(((flee_angle + math.pi) / (2 * math.pi)) * 4.0) % 4

        # 3-bin evader obstacle proximity: near(<1.5) / mid(<4) / far
        d_obs_e = self._nearest_obstacle_dist(self.e)
        oe_bin = 0 if d_obs_e < 1.5 else (1 if d_obs_e < 4.0 else 2)
        
        # 新增: 3-bin pursuer obstacle proximity (追逐者是否可以利用地形)
        d_obs_p = self._nearest_obstacle_dist(self.p)
        op_bin = 0 if d_obs_p < 2.0 else (1 if d_obs_p < 5.0 else 2)
        
        # 新增: 2-bin 逃跑者是否靠近边界
        eb_bin = 0 if self._is_near_boundary(self.e) else 1
        
        # 新增: 上一个动作 (8个方向，简化为4个区间)
        prev_a_bin = self.prev_action // 2  # 0-3

        # total: 8 * 8 * 4 * 4 * 3 * 3 * 2 * 4 = 73728 states
        return d_bin, b_bin, h_bin, ef_bin, oe_bin, op_bin, eb_bin, prev_a_bin

    def reset(self) -> Tuple[int, int, int, int, int]:
        while True:
            self.p = self.rng.uniform(-0.6 * self.L, 0.6 * self.L, size=2).astype(np.float32)
            self.e = self.rng.uniform(-0.6 * self.L, 0.6 * self.L, size=2).astype(np.float32)
            if np.linalg.norm(self.e - self.p) < 6.0:
                continue
            if any(obs.contains(self.p) or obs.contains(self.e) for obs in self.obstacles):
                continue
            break

        self.vp[:] = 0.0
        self.ve[:] = 0.0
        self.step_n = 0
        self.prev_d = float(np.linalg.norm(self.e - self.p))
        
        # 重置绕圈检测
        self.prev_action = 0
        self.position_history = [self.p.copy()]
        self.action_history = []
        
        return self._state_discrete()

    def step(self, action_idx: int):
        rel = self.e - self.p
        alpha = math.atan2(rel[1], rel[0])
        theta = alpha + action_idx * (2.0 * math.pi / self.n_actions)
        subgoal = self.p + self.subgoal_len * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

        for _ in range(self.low_k):
            # pursuer low-level controller toward subgoal
            a_p = 1.6 * (subgoal - self.p) - 0.9 * self.vp
            a_p = self._clip_norm(a_p, self.a_max_p)
            self.vp = self._clip_norm(self.vp + a_p * self.dt, self.v_max_p)
            self.p, self.vp = self._move_with_collisions(self.p, self.vp)

            # evader heuristic
            a_e = self._evader_acc()
            self.ve = self._clip_norm(self.ve + a_e * self.dt, self.v_max_e)
            self.e, self.ve = self._move_with_collisions(self.e, self.ve)

        self.step_n += 1
        d = float(np.linalg.norm(self.e - self.p))
        done = False
        captured = False
        
        # 计算位置信息
        d_obs_e = self._nearest_obstacle_dist(self.e)
        d_obs_p = self._nearest_obstacle_dist(self.p)
        evader_near_boundary = self._is_near_boundary(self.e)
        pursuer_near_boundary = self._is_near_boundary(self.p)
        capture_pos = self.e.copy()  # 记录捕获位置

        reward = -0.25  # 轻微时间惩罚

        # ============== 防止绕圈惩罚 ==============
        
        # 1. 动作变化惩罚 - 惩罚频繁大幅变向
        action_diff = min(abs(action_idx - self.prev_action), 
                          self.n_actions - abs(action_idx - self.prev_action))
        if action_diff >= 3:  # 变向超过135度
            reward -= 0.8 * action_diff
        elif action_diff >= 2:  # 变向90度左右
            reward -= 0.3 * action_diff
        
        # 2. 绕圈检测惩罚 - 检测是否反复经过相近位置
        self.position_history.append(self.p.copy())
        self.action_history.append(action_idx)
        if len(self.position_history) > self.history_len:
            self.position_history.pop(0)
            self.action_history.pop(0)
        
        # 检测绕圈：当前位置是否接近历史位置
        if len(self.position_history) >= 8:
            loop_penalty = 0.0
            for i, old_pos in enumerate(self.position_history[:-4]):  # 排除最近4步
                dist_to_old = float(np.linalg.norm(self.p - old_pos))
                if dist_to_old < 2.0:  # 回到2m内的旧位置
                    loop_penalty += 0.5 * (2.0 - dist_to_old)
            if loop_penalty > 0:
                reward -= min(loop_penalty, 3.0)  # 上限3.0
        
        # 3. 动作震荡惩罚 - 检测来回切换动作 (A->B->A->B)
        if len(self.action_history) >= 4:
            recent = self.action_history[-4:]
            # 检测 A-B-A-B 或 A-B-A 模式
            if (recent[0] == recent[2] and recent[1] == recent[3] and recent[0] != recent[1]):
                reward -= 1.5  # 来回震荡惩罚
            elif len(set(recent)) <= 2:  # 只在2个动作间切换
                changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
                if changes >= 3:  # 频繁切换
                    reward -= 0.8
        
        # 更新上一个动作
        self.prev_action = action_idx

        # ============== 原有奖励 ==============
        
        # 4. Distance improvement
        improve = self.prev_d - d
        reward += 6.0 * improve

        # 5. 地形利用奖励: 奖励将逃跑者逼向障碍物
        if d_obs_e < 3.0:
            terrain_bonus = 1.2 * (3.0 - d_obs_e)
            reward += terrain_bonus

        # 6. 拦截几何: 追逐者在逃跑者逃跑路径前方
        e_dir = self.ve / (float(np.linalg.norm(self.ve)) + 1e-6)
        p_ahead = self.p - self.e
        intercept_proj = float(np.dot(p_ahead, e_dir))
        if intercept_proj > 0:
            reward += 0.6 * min(intercept_proj, 4.0)
        
        # 7. 追逐者靠近障碍物时的小奖励 (学会利用地形掩护)
        if d_obs_p < 2.5 and d < 8.0:  # 近距离追逐时
            reward += 0.3 * (2.5 - d_obs_p)

        # ============== 捕获奖励设计 ==============
        if d <= self.capture_dist:
            done = True
            captured = True
            
            # 基础捕获奖励
            base_capture_reward = 100.0
            
            # 地形利用奖励: 在障碍物附近捕获获得大额奖励
            if d_obs_e < 2.0:
                terrain_capture_bonus = 80.0 * (2.0 - d_obs_e)  # 最高+160
            elif d_obs_e < 4.0:
                terrain_capture_bonus = 30.0 * (4.0 - d_obs_e) / 2.0  # 中等奖励
            else:
                terrain_capture_bonus = 0.0
            
            # 边界惩罚: 在边界附近捕获会减少奖励
            boundary_dist_e = self._boundary_dist(self.e)
            if boundary_dist_e < self.boundary_margin:
                # 越靠近边界惩罚越重
                boundary_penalty = -60.0 * (1.0 - boundary_dist_e / self.boundary_margin)
            else:
                boundary_penalty = 0.0
            
            # 内部捕获额外奖励: 在地图中央区域捕获
            center_dist = float(np.linalg.norm(self.e))
            if center_dist < 8.0:  # 在中央区域
                center_bonus = 40.0 * (1.0 - center_dist / 8.0)
            else:
                center_bonus = 0.0
            
            reward += base_capture_reward + terrain_capture_bonus + boundary_penalty + center_bonus
            
        elif d >= self.escape_dist or self.step_n >= self.max_steps:
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


def epsilon_by_episode(ep: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    return eps_end + (eps_start - eps_end) * math.exp(-ep / max(1, eps_decay))


def train_tabular(
    episodes: int = 3000,
    alpha: float = 0.15,
    gamma: float = 0.98,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: int = 1200,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)

    env = PursuitEvasionTabularEnv(seed=seed)
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

        if ep % 100 == 0:
            recent_cap = float(np.mean(capture_history[-100:]))
            recent_ret = float(np.mean(reward_history[-100:]))
            print(
                f"[tabular] ep={ep:4d} eps={eps:.3f} recent_capture={recent_cap:.2%} "
                f"recent_return={recent_ret:.2f} q_states={len(q_table)}"
            )

    return env, q_table, capture_history, reward_history


def evaluate(env: PursuitEvasionTabularEnv, q_table, episodes: int = 200, seed: int = 123):
    """评估并收集捕获位置统计"""
    random.seed(seed)
    np.random.seed(seed)

    def get_q(s):
        if s not in q_table:
            return np.zeros(env.n_actions, dtype=np.float32)
        return q_table[s]

    wins = 0
    capture_positions = []
    terrain_captures = 0
    boundary_captures = 0
    
    for _ in range(episodes):
        env_copy = PursuitEvasionTabularEnv(seed=random.randint(0, 100000))
        s = env_copy.reset()
        done = False
        captured = False
        info = {}
        while not done:
            a = int(np.argmax(get_q(s)))
            s, _, done, info = env_copy.step(a)
            captured = captured or bool(info.get("captured", False))
        
        if captured:
            wins += 1
            if info.get("capture_pos") is not None:
                capture_positions.append(info["capture_pos"])
            if info.get("near_terrain", False):
                terrain_captures += 1
            if info.get("near_boundary", False):
                boundary_captures += 1
    
    capture_rate = wins / max(1, episodes)
    terrain_rate = terrain_captures / max(1, wins) if wins > 0 else 0
    boundary_rate = boundary_captures / max(1, wins) if wins > 0 else 0
    
    return {
        "capture_rate": capture_rate,
        "terrain_capture_rate": terrain_rate,
        "boundary_capture_rate": boundary_rate,
        "capture_positions": capture_positions,
        "total_captures": wins,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="results_tabular")
    return p.parse_args()


def plot_training_curves(cap_hist, ret_hist, save_dir, window=100):
    """Plot and save training curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = np.arange(1, len(cap_hist) + 1)
    
    # Compute rolling averages
    def rolling_mean(arr, w):
        arr = np.array(arr)
        result = np.zeros(len(arr))
        for i in range(len(arr)):
            start = max(0, i - w + 1)
            result[i] = np.mean(arr[start:i+1])
        return result
    
    cap_smooth = rolling_mean(cap_hist, window)
    ret_smooth = rolling_mean(ret_hist, window)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Capture rate over episodes
    ax = axes[0, 0]
    ax.plot(episodes, cap_hist, alpha=0.3, color='blue', label='Raw')
    ax.plot(episodes, cap_smooth, color='blue', linewidth=2, label=f'{window}-ep rolling avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Capture Rate')
    ax.set_title('Capture Rate During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # 2. Episode return over episodes
    ax = axes[0, 1]
    ax.plot(episodes, ret_hist, alpha=0.3, color='green', label='Raw')
    ax.plot(episodes, ret_smooth, color='green', linewidth=2, label=f'{window}-ep rolling avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Return')
    ax.set_title('Episode Return During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Capture rate histogram (last 1000 episodes)
    ax = axes[1, 0]
    last_n = min(1000, len(cap_hist))
    recent_caps = cap_hist[-last_n:]
    win_rate = np.mean(recent_caps)
    ax.bar(['Captured', 'Escaped'], [win_rate, 1-win_rate], color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Proportion')
    ax.set_title(f'Last {last_n} Episodes: Capture Rate = {win_rate:.1%}')
    ax.set_ylim(0, 1)
    for i, v in enumerate([win_rate, 1-win_rate]):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)
    
    # 4. Learning stages comparison
    ax = axes[1, 1]
    n_stages = 5
    stage_size = len(cap_hist) // n_stages
    stage_rates = []
    stage_labels = []
    for i in range(n_stages):
        start = i * stage_size
        end = (i + 1) * stage_size if i < n_stages - 1 else len(cap_hist)
        stage_rates.append(np.mean(cap_hist[start:end]))
        stage_labels.append(f'Ep {start+1}-{end}')
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_stages))
    bars = ax.bar(stage_labels, stage_rates, color=colors)
    ax.set_ylabel('Capture Rate')
    ax.set_title('Capture Rate by Training Stage')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=15)
    for bar, rate in zip(bars, stage_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {save_path}")
    return save_path


def visualize_trajectory(env, q_table, save_dir, num_episodes=3, seed=999):
    """Visualize pursuit-evasion trajectories."""
    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    
    def get_q(s):
        if s not in q_table:
            return np.zeros(env.n_actions, dtype=np.float32)
        return q_table[s]
    
    fig, axes = plt.subplots(1, num_episodes, figsize=(5*num_episodes, 5))
    if num_episodes == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        env_copy = PursuitEvasionTabularEnv(seed=seed + idx * 100)
        s = env_copy.reset()
        
        p_traj = [env_copy.p.copy()]
        e_traj = [env_copy.e.copy()]
        
        done = False
        captured = False
        info = {}
        while not done:
            a = int(np.argmax(get_q(s)))
            s, _, done, info = env_copy.step(a)
            p_traj.append(env_copy.p.copy())
            e_traj.append(env_copy.e.copy())
            captured = captured or bool(info.get("captured", False))
        
        p_traj = np.array(p_traj)
        e_traj = np.array(e_traj)
        
        # Plot obstacles
        for obs in env_copy.obstacles:
            rect = plt.Rectangle((obs.x_min, obs.y_min), obs.w, obs.h, 
                                  color='gray', alpha=0.5)
            ax.add_patch(rect)
        
        # Plot boundary margin zone (淡红色)
        margin = env_copy.boundary_margin
        L = env_copy.L
        # 上边界区
        ax.fill_between([-L, L], [L-margin, L-margin], [L, L], color='red', alpha=0.1)
        # 下边界区
        ax.fill_between([-L, L], [-L, -L], [-L+margin, -L+margin], color='red', alpha=0.1)
        # 左边界区
        ax.fill_betweenx([-L+margin, L-margin], [-L, -L], [-L+margin, -L+margin], color='red', alpha=0.1)
        # 右边界区
        ax.fill_betweenx([-L+margin, L-margin], [L-margin, L-margin], [L, L], color='red', alpha=0.1)
        
        # Plot trajectories
        ax.plot(p_traj[:, 0], p_traj[:, 1], 'b-', linewidth=2, label='Pursuer', alpha=0.7)
        ax.plot(e_traj[:, 0], e_traj[:, 1], 'r-', linewidth=2, label='Evader', alpha=0.7)
        
        # Plot start and end points
        ax.scatter(p_traj[0, 0], p_traj[0, 1], c='blue', s=100, marker='o', zorder=5, edgecolor='white')
        ax.scatter(e_traj[0, 0], e_traj[0, 1], c='red', s=100, marker='o', zorder=5, edgecolor='white')
        ax.scatter(p_traj[-1, 0], p_traj[-1, 1], c='blue', s=150, marker='*', zorder=5, edgecolor='white')
        ax.scatter(e_traj[-1, 0], e_traj[-1, 1], c='red', s=150, marker='*', zorder=5, edgecolor='white')
        
        # Arrows showing direction
        for i in range(0, len(p_traj)-1, max(1, len(p_traj)//8)):
            ax.annotate('', xy=p_traj[i+1], xytext=p_traj[i],
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
        for i in range(0, len(e_traj)-1, max(1, len(e_traj)//8)):
            ax.annotate('', xy=e_traj[i+1], xytext=e_traj[i],
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
        
        ax.set_xlim(-env_copy.L - 1, env_copy.L + 1)
        ax.set_ylim(-env_copy.L - 1, env_copy.L + 1)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        
        # 标题显示捕获类型
        if captured:
            near_terrain = info.get("near_terrain", False)
            near_boundary = info.get("near_boundary", False)
            capture_type = "TERRAIN" if near_terrain else ("BOUNDARY" if near_boundary else "OPEN")
            title_color = 'green' if near_terrain else ('orange' if near_boundary else 'blue')
            ax.set_title(f'Ep {idx+1}: CAPTURED ({capture_type})', color=title_color, fontweight='bold')
        else:
            ax.set_title(f'Episode {idx+1}: ESCAPED', color='red', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # World boundary
        ax.plot([-env_copy.L, env_copy.L, env_copy.L, -env_copy.L, -env_copy.L],
                [-env_copy.L, -env_copy.L, env_copy.L, env_copy.L, -env_copy.L],
                'k--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'trajectories.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Trajectory visualization saved to: {save_path}")
    return save_path


def plot_capture_heatmap(env, eval_results, save_dir):
    """绘制捕获位置热力图"""
    os.makedirs(save_dir, exist_ok=True)
    
    capture_positions = eval_results["capture_positions"]
    if len(capture_positions) == 0:
        print("No captures to plot heatmap")
        return None
    
    positions = np.array(capture_positions)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 散点图 + 障碍物
    ax = axes[0]
    
    # 绘制障碍物
    for obs in env.obstacles:
        rect = plt.Rectangle((obs.x_min, obs.y_min), obs.w, obs.h, 
                              color='gray', alpha=0.6, zorder=2)
        ax.add_patch(rect)
    
    # 绘制边界惩罚区
    margin = env.boundary_margin
    L = env.L
    ax.fill_between([-L, L], [L-margin, L-margin], [L, L], color='red', alpha=0.15, label='Boundary zone')
    ax.fill_between([-L, L], [-L, -L], [-L+margin, -L+margin], color='red', alpha=0.15)
    ax.axvline(x=-L+margin, color='red', linestyle=':', alpha=0.3)
    ax.axvline(x=L-margin, color='red', linestyle=':', alpha=0.3)
    ax.axhline(y=-L+margin, color='red', linestyle=':', alpha=0.3)
    ax.axhline(y=L-margin, color='red', linestyle=':', alpha=0.3)
    
    # 绘制捕获位置
    ax.scatter(positions[:, 0], positions[:, 1], c='green', s=30, alpha=0.6, zorder=3, label='Capture')
    
    ax.set_xlim(-L - 1, L + 1)
    ax.set_ylim(-L - 1, L + 1)
    ax.set_aspect('equal')
    ax.set_title(f'Capture Positions (n={len(positions)})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. 统计分析
    ax = axes[1]
    
    terrain_rate = eval_results["terrain_capture_rate"]
    boundary_rate = eval_results["boundary_capture_rate"]
    other_rate = 1.0 - terrain_rate - boundary_rate
    
    categories = ['Near Terrain\n(Goal)', 'Near Boundary\n(Penalized)', 'Open Area']
    values = [terrain_rate, boundary_rate, other_rate]
    colors = ['green', 'red', 'blue']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Proportion of Captures')
    ax.set_title('Capture Location Analysis')
    ax.set_ylim(0, 1)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'capture_analysis.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Capture analysis saved to: {save_path}")
    return save_path


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"Tabular Q-Learning Pursuit-Evasion Training (Terrain Focus)")
    print(f"Episodes: {args.episodes}, Seed: {args.seed}")
    print(f"Alpha: {args.alpha}, Gamma: {args.gamma}")
    print(f"Epsilon: {args.eps_start} -> {args.eps_end} (decay={args.eps_decay})")
    print("=" * 60)
    print("Key Design Changes:")
    print("  - Rich obstacle layout with dead-ends and corridors")
    print("  - Boundary capture PENALTY, terrain capture BONUS")
    print("  - Evader strongly avoids boundaries")
    print("  - State space includes pursuer-obstacle proximity")
    print("=" * 60)
    
    env, q_table, cap_hist, ret_hist = train_tabular(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        seed=args.seed,
    )
    
    # 详细评估
    print("\nEvaluating with capture location analysis...")
    eval_results = evaluate(env, q_table, episodes=args.eval_episodes, seed=args.seed + 17)
    
    print("-" * 60)
    print(f"Train episodes: {args.episodes}")
    print(f"Q-table states: {len(q_table)}")
    print(f"Final 100-ep capture rate: {np.mean(cap_hist[-100:]):.2%}")
    print(f"Greedy eval capture rate:   {eval_results['capture_rate']:.2%}  (over {args.eval_episodes} episodes)")
    print("-" * 60)
    print("Capture Location Breakdown:")
    print(f"  Near Terrain (GOAL):     {eval_results['terrain_capture_rate']:.1%}")
    print(f"  Near Boundary (BAD):     {eval_results['boundary_capture_rate']:.1%}")
    print(f"  Open Area:               {1-eval_results['terrain_capture_rate']-eval_results['boundary_capture_rate']:.1%}")
    print("-" * 60)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(cap_hist, ret_hist, args.save_dir)
    visualize_trajectory(env, q_table, args.save_dir, num_episodes=4, seed=args.seed + 200)
    plot_capture_heatmap(env, eval_results, args.save_dir)
    
    # Save Q-table stats
    import pickle
    qtable_path = os.path.join(args.save_dir, 'q_table.pkl')
    with open(qtable_path, 'wb') as f:
        pickle.dump({
            'q_table': q_table, 
            'cap_hist': cap_hist, 
            'ret_hist': ret_hist,
            'eval_results': eval_results,
        }, f)
    print(f"Q-table saved to: {qtable_path}")
    
    print("\n" + "=" * 60)
    print("Training complete! Check results in:", args.save_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
