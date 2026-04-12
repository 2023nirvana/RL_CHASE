#!/usr/bin/env python3
"""
2D Pursuit-Evasion with Obstacles - Continuous High-Level Control
================================================================

目标：给现有的 xx.py 提供一个更贴近高层 MDP 文档定义的连续动作版本。

关键变化：
1. 高层动作为连续 Box([-1, 1], [-1, 1])，表示相对当前位置的子目标位移。
2. 追击者保留全局可观测优势：可见地形、逃逸者位置与速度。
3. 不使用碰撞反弹，改为“投影到可行边界 + 去除法向内冲速度”的软约束。
4. 每个高层动作触发 K 个低层微步，低层负责平滑跟踪 waypoint。
5. 奖励项与文档中的 high-level MDP 尽量对齐：capture / close / intercept /
   terrain / energy / time。

运行示例：
  python high_tabular/xx_continuous.py --mode smoke
  python high_tabular/xx_continuous.py --mode train --updates 200
"""

from __future__ import annotations

import argparse
import os
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= max_norm or max_norm <= 0.0:
        return v
    if norm < 1e-12:
        return v
    return v * (max_norm / norm)


@dataclass
class SoftRectObstacle:
    cx: float
    cy: float
    w: float
    h: float

    def __post_init__(self):
        self.x_min = self.cx - self.w / 2.0
        self.x_max = self.cx + self.w / 2.0
        self.y_min = self.cy - self.h / 2.0
        self.y_max = self.cy + self.h / 2.0

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        return (
            self.x_min - margin <= p[0] <= self.x_max + margin
            and self.y_min - margin <= p[1] <= self.y_max + margin
        )

    def distance(self, p: np.ndarray) -> float:
        dx = max(self.x_min - p[0], 0.0, p[0] - self.x_max)
        dy = max(self.y_min - p[1], 0.0, p[1] - self.y_max)
        if dx == 0.0 and dy == 0.0:
            return -min(
                p[0] - self.x_min,
                self.x_max - p[0],
                p[1] - self.y_min,
                self.y_max - p[1],
            )
        return math.hypot(dx, dy)

    def nearest_boundary_point(self, p: np.ndarray) -> np.ndarray:
        px = float(np.clip(p[0], self.x_min, self.x_max))
        py = float(np.clip(p[1], self.y_min, self.y_max))
        return np.array([px, py], dtype=np.float64)

    def outward_normal(self, p: np.ndarray) -> np.ndarray:
        bp = self.nearest_boundary_point(p)
        diff = p - bp
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            return diff / norm

        distances = {
            "left": abs(p[0] - self.x_min),
            "right": abs(p[0] - self.x_max),
            "bottom": abs(p[1] - self.y_min),
            "top": abs(p[1] - self.y_max),
        }
        face = min(distances, key=distances.get)
        if face == "left":
            return np.array([-1.0, 0.0], dtype=np.float64)
        if face == "right":
            return np.array([1.0, 0.0], dtype=np.float64)
        if face == "bottom":
            return np.array([0.0, -1.0], dtype=np.float64)
        return np.array([0.0, 1.0], dtype=np.float64)

    def project_outside(self, p: np.ndarray, margin: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        if not self.contains(p):
            normal = self.outward_normal(p)
            return p.copy(), normal

        distances = {
            "left": abs(p[0] - self.x_min),
            "right": abs(p[0] - self.x_max),
            "bottom": abs(p[1] - self.y_min),
            "top": abs(p[1] - self.y_max),
        }
        face = min(distances, key=distances.get)
        out = p.copy()
        if face == "left":
            out[0] = self.x_min - margin
            normal = np.array([-1.0, 0.0], dtype=np.float64)
        elif face == "right":
            out[0] = self.x_max + margin
            normal = np.array([1.0, 0.0], dtype=np.float64)
        elif face == "bottom":
            out[1] = self.y_min - margin
            normal = np.array([0.0, -1.0], dtype=np.float64)
        else:
            out[1] = self.y_max + margin
            normal = np.array([0.0, 1.0], dtype=np.float64)
        return out, normal


class ContinuousHighLevelPursuitEvasion:
    """连续高层动作的 2D 追逃环境。"""

    def __init__(self, cfg: Optional[Dict] = None):
        c = cfg or {}

        self.v_max_p = float(c.get("v_max_p", 1.0))
        self.v_max_e = float(c.get("v_max_e", 1.45))
        self.a_max_p = float(c.get("a_max_p", 2.4))
        self.a_max_e = float(c.get("a_max_e", 0.7))
        self.dt = float(c.get("dt", 0.1))
        self.K = int(c.get("K", 10))
        self.N_max = int(c.get("N_max", 100))
        self.L = float(c.get("L", 15.0))

        default_obs = [
            {"cx": 0.0, "cy": 0.0, "w": 10.0, "h": 2.5},
            {"cx": -6.0, "cy": 6.0, "w": 3.0, "h": 3.0},
            {"cx": 7.0, "cy": -5.0, "w": 3.0, "h": 4.0},
        ]
        self.obstacles = [SoftRectObstacle(**o) for o in c.get("obstacles", default_obs)]

        self.D_c = float(c.get("D_c", 1.0))
        self.D_escape = float(c.get("D_escape", 28.0))
        self.l_high = float(c.get("l_high", 4.5))
        self.agent_radius = float(c.get("agent_radius", 0.25))

        self.R_win = float(c.get("R_win", 220.0))
        self.R_lose = float(c.get("R_lose", 90.0))
        self.w_close = float(c.get("w_close", 10.0))
        self.w_int = float(c.get("w_int", 0.6))
        self.w_terrain = float(c.get("w_terrain", 3.5))
        self.w_edge = float(c.get("w_edge", 3.0))
        self.w_energy = float(c.get("w_energy", 0.08))
        self.c_time = float(c.get("c_time", 0.35))
        self.tau_max = float(c.get("tau_max", 5.0))
        self.edge_margin = float(c.get("edge_margin", 2.6))
        self.edge_obs_exempt = float(c.get("edge_obs_exempt", 2.4))
        self.corner_margin = float(c.get("corner_margin", 2.4))
        self.boundary_capture_penalty = float(c.get("boundary_capture_penalty", 80.0))
        self.evader_boundary_gain = float(c.get("evader_boundary_gain", 2.2))
        self.evader_center_gain = float(c.get("evader_center_gain", 1.4))

        self.n_rays = int(c.get("n_rays", 8))
        self.ray_max = float(c.get("ray_max", 15.0))
        self.obs_dim = 5 + 4 + 5 + self.n_rays * 2 + 1
        self.action_dim = 2

        self.evader_policy = str(c.get("evader_policy", "smart_escape"))
        self.goal_tol = float(c.get("goal_tol", 0.4))
        self.prev_action = np.zeros(2, dtype=np.float64)

        self.reset()

    def _in_any_obstacle(self, p: np.ndarray, margin: float = 0.0) -> bool:
        return any(obs.contains(p, margin=margin) for obs in self.obstacles)

    def _min_obs_dist(self, p: np.ndarray) -> float:
        if not self.obstacles:
            return self.ray_max
        return min(obs.distance(p) for obs in self.obstacles)

    def _nearest_obstacle(self, p: np.ndarray) -> Optional[SoftRectObstacle]:
        if not self.obstacles:
            return None
        return min(self.obstacles, key=lambda obs: obs.distance(p))

    def _boundary_distance(self, p: np.ndarray) -> float:
        return float(min(self.L - abs(p[0]), self.L - abs(p[1])))

    def _corner_distance(self, p: np.ndarray) -> float:
        corners = np.array(
            [
                [self.L, self.L],
                [self.L, -self.L],
                [-self.L, self.L],
                [-self.L, -self.L],
            ],
            dtype=np.float64,
        )
        return float(np.min(np.linalg.norm(corners - p[None, :], axis=1)))

    def _edge_trap_score(self, p: np.ndarray) -> float:
        d_bound = self._boundary_distance(p)
        d_obs = self._min_obs_dist(p)
        d_corner = self._corner_distance(p)
        edge_pressure = max(0.0, self.edge_margin - d_bound) / max(self.edge_margin, 1e-6)
        no_terrain_cover = max(0.0, d_obs - self.edge_obs_exempt) / max(self.ray_max - self.edge_obs_exempt, 1e-6)
        corner_pressure = max(0.0, self.corner_margin - d_corner) / max(self.corner_margin, 1e-6)
        return float(edge_pressure * (0.6 + 0.4 * no_terrain_cover) + 0.6 * corner_pressure)

    def _directional_clearance(self, p: np.ndarray, direction: np.ndarray, lookahead: float = 4.0, steps: int = 8) -> float:
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return 0.0
        direction = direction / norm
        score = 0.0
        for frac in np.linspace(0.25, 1.0, steps):
            test = p + direction * (lookahead * frac)
            d_bound = self._boundary_distance(test)
            d_obs = self._min_obs_dist(test)
            score += min(d_bound, d_obs)
            if abs(test[0]) >= self.L or abs(test[1]) >= self.L:
                break
        return float(score)

    def _interior_bias(self, p: np.ndarray) -> np.ndarray:
        d_bound = self._boundary_distance(p)
        if d_bound >= self.edge_margin:
            return np.zeros(2, dtype=np.float64)
        toward_center = -p
        norm = np.linalg.norm(toward_center)
        if norm < 1e-8:
            return np.zeros(2, dtype=np.float64)
        strength = (self.edge_margin - d_bound) / max(self.edge_margin, 1e-6)
        return self.evader_center_gain * strength * (toward_center / norm)

    def _cast_rays(self, p: np.ndarray, n: int) -> np.ndarray:
        dists = np.full(n, self.ray_max, dtype=np.float32)
        for i in range(n):
            ang = i * 2 * np.pi / n
            direction = np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
            for t_step in np.linspace(0.2, self.ray_max, 60):
                test = p + t_step * direction
                if abs(test[0]) > self.L or abs(test[1]) > self.L:
                    dists[i] = t_step
                    break
                if self._in_any_obstacle(test, margin=self.agent_radius):
                    dists[i] = t_step
                    break
        return dists

    def _boundary_repulsion(self, p: np.ndarray, gain: float = 1.5) -> np.ndarray:
        rep = np.zeros(2, dtype=np.float64)
        margin = self.edge_margin
        for dim in range(2):
            to_pos = self.L - p[dim]
            to_neg = p[dim] + self.L
            if to_pos < margin:
                rep[dim] -= gain * (((margin - to_pos) / margin) ** 2)
            if to_neg < margin:
                rep[dim] += gain * (((margin - to_neg) / margin) ** 2)
        return rep

    def _obstacle_repulsion(self, p: np.ndarray, range_scale: float = 3.5, gain: float = 1.6) -> np.ndarray:
        rep = np.zeros(2, dtype=np.float64)
        for obs in self.obstacles:
            d = obs.distance(p)
            if d < range_scale:
                bp = obs.nearest_boundary_point(p)
                diff = p - bp
                norm = np.linalg.norm(diff)
                if norm < 1e-8:
                    diff = obs.outward_normal(p)
                    norm = np.linalg.norm(diff)
                rep += gain * ((range_scale - d) / range_scale) * (diff / (norm + 1e-8))
        return rep

    def _soft_project(self, p_new: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_out = p_new.copy()
        v_out = v.copy()
        margin = self.agent_radius * 0.5 + 0.02

        for obs in self.obstacles:
            if obs.contains(p_out, margin=self.agent_radius * 0.5):
                p_out, normal = obs.project_outside(p_out, margin=margin)
                inward = float(np.dot(v_out, normal))
                if inward < 0.0:
                    v_out = v_out - inward * normal

        if p_out[0] < -self.L + margin:
            p_out[0] = -self.L + margin
            if v_out[0] < 0.0:
                v_out[0] = 0.0
        elif p_out[0] > self.L - margin:
            p_out[0] = self.L - margin
            if v_out[0] > 0.0:
                v_out[0] = 0.0

        if p_out[1] < -self.L + margin:
            p_out[1] = -self.L + margin
            if v_out[1] < 0.0:
                v_out[1] = 0.0
        elif p_out[1] > self.L - margin:
            p_out[1] = self.L - margin
            if v_out[1] > 0.0:
                v_out[1] = 0.0

        return p_out, v_out

    def _safe_spawn(self) -> np.ndarray:
        for _ in range(500):
            p = np.random.uniform(-self.L * 0.65, self.L * 0.65, size=2)
            if not self._in_any_obstacle(p, margin=0.5):
                return p.astype(np.float64)
        return np.array([self.L * 0.7, self.L * 0.7], dtype=np.float64)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.p_p = self._safe_spawn()
        self.v_p = np.zeros(2, dtype=np.float64)
        self.psi_p = np.random.uniform(-np.pi, np.pi)

        for _ in range(300):
            ang = np.random.uniform(0.0, 2 * np.pi)
            dist = np.random.uniform(5.0, 10.0)
            cand = self.p_p + dist * np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
            if not self._in_any_obstacle(cand, margin=0.5) and np.all(np.abs(cand) < self.L * 0.7):
                self.p_e = cand
                break
        else:
            self.p_e = self._safe_spawn()

        e_ang = np.random.uniform(0.0, 2 * np.pi)
        self.v_e = 0.55 * self.v_max_e * np.array([np.cos(e_ang), np.sin(e_ang)], dtype=np.float64)
        self.psi_e = math.atan2(self.v_e[1], self.v_e[0])

        self.step_n = 0
        self.done = False
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.info = {}
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        dp = self.p_e - self.p_p
        d = np.linalg.norm(dp) + 1e-8
        hat = dp / d
        sp = np.linalg.norm(self.v_p)
        se = np.linalg.norm(self.v_e)
        alpha = np.arctan2(dp[1], dp[0])

        if sp > 1e-6:
            self.psi_p = np.arctan2(self.v_p[1], self.v_p[0])
        if se > 1e-6:
            self.psi_e = np.arctan2(self.v_e[1], self.v_e[0])

        if se > 1e-6:
            cos_b = np.clip(np.dot(self.v_e, dp) / (se * d), -1.0, 1.0)
            beta = np.arccos(cos_b)
        else:
            beta = np.pi / 2.0

        v_close = np.dot(self.v_p - self.v_e, hat)
        delta_psi = wrap_angle(alpha - self.psi_p)

        rays_p = self._cast_rays(self.p_p, self.n_rays) / self.ray_max
        rays_e = self._cast_rays(self.p_e, self.n_rays) / self.ray_max
        d_e_obs = self._min_obs_dist(self.p_e) / self.ray_max

        obs = np.concatenate(
            [
                [
                    self.p_p[0] / self.L,
                    self.p_p[1] / self.L,
                    self.psi_p / np.pi,
                    self.v_p[0] / self.v_max_p,
                    self.v_p[1] / self.v_max_p,
                ],
                [
                    self.p_e[0] / self.L,
                    self.p_e[1] / self.L,
                    self.psi_e / np.pi,
                    se / self.v_max_e,
                ],
                [
                    d / (2.0 * self.L),
                    alpha / np.pi,
                    beta / np.pi,
                    v_close / (self.v_max_p + self.v_max_e + 1e-8),
                    delta_psi / np.pi,
                ],
                rays_p,
                rays_e,
                [d_e_obs],
            ]
        ).astype(np.float32)
        return obs

    def _get_obs_dict(self) -> Dict[str, np.ndarray | float]:
        dp = self.p_e - self.p_p
        d = np.linalg.norm(dp) + 1e-8
        alpha = np.arctan2(dp[1], dp[0])
        return {
            "p_p": self.p_p.copy(),
            "v_p": self.v_p.copy(),
            "p_e": self.p_e.copy(),
            "v_e": self.v_e.copy(),
            "d_pe": float(d),
            "alpha_pe": float(alpha),
            "d_e_obs": float(self._min_obs_dist(self.p_e)),
        }

    def _ctrl_pursuer(self, goal: np.ndarray):
        dp = goal - self.p_p
        dist = np.linalg.norm(dp)
        goal_dir = dp / (dist + 1e-8)

        slow_radius = max(0.8, self.l_high * 0.6)
        target_speed = self.v_max_p * min(1.0, dist / slow_radius)
        desired_vel = goal_dir * target_speed
        desired_vel += 0.35 * self._obstacle_repulsion(self.p_p, range_scale=2.5, gain=1.5)
        desired_vel += 0.25 * self._boundary_repulsion(self.p_p, gain=1.4)
        desired_vel = clip_norm(desired_vel, self.v_max_p)

        dv = desired_vel - self.v_p
        dv = clip_norm(dv, self.a_max_p * self.dt)
        self.v_p = clip_norm(self.v_p + dv, self.v_max_p)

        p_new = self.p_p + self.v_p * self.dt
        self.p_p, self.v_p = self._soft_project(p_new, self.v_p)
        if np.linalg.norm(self.v_p) > 1e-6:
            self.psi_p = np.arctan2(self.v_p[1], self.v_p[0])

    def _ctrl_evader(self):
        dp = self.p_e - self.p_p
        dist = np.linalg.norm(dp) + 1e-8
        away = dp / dist

        if self.evader_policy == "smart_escape":
            force = 1.25 * away
            force += 1.5 * self._obstacle_repulsion(self.p_e, range_scale=4.0, gain=1.8)
            force += self.evader_boundary_gain * self._boundary_repulsion(self.p_e, gain=1.0)
            force += self._interior_bias(self.p_e)

            tang_a = np.array([-away[1], away[0]], dtype=np.float64)
            tang_b = -tang_a
            clearance_a = self._directional_clearance(self.p_e, tang_a, lookahead=4.5)
            clearance_b = self._directional_clearance(self.p_e, tang_b, lookahead=4.5)
            tang = tang_a if clearance_a >= clearance_b else tang_b

            nearest = self._nearest_obstacle(self.p_e)
            if nearest is not None and nearest.distance(self.p_e) < 4.5:
                force += 0.45 * tang
            else:
                force += 0.18 * tang
            desired_dir = force / (np.linalg.norm(force) + 1e-8)
        else:
            desired_dir = away

        desired_vel = desired_dir * self.v_max_e
        dv = desired_vel - self.v_e
        dv = clip_norm(dv, self.a_max_e * self.dt)
        self.v_e = clip_norm(self.v_e + dv, self.v_max_e)

        p_new = self.p_e + self.v_e * self.dt
        self.p_e, self.v_e = self._soft_project(p_new, self.v_e)
        if np.linalg.norm(self.v_e) > 1e-6:
            self.psi_e = np.arctan2(self.v_e[1], self.v_e[0])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float64).reshape(2)
        action = np.clip(action, -1.0, 1.0)

        obs0 = self._get_obs_dict()
        d0 = obs0["d_pe"]
        d_e_obs0 = obs0["d_e_obs"]
        edge_score0 = self._edge_trap_score(self.p_e)

        goal = self.p_p + self.l_high * action
        goal = np.clip(goal, -self.L + 0.1, self.L - 0.1)

        captured = False
        escaped = False
        micro_steps = 0
        for _ in range(self.K):
            micro_steps += 1
            self._ctrl_pursuer(goal)
            self._ctrl_evader()
            d_now = np.linalg.norm(self.p_e - self.p_p)
            if d_now < self.D_c:
                captured = True
                break
            if d_now > self.D_escape:
                escaped = True
                break
            if np.linalg.norm(goal - self.p_p) < self.goal_tol:
                break

        self.step_n += 1
        obs1 = self._get_obs_dict()
        d1 = obs1["d_pe"]
        d_e_obs1 = obs1["d_e_obs"]
        edge_score1 = self._edge_trap_score(self.p_e)

        reward = 0.0
        r_capture = 0.0
        if captured:
            r_capture += self.R_win
        elif escaped or self.step_n >= self.N_max:
            r_capture -= self.R_lose
        reward += r_capture

        r_close = self.w_close * (d0 - d1)
        reward += r_close

        taus = np.linspace(0.0, self.tau_max, 20)
        preds = obs0["p_e"][None, :] + taus[:, None] * obs0["v_e"][None, :]
        dists = np.linalg.norm(obs1["p_p"][None, :] - preds, axis=1)
        r_intercept = -self.w_int * float(min(dists.min(), 20.0))
        reward += r_intercept

        r_terrain = self.w_terrain * (d_e_obs0 - d_e_obs1)
        reward += r_terrain

        r_edge = -self.w_edge * (edge_score1 - edge_score0)
        reward += r_edge

        r_energy = -self.w_energy * float(np.sum((action - self.prev_action) ** 2))
        reward += r_energy

        r_time = -self.c_time
        reward += r_time

        self.prev_action = action.copy()
        terminated = captured or escaped
        truncated = (self.step_n >= self.N_max) and not terminated
        self.done = terminated or truncated
        self.info = {
            "captured": captured,
            "escaped": escaped,
            "step_n": self.step_n,
            "d_pe": d1,
            "d_e_obs": d_e_obs1,
            "micro_steps": micro_steps,
            "reward_capture": r_capture,
            "reward_close": r_close,
            "reward_intercept": r_intercept,
            "reward_terrain": r_terrain,
            "reward_edge": r_edge,
            "reward_energy": r_energy,
            "reward_time": r_time,
            "goal": goal.copy(),
            "edge_score": edge_score1,
        }
        return self._get_obs(), float(reward), terminated, truncated, self.info


class GaussianActorCritic(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 1.0

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 128, 64)):
        super().__init__()
        actor_layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*actor_layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.6)

        critic_layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

    def get_dist(self, x: torch.Tensor) -> torch.distributions.Normal:
        mu = self.mu_head(self.actor_body(x))
        log_std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        dist = self.get_dist(x)
        z = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(z)
        logp = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, logp, entropy

    def evaluate_actions(self, x: torch.Tensor, action: torch.Tensor):
        dist = self.get_dist(x)
        z = self._atanh(action)
        logp = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy


def policy_direct(obs_vec: np.ndarray, env: ContinuousHighLevelPursuitEvasion) -> np.ndarray:
    del obs_vec
    rel = (env.p_e - env.p_p) / max(env.l_high, 1e-6)
    return np.clip(rel, -1.0, 1.0).astype(np.float32)


def policy_intercept(obs_vec: np.ndarray, env: ContinuousHighLevelPursuitEvasion) -> np.ndarray:
    del obs_vec
    dp = env.p_e - env.p_p
    d = np.linalg.norm(dp)
    tau = np.clip(d / max(env.v_max_p, 1e-6) * 0.55, 0.25, env.tau_max)
    pred = env.p_e + tau * env.v_e
    rel = (pred - env.p_p) / max(env.l_high, 1e-6)
    return np.clip(rel, -1.0, 1.0).astype(np.float32)


def policy_terrain_intercept(obs_vec: np.ndarray, env: ContinuousHighLevelPursuitEvasion) -> np.ndarray:
    del obs_vec
    nearest = env._nearest_obstacle(env.p_e)
    pred = env.p_e + np.clip(np.linalg.norm(env.p_e - env.p_p) / max(env.v_max_p, 1e-6) * 0.45, 0.25, env.tau_max) * env.v_e
    target = pred.copy()

    if nearest is not None:
        d_obs = nearest.distance(env.p_e)
        bp = nearest.nearest_boundary_point(env.p_e)
        normal = env.p_e - bp
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        else:
            normal = nearest.outward_normal(env.p_e)
        tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
        if np.dot(tangent, env.v_e) < 0.0:
            tangent = -tangent
        if d_obs < 4.0:
            target = env.p_e + 1.6 * env.v_e + 1.5 * tangent - 0.8 * normal
        else:
            target = pred + 0.6 * tangent

    rel = (target - env.p_p) / max(env.l_high, 1e-6)
    return np.clip(rel, -1.0, 1.0).astype(np.float32)


def run_episode(env: ContinuousHighLevelPursuitEvasion, policy_fn, seed: int = 42):
    obs = env.reset(seed=seed)
    frames = []
    rewards = []
    frames.append((env.p_p.copy(), env.p_e.copy(), env.info.copy(), np.zeros(2, dtype=np.float64)))
    while not env.done:
        action = np.asarray(policy_fn(obs, env), dtype=np.float64)
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        frames.append((env.p_p.copy(), env.p_e.copy(), info.copy(), action.copy()))
        if term or trunc:
            break
    return frames, rewards


def evaluate(cfg: Dict, policy_fn, n: int = 100, label: str = "") -> float:
    env = ContinuousHighLevelPursuitEvasion(cfg)
    cap = 0
    mean_len = []
    for ep in range(n):
        obs = env.reset(seed=1000 + ep)
        steps = 0
        while not env.done:
            action = policy_fn(obs, env)
            obs, _, _, _, info = env.step(action)
            steps += 1
        mean_len.append(steps)
        if info.get("captured"):
            cap += 1
    rate = cap / max(n, 1)
    print(f"  {label:28s} capture={rate:.3f} ({cap}/{n}) mean_len={np.mean(mean_len):.1f}")
    return rate


def evaluate_with_stats(cfg: Dict, policy_fn, n: int = 100) -> Dict[str, float]:
    env = ContinuousHighLevelPursuitEvasion(cfg)
    captures = 0
    rewards = []
    lengths = []
    for ep in range(n):
        obs = env.reset(seed=7000 + ep)
        ep_reward = 0.0
        steps = 0
        info = {}
        while not env.done:
            action = policy_fn(obs, env)
            obs, reward, _, _, info = env.step(action)
            ep_reward += reward
            steps += 1
        captures += int(info.get("captured", False))
        rewards.append(ep_reward)
        lengths.append(steps)
    return {
        "capture_rate": float(captures / max(n, 1)),
        "mean_return": float(np.mean(rewards) if rewards else 0.0),
        "mean_length": float(np.mean(lengths) if lengths else 0.0),
    }


def train_ppo(
    cfg: Dict,
    updates: int = 120,
    n_envs: int = 8,
    n_steps: int = 128,
    batch_size: int = 256,
    n_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    log_interval: int = 10,
    eval_interval: int = 50,
    eval_episodes: int = 40,
    save_prefix: str = "high_tabular/xx_continuous",
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    envs = [ContinuousHighLevelPursuitEvasion(cfg) for _ in range(n_envs)]
    obs_dim = envs[0].obs_dim
    act_dim = envs[0].action_dim
    model = GaussianActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    save_dir = os.path.dirname(save_prefix)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    obs = np.stack([env.reset(seed=seed + i) for i, env in enumerate(envs)])
    caps: List[int] = []
    rews: List[float] = []
    ep_returns: List[float] = []
    ep_lengths: List[int] = []
    running_returns = np.zeros(n_envs, dtype=np.float32)
    running_lengths = np.zeros(n_envs, dtype=np.int32)
    reward_comp_hist: Dict[str, List[float]] = {
        "capture": [],
        "close": [],
        "intercept": [],
        "terrain": [],
        "edge": [],
        "energy": [],
        "time": [],
    }
    history: Dict[str, List[float]] = {
        "update": [],
        "train_capture_rate": [],
        "mean_step_reward": [],
        "mean_episode_return": [],
        "mean_episode_length": [],
        "eval_capture_rate": [],
        "eval_mean_return": [],
        "eval_mean_length": [],
    }

    for update in range(1, updates + 1):
        obs_buf = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for t in range(n_steps):
            obs_buf[t] = obs
            x = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action, logp, _ = model.get_action(x, deterministic=False)
                value = model.get_value(x)

            act_np = action.cpu().numpy()
            act_buf[t] = act_np
            logp_buf[t] = logp.cpu().numpy()
            val_buf[t] = value.cpu().numpy()

            next_obs = []
            for i, env in enumerate(envs):
                obs2, reward, term, trunc, info = env.step(act_np[i])
                rew_buf[t, i] = reward
                done = term or trunc
                done_buf[t, i] = 1.0 if done else 0.0
                running_returns[i] += reward
                running_lengths[i] += 1
                reward_comp_hist["capture"].append(float(info.get("reward_capture", 0.0)))
                reward_comp_hist["close"].append(float(info.get("reward_close", 0.0)))
                reward_comp_hist["intercept"].append(float(info.get("reward_intercept", 0.0)))
                reward_comp_hist["terrain"].append(float(info.get("reward_terrain", 0.0)))
                reward_comp_hist["edge"].append(float(info.get("reward_edge", 0.0)))
                reward_comp_hist["energy"].append(float(info.get("reward_energy", 0.0)))
                reward_comp_hist["time"].append(float(info.get("reward_time", 0.0)))
                if done:
                    caps.append(int(info.get("captured", False)))
                    ep_returns.append(float(running_returns[i]))
                    ep_lengths.append(int(running_lengths[i]))
                    running_returns[i] = 0.0
                    running_lengths[i] = 0
                    next_obs.append(env.reset())
                else:
                    next_obs.append(obs2)
                rews.append(reward)
            obs = np.array(next_obs, dtype=np.float32)

        with torch.no_grad():
            x_last = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            last_val = model.get_value(x_last).cpu().numpy()

        adv = np.zeros_like(rew_buf)
        last_gae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(n_steps)):
            next_non_term = 1.0 - done_buf[t]
            next_val = last_val if t == n_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_val * next_non_term - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            adv[t] = last_gae
        ret = adv + val_buf

        b_obs = torch.tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=DEVICE)
        b_act = torch.tensor(act_buf.reshape(-1, act_dim), dtype=torch.float32, device=DEVICE)
        b_logp = torch.tensor(logp_buf.reshape(-1), dtype=torch.float32, device=DEVICE)
        b_adv = torch.tensor(adv.reshape(-1), dtype=torch.float32, device=DEVICE)
        b_ret = torch.tensor(ret.reshape(-1), dtype=torch.float32, device=DEVICE)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(n_epochs):
            perm = torch.randperm(len(b_obs), device=DEVICE)
            for start in range(0, len(b_obs), batch_size):
                idx = perm[start:start + batch_size]
                logp_new, ent_each = model.evaluate_actions(b_obs[idx], b_act[idx])
                entropy = ent_each.mean()
                value = model.get_value(b_obs[idx])

                ratio = torch.exp(logp_new - b_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_adv[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = F.mse_loss(value, b_ret[idx])
                loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if update % log_interval == 0:
            window = 200
            cap_rate = np.mean(caps[-window:]) if caps else 0.0
            mean_rew = np.mean(rews[-window * n_envs:]) if rews else 0.0
            mean_ep_return = np.mean(ep_returns[-window:]) if ep_returns else 0.0
            mean_ep_len = np.mean(ep_lengths[-window:]) if ep_lengths else 0.0
            comp_window = min(len(reward_comp_hist["capture"]), window * n_steps)
            comp_means = {}
            if comp_window > 0:
                for key, values in reward_comp_hist.items():
                    comp_means[key] = float(np.mean(values[-comp_window:]))
            else:
                for key in reward_comp_hist:
                    comp_means[key] = 0.0

            history["update"].append(float(update))
            history["train_capture_rate"].append(float(cap_rate))
            history["mean_step_reward"].append(float(mean_rew))
            history["mean_episode_return"].append(float(mean_ep_return))
            history["mean_episode_length"].append(float(mean_ep_len))

            print(
                "  "
                f"update={update:4d} "
                f"train_cap={cap_rate:.3f} "
                f"step_rew={mean_rew:+.3f} "
                f"ep_ret={mean_ep_return:+.1f} "
                f"ep_len={mean_ep_len:5.1f}"
            )
            print(
                "    reward_parts: "
                f"cap={comp_means['capture']:+.2f} "
                f"close={comp_means['close']:+.2f} "
                f"int={comp_means['intercept']:+.2f} "
                f"terrain={comp_means['terrain']:+.2f} "
                f"edge={comp_means['edge']:+.2f} "
                f"energy={comp_means['energy']:+.2f} "
                f"time={comp_means['time']:+.2f}"
            )

        if update % eval_interval == 0 or update == updates:
            greedy_policy = greedy_policy_from_model(model)
            eval_stats = evaluate_with_stats(cfg, greedy_policy, n=eval_episodes)
            history["eval_capture_rate"].append(eval_stats["capture_rate"])
            history["eval_mean_return"].append(eval_stats["mean_return"])
            history["eval_mean_length"].append(eval_stats["mean_length"])
            print(
                "    greedy_eval: "
                f"cap={eval_stats['capture_rate']:.3f} "
                f"ret={eval_stats['mean_return']:+.1f} "
                f"len={eval_stats['mean_length']:.1f}"
            )
            ckpt = {
                "model": model.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "cfg": cfg,
                "update": update,
                "history": history,
            }
            torch.save(ckpt, f"{save_prefix}_latest.pth")

    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "cfg": cfg,
            "update": updates,
            "history": history,
        },
        f"{save_prefix}_final.pth",
    )
    return model, history


def plot_training_curves(history: Dict[str, List[float]], filename: str):
    if not history["update"]:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    x = np.array(history["update"])

    axes[0].plot(x, history["train_capture_rate"], label="Train recent capture", lw=1.8)
    eval_x = x[: len(history["eval_capture_rate"])]
    if len(history["eval_capture_rate"]) > 0:
        axes[0].plot(eval_x, history["eval_capture_rate"], label="Greedy eval capture", lw=1.8)
    axes[0].set_ylabel("Capture Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, history["mean_episode_return"], label="Mean episode return", lw=1.8)
    if len(history["eval_mean_return"]) > 0:
        axes[1].plot(eval_x, history["eval_mean_return"], label="Greedy eval return", lw=1.8)
    axes[1].set_ylabel("Return")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, history["mean_episode_length"], label="Mean episode length", lw=1.8)
    if len(history["eval_mean_length"]) > 0:
        axes[2].plot(eval_x, history["eval_mean_length"], label="Greedy eval length", lw=1.8)
    axes[2].set_ylabel("Episode Length")
    axes[2].set_xlabel("Update")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Continuous High-Level PPO Training")
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)


def plot_episode(frames, env: ContinuousHighLevelPursuitEvasion, title: str, filename: str):
    traj_p = np.array([f[0] for f in frames])
    traj_e = np.array([f[1] for f in frames])
    info = frames[-1][2]

    fig, ax = plt.subplots(figsize=(9, 9))
    for obs_rect in env.obstacles:
        rect = patches.Rectangle(
            (obs_rect.x_min, obs_rect.y_min),
            obs_rect.w,
            obs_rect.h,
            linewidth=2,
            edgecolor="black",
            facecolor="gray",
            alpha=0.55,
        )
        ax.add_patch(rect)

    ax.plot(traj_p[:, 0], traj_p[:, 1], "b-o", ms=2, lw=1.6, label="Pursuer")
    ax.plot(traj_e[:, 0], traj_e[:, 1], "r-s", ms=2, lw=1.6, label="Evader")
    ax.plot(*traj_p[0], "b^", ms=12)
    ax.plot(*traj_e[0], "r^", ms=12)
    ax.plot(*traj_p[-1], "bD", ms=9)
    ax.plot(*traj_e[-1], "rD", ms=9)

    circle = plt.Circle(traj_e[-1], env.D_c, fill=False, color="green", ls="--", lw=2)
    ax.add_patch(circle)

    border = patches.Rectangle(
        (-env.L, -env.L),
        2 * env.L,
        2 * env.L,
        fill=False,
        edgecolor="navy",
        lw=2,
        ls="--",
    )
    ax.add_patch(border)

    status = "CAPTURED" if info.get("captured") else ("ESCAPED" if info.get("escaped") else "TIMEOUT")
    ax.set_title(f"{title} | {status}")
    ax.set_xlim(-env.L - 1, env.L + 1)
    ax.set_ylim(-env.L - 1, env.L + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)


def save_gif(frames, env: ContinuousHighLevelPursuitEvasion, title: str, filename: str, fps: int = 8):
    fig, ax = plt.subplots(figsize=(7, 7))
    for obs_rect in env.obstacles:
        rect = patches.Rectangle(
            (obs_rect.x_min, obs_rect.y_min),
            obs_rect.w,
            obs_rect.h,
            linewidth=2,
            edgecolor="black",
            facecolor="gray",
            alpha=0.55,
        )
        ax.add_patch(rect)

    border = patches.Rectangle(
        (-env.L, -env.L),
        2 * env.L,
        2 * env.L,
        fill=False,
        edgecolor="navy",
        lw=2,
        ls="--",
    )
    ax.add_patch(border)

    pursuer_line, = ax.plot([], [], "b-", lw=1.8, label="Pursuer")
    evader_line, = ax.plot([], [], "r-", lw=1.8, label="Evader")
    pursuer_dot, = ax.plot([], [], "bo", ms=6)
    evader_dot, = ax.plot([], [], "ro", ms=6)
    status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    ax.set_xlim(-env.L - 1, env.L + 1)
    ax.set_ylim(-env.L - 1, env.L + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_title(title)

    traj_p = []
    traj_e = []

    def update(i: int):
        p_p, p_e, info, _ = frames[i]
        traj_p.append(p_p)
        traj_e.append(p_e)
        tp = np.array(traj_p)
        te = np.array(traj_e)
        pursuer_line.set_data(tp[:, 0], tp[:, 1])
        evader_line.set_data(te[:, 0], te[:, 1])
        pursuer_dot.set_data([p_p[0]], [p_p[1]])
        evader_dot.set_data([p_e[0]], [p_e[1]])
        if info.get("captured"):
            txt = "CAPTURED"
        elif info.get("escaped"):
            txt = "ESCAPED"
        else:
            txt = f"RUNNING step={info.get('step_n', i)}"
        status_text.set_text(txt)
        return pursuer_line, evader_line, pursuer_dot, evader_dot, status_text

    anim = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / fps), blit=True)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)


def greedy_policy_from_model(model: GaussianActorCritic):
    def _policy(obs_vec: np.ndarray, env: ContinuousHighLevelPursuitEvasion) -> np.ndarray:
        del env
        x = torch.tensor(obs_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.get_action(x, deterministic=True)
        return action[0].cpu().numpy()

    return _policy


def default_config() -> Dict:
    return {
        "v_max_p": 1.0,
        "v_max_e": 1.45,
        "a_max_p": 2.4,
        "a_max_e": 0.7,
        "dt": 0.1,
        "K": 10,
        "N_max": 100,
        "L": 15.0,
        "obstacles": [
            {"cx": 0.0, "cy": 0.0, "w": 10.0, "h": 2.5},
            {"cx": -6.0, "cy": 6.0, "w": 3.0, "h": 3.0},
            {"cx": 7.0, "cy": -5.0, "w": 3.0, "h": 4.0},
        ],
        "D_c": 1.0,
        "D_escape": 28.0,
        "l_high": 4.5,
        "R_win": 220.0,
        "R_lose": 90.0,
        "w_close": 10.0,
        "w_int": 0.6,
        "w_terrain": 3.5,
        "w_edge": 3.0,
        "w_energy": 0.08,
        "c_time": 0.35,
        "tau_max": 5.0,
        "edge_margin": 2.6,
        "edge_obs_exempt": 2.4,
        "corner_margin": 2.4,
        "boundary_capture_penalty": 80.0,
        "evader_boundary_gain": 2.2,
        "evader_center_gain": 1.4,
        "n_rays": 8,
        "ray_max": 15.0,
        "evader_policy": "smart_escape",
    }


def main():
    parser = argparse.ArgumentParser(description="Continuous high-level pursuit-evasion")
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--save-prefix", type=str, default="high_tabular/xx_continuous")
    args = parser.parse_args()

    cfg = default_config()
    print("=" * 70)
    print(" Continuous 2D Pursuit-Evasion on", DEVICE)
    print(" High-level action: Box([-1,1]^2) relative waypoint")
    print(" Collision model: soft projection without rebound")
    print("=" * 70)

    print("\n[1] Baselines")
    evaluate(cfg, policy_direct, n=100, label="Direct Pursuit")
    evaluate(cfg, policy_intercept, n=100, label="Intercept")
    evaluate(cfg, policy_terrain_intercept, n=100, label="Terrain-aware Intercept")

    env_vis = ContinuousHighLevelPursuitEvasion(cfg)
    frames, _ = run_episode(env_vis, policy_terrain_intercept, seed=args.seed)
    plot_episode(frames, env_vis, "Terrain-aware Intercept", "high_tabular/xx_continuous_smoke.png")
    print("  Saved -> high_tabular/xx_continuous_smoke.png")
    if args.save_gif:
        save_gif(frames, env_vis, "Continuous Terrain-aware Intercept", "high_tabular/xx_continuous_smoke.gif")
        print("  Saved -> high_tabular/xx_continuous_smoke.gif")

    if args.mode == "train":
        print("\n[2] PPO Training")
        model, history = train_ppo(
            cfg,
            updates=args.updates,
            seed=args.seed,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            save_prefix=args.save_prefix,
        )
        plot_training_curves(history, f"{args.save_prefix}_training_curves.png")
        print(f"  Saved -> {args.save_prefix}_training_curves.png")
        learned_policy = greedy_policy_from_model(model)
        print("\n[3] Greedy Evaluation")
        evaluate(cfg, learned_policy, n=100, label="PPO (greedy)")
        env_eval = ContinuousHighLevelPursuitEvasion(cfg)
        frames_learned, _ = run_episode(env_eval, learned_policy, seed=args.seed)
        plot_episode(frames_learned, env_eval, "PPO Greedy", f"{args.save_prefix}_ppo.png")
        print(f"  Saved -> {args.save_prefix}_ppo.png")
        if args.save_gif:
            save_gif(frames_learned, env_eval, "Continuous PPO Greedy", f"{args.save_prefix}_ppo.gif")
            print(f"  Saved -> {args.save_prefix}_ppo.gif")

    print("\nDone.")


if __name__ == "__main__":
    main()