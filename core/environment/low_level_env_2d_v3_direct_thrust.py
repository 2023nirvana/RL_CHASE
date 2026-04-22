"""
二维环境 V3 直接推力版本（跳过 PID，直接输出 8 推进器推力）
=============================================================

与 v3_continuous 版本相比：
- 动作空间：8 维连续 [-1, 1]^8，映射到 8 个推进器的推力
- 跳过速度闭环 PID 和姿态 PID，RL 直接控制推进器
- 观测、奖励、任务场景与 v4_tight_heading_terminal 完全一致
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env import LowLevelAUVEnv


class LowLevelAUV2DEnvV3DirectThrust(gym.Env):
    """二维 V3 直接推力环境：RL 直接输出 8 个推进器力。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    NUM_THRUSTERS = 8

    def __init__(
        self,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        curriculum_stage: int = 0,
        enable_domain_randomization: bool = False,
        seed: Optional[int] = None,
        # ---------- V3 核心参数 ----------
        fixed_z: float = 2.0,
        target_distance_max: float = 1.0,
        target_distance_min: float = 0.3,
        reach_threshold: float = 0.10,
        max_episode_seconds: float = 30.0,
        # ---------- 奖励权重 ----------
        reward_reach: float = 100.0,
        reward_speed_bonus: float = 50.0,
        reward_improve_scale: float = 50.0,
        reward_heading_scale: float = 0.0,
        reward_vel_align_scale: float = 0.0,
        reward_time_penalty: float = -0.2,
        reward_heading_terminal: float = 0.0,
        # ---------- 初始速度随机化 ----------
        init_velocity_range: float = 0.0,
        init_yaw_rate_range: float = 0.0,
        # ---------- 推力缩放 ----------
        thrust_scale: float = 40.0,  # action * thrust_scale = 推力(N)
    ):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")

        self.inner = LowLevelAUVEnv(
            config_path=config_path,
            render_mode=render_mode,
            curriculum_stage=curriculum_stage,
            enable_domain_randomization=enable_domain_randomization,
            seed=seed,
        )

        # ---------- 任务参数 ----------
        self.fixed_z = fixed_z
        self.target_distance_max = target_distance_max
        self.target_distance_min = target_distance_min
        self.reach_threshold = reach_threshold
        self.max_episode_seconds = max_episode_seconds

        # ---------- 奖励系数 ----------
        self.reward_reach = reward_reach
        self.reward_speed_bonus = reward_speed_bonus
        self.reward_improve_scale = reward_improve_scale
        self.reward_heading_scale = reward_heading_scale
        self.reward_vel_align_scale = reward_vel_align_scale
        self.reward_time_penalty = reward_time_penalty
        self.reward_heading_terminal = reward_heading_terminal

        # ---------- 初始速度随机化 ----------
        self.init_velocity_range = float(init_velocity_range)
        self.init_yaw_rate_range = float(init_yaw_rate_range) if init_yaw_rate_range > 0 else self.init_velocity_range * 0.5

        # ---------- 推力缩放 ----------
        self.thrust_scale = float(thrust_scale)

        # ---------- 空间定义 ----------
        # 8 维动作：直接对应 8 个推进器
        self.action_space = spaces.Box(
            low=-np.ones(self.NUM_THRUSTERS, dtype=np.float32),
            high=np.ones(self.NUM_THRUSTERS, dtype=np.float32),
            dtype=np.float32,
        )

        # 8 维观测（与 V2/V3 兼容）
        obs_high = np.ones(8, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # ---------- 统计 ----------
        self.episode_count = 0
        self.success_count = 0
        self.prev_pos_error: Optional[float] = None

    # --------------------------------------------------------------------- #
    #                           辅助方法                                     #
    # --------------------------------------------------------------------- #

    def seed(self, seed: Optional[int] = None):
        return self.inner.seed(seed)

    def _enforce_2d_state(self):
        """强制 z / roll / pitch 保持固定。"""
        try:
            eta = self.inner.dynamics.eta
            nu = self.inner.dynamics.nu
            eta[2] = float(self.fixed_z)
            eta[3] = 0.0
            eta[4] = 0.0
            nu[2] = 0.0
            nu[3] = 0.0
            nu[4] = 0.0
        except Exception:
            pass

    def _sample_2d_target(self):
        """在当前位置附近随机采样小目标。"""
        angle = self.inner.np_random.uniform(0, 2 * np.pi)
        distance = self.inner.np_random.uniform(self.target_distance_min, self.target_distance_max)

        init_pos = self.inner.dynamics.eta[:3].copy()
        target_x = init_pos[0] + distance * np.cos(angle)
        target_y = init_pos[1] + distance * np.sin(angle)

        self.inner.target_state = np.array([
            target_x, target_y, self.fixed_z,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])

    def _get_observation(self) -> np.ndarray:
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state

        pos_error_world = target[:2] - eta[:2]
        distance = float(np.linalg.norm(pos_error_world))
        yaw = eta[5]
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        heading_error = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
        pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]

        vx = nu[0]
        vy = nu[1]
        yaw_rate = nu[5]

        obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            vx / 2.0,
            vy / 2.0,
            yaw_rate / 2.0,
            distance / 5.0,
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    # --------------------------------------------------------------------- #
    #                       奖励函数（与 v3_continuous 完全一致）              #
    # --------------------------------------------------------------------- #

    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state

        pos_error = float(np.linalg.norm(eta[:2] - target[:2]))
        pos_error_world = target[:2] - eta[:2]
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))

        info: Dict[str, Any] = {}

        # (a) 稀疏到达奖励 + 快速到达附加
        reached = pos_error < self.reach_threshold
        if reached:
            max_steps = int(self.max_episode_seconds / self.inner.dt)
            time_ratio = self.inner.current_step / max(max_steps, 1)
            raw_reach = self.reward_reach + self.reward_speed_bonus * (1.0 - time_ratio)
            if self.reward_heading_terminal > 0.0:
                alignment = 1.0 - heading_error / np.pi
                floor = 1.0 - self.reward_heading_terminal
                heading_factor = floor + self.reward_heading_terminal * alignment
                reward_reach = raw_reach * heading_factor
            else:
                reward_reach = raw_reach
        else:
            reward_reach = 0.0
        info["reward_reach"] = reward_reach
        info["reached"] = reached

        # (b) 距离改善
        improvement = 0.0
        if self.prev_pos_error is not None:
            improvement = self.prev_pos_error - pos_error
        self.prev_pos_error = pos_error
        reward_improve = self.reward_improve_scale * improvement
        info["reward_improve"] = reward_improve

        # (c) 航向对齐
        alignment = 1.0 - heading_error / np.pi
        speed = float(np.sqrt(nu[0]**2 + nu[1]**2))
        speed_gate = min(1.0, speed / 0.1)
        reward_heading = self.reward_heading_scale * alignment * speed_gate
        info["reward_heading"] = reward_heading

        # (c2) 速度方向对齐
        reward_vel_align = 0.0
        if self.reward_vel_align_scale != 0.0:
            vx_body, vy_body = nu[0], nu[1]
            speed = np.sqrt(vx_body**2 + vy_body**2)
            if speed > 0.02 and pos_error > self.reach_threshold * 2:
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                vx_w = vx_body * cos_yaw - vy_body * sin_yaw
                vy_w = vx_body * sin_yaw + vy_body * cos_yaw
                vel_dir = np.array([vx_w, vy_w]) / speed
                tgt_dir = pos_error_world / max(pos_error, 1e-8)
                cos_align = float(np.dot(vel_dir, tgt_dir))
                dist_scale = min(1.0, pos_error / max(self.reach_threshold * 5, 0.1))
                reward_vel_align = self.reward_vel_align_scale * cos_align * dist_scale
        info["reward_vel_align"] = reward_vel_align

        # (d) 时间惩罚
        reward_time = self.reward_time_penalty
        info["reward_time"] = reward_time

        total_reward = reward_reach + reward_improve + reward_heading + reward_vel_align + reward_time
        info["total"] = total_reward
        info["pos_error"] = pos_error
        info["heading_error"] = heading_error

        return total_reward, info

    # --------------------------------------------------------------------- #
    #                           终止判定                                     #
    # --------------------------------------------------------------------- #

    def _check_termination(self) -> Tuple[bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        eta = self.inner.dynamics.eta
        target = self.inner.target_state

        pos_error = float(np.linalg.norm(eta[:2] - target[:2]))

        if pos_error < self.reach_threshold:
            info["termination_reason"] = "success"
            return True, False, info

        if abs(eta[0]) > 20 or abs(eta[1]) > 20:
            info["termination_reason"] = "out_of_bounds"
            return True, False, info

        max_steps = int(self.max_episode_seconds / self.inner.dt)
        if self.inner.current_step >= max_steps:
            info["termination_reason"] = "timeout"
            return False, True, info

        return False, False, info

    # --------------------------------------------------------------------- #
    #                           Gym 接口                                     #
    # --------------------------------------------------------------------- #

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        _, _ = self.inner.reset(seed=seed, options=options)
        self._enforce_2d_state()
        self._sample_2d_target()
        self._enforce_2d_state()

        if self.init_velocity_range > 0:
            rng = self.inner.np_random
            self.inner.dynamics.eta[5] = rng.uniform(-np.pi, np.pi)
            nu = self.inner.dynamics.nu
            nu[0] = rng.uniform(-self.init_velocity_range, self.init_velocity_range)
            nu[1] = rng.uniform(-self.init_velocity_range, self.init_velocity_range)
            nu[5] = rng.uniform(-self.init_yaw_rate_range, self.init_yaw_rate_range)

        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        self.prev_pos_error = float(np.linalg.norm(eta[:2] - target[:2]))

        obs = self._get_observation()
        info = {
            "simplified_mode": "2d_v3_direct_thrust",
            "target_distance": self.prev_pos_error,
            "target_pos": target[:2].tolist(),
            "init_pos": eta[:2].tolist(),
        }
        self.episode_count += 1
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.NUM_THRUSTERS)
        action = np.clip(action, -1.0, 1.0)

        # ---------- 直接映射到推进器推力 ----------
        # action ∈ [-1, 1]^8 → thrust ∈ [-thrust_scale, +thrust_scale]^8
        # 动力学内部还有饱和/死区/一阶延迟，这里只做缩放
        thrust_command = action.astype(np.float64) * self.thrust_scale
        self.inner.last_thrust_command = thrust_command.copy()

        # 物理步进
        external_force = self.inner._compute_external_force()
        self.inner.dynamics.step(thrust_command, external_force)
        self.inner.water_current.step(self.inner.dt)
        self.inner.current_step += 1

        self._enforce_2d_state()

        # 奖励 & 终止
        reward, reward_info = self._compute_reward()
        terminated, truncated, term_info = self._check_termination()

        obs = self._get_observation()

        if terminated and term_info.get("termination_reason") == "success":
            self.success_count += 1

        info = {
            **reward_info,
            **term_info,
            "action_thrust": action.tolist(),
            "simplified_mode": "2d_v3_direct_thrust",
            "step": self.inner.current_step,
            "success_rate": self.success_count / max(1, self.episode_count),
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None):
        return self.inner.render()

    def close(self):
        return self.inner.close()

    def get_episode_statistics(self) -> Dict[str, Any]:
        stats = self.inner.get_episode_statistics() if hasattr(self.inner, "get_episode_statistics") else {}
        stats["success_rate_2d_v3_direct"] = self.success_count / max(1, self.episode_count)
        stats["episode_count_2d_v3_direct"] = self.episode_count
        return stats


# --------------------------------------------------------------------------- #
#                       向量化封装                                             #
# --------------------------------------------------------------------------- #


class VecLowLevelAUV2DEnvV3DirectThrust:
    """向量化 2D V3 直接推力环境。"""

    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV2DEnvV3DirectThrust(**kwargs) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed: Optional[int] = None):
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            s = seed + i if seed is not None else None
            obs, info = env.reset(seed=s)
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list

    def step(self, actions: np.ndarray):
        obs_list, reward_list, term_list, trunc_list, info_list = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)
        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(term_list),
            np.array(trunc_list),
            info_list,
        )

    def close(self):
        for env in self.envs:
            env.close()
