"""
3D 环境 V2 消融实验版 - 不修改原 low_level_env_3d_v2.py
=========================================================

新增功能（仅用于消融实验）：
- success_threshold: 可配置到达精度阈值（默认 0.5，可设为 0.3 提高精度）

使用方式：
    from core.environment.low_level_env_3d_v2_ablation import (
        LowLevelAUV3DEnvV2Ablation,
        VecLowLevelAUV3DEnvV2Ablation,
    )
"""
from typing import Tuple, Dict

import numpy as np

from .low_level_env_3d_v2 import LowLevelAUV3DEnvV2


class LowLevelAUV3DEnvV2Ablation(LowLevelAUV3DEnvV2):
    """
    3D V2 消融版：支持可配置 success_threshold（到达精度）
    其余行为与原 LowLevelAUV3DEnvV2 完全一致。
    """

    def __init__(self, success_threshold: float = 0.5, **kwargs):
        """
        Args:
            success_threshold: 到达成功判定距离（m），默认 0.5；设为 0.3 可提高精度
            其余参数同 LowLevelAUV3DEnvV2
        """
        super().__init__(**kwargs)
        self.success_threshold = success_threshold

    def _compute_reward(self, action: int) -> Tuple[float, Dict]:
        """重写：使用 self.success_threshold 替代硬编码 0.5"""
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state

        pos_error = np.linalg.norm(eta[:3] - target[:3])

        pos_error_world = target[:2] - eta[:2]
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))

        reward_info = {}

        position_reward = -0.3 * pos_error
        reward_info['position'] = position_reward

        improvement_reward = 0.0
        if self.prev_pos_error is not None:
            improvement = self.prev_pos_error - pos_error
            improvement_reward = 10.0 * improvement
        self.prev_pos_error = pos_error
        reward_info['improvement'] = improvement_reward

        heading_reward = self.heading_coef * (1.0 - heading_error / np.pi)
        if heading_error < np.pi / 6:
            heading_reward += 0.15
        elif heading_error < np.pi / 3:
            heading_reward += 0.05
        reward_info['heading'] = heading_reward

        xy_error = np.linalg.norm(eta[:2] - target[:2])
        direction_reward = 0.0
        if pos_error > 0.1:
            forward_vel = nu[0]
            if self.strict_direction_heading:
                if self.relaxed_heading_when_xy_close and xy_error < 0.5:
                    align_ok = heading_error < np.pi / 3  # 60° when XY close
                else:
                    align_ok = heading_error < np.pi / 6
            else:
                align_ok = heading_error < 0.5
            if align_ok and forward_vel > 0:
                direction_reward = self.direction_coef * forward_vel
        reward_info['direction'] = direction_reward

        wrong_way_penalty = 0.0
        if self.heading_penalty_wrong > 0 and pos_error > 0.1 and nu[0] > 0 and heading_error > np.pi / 2:
            wrong_way_penalty = -self.heading_penalty_wrong
        reward_info['wrong_way_penalty'] = wrong_way_penalty

        backward_penalty = 0.0
        if self.backward_facing_target_penalty > 0 and pos_error > 0.5:
            if action == 1 and heading_error < np.pi / 2:
                backward_penalty = -self.backward_facing_target_penalty
                reward_info['backward_type'] = 'facing_target_but_backward'
        reward_info['backward_penalty'] = backward_penalty

        z_error = target[2] - eta[2]
        z_vel = nu[2]
        z_direction_reward = 0.0
        if abs(z_error) > 0.1:
            if z_error > 0 and z_vel > 0:
                z_direction_reward = self.z_coef * z_vel
            elif z_error < 0 and z_vel < 0:
                z_direction_reward = self.z_coef * abs(z_vel)
        if xy_error < self.xy_close_threshold and z_direction_reward > 0:
            z_direction_reward *= self.z_boost_when_xy_close
        reward_info['z_direction'] = z_direction_reward

        th = self.success_threshold
        if pos_error < th:
            proximity_reward = 3.0 * (th - pos_error)
        else:
            proximity_reward = 0.0
        reward_info['proximity'] = proximity_reward

        goal_bonus = 0.0
        if pos_error < th:
            goal_bonus = 100.0
            reward_info['success'] = True
        else:
            reward_info['success'] = False
        reward_info['goal'] = goal_bonus

        time_penalty = -0.02
        reward_info['time'] = time_penalty

        total_reward = (position_reward + improvement_reward + heading_reward +
                       direction_reward + wrong_way_penalty + backward_penalty +
                       z_direction_reward + proximity_reward + goal_bonus + time_penalty)
        reward_info['total'] = total_reward
        reward_info['pos_error'] = pos_error
        reward_info['heading_error'] = heading_error

        return total_reward, reward_info

    def _check_termination(self) -> Tuple[bool, bool, Dict]:
        """重写：使用 self.success_threshold"""
        info = {}
        eta = self.inner.dynamics.eta
        target = self.inner.target_state

        pos_error = np.linalg.norm(eta[:3] - target[:3])

        if pos_error < self.success_threshold:
            info['termination_reason'] = 'success'
            self.success_count += 1
            return True, False, info

        if abs(eta[0]) > 25 or abs(eta[1]) > 25 or eta[2] < 0 or eta[2] > 10:
            info['termination_reason'] = 'out_of_bounds'
            return True, False, info

        if self.step_count >= self.max_steps:
            info['termination_reason'] = 'timeout'
            return False, True, info

        info['termination_reason'] = None
        info['pos_error'] = pos_error
        return False, False, info


class VecLowLevelAUV3DEnvV2Ablation:
    """向量化 3D 环境 V2 消融版"""

    def __init__(self, num_envs: int = 4, success_threshold: float = 0.5, **kwargs):
        self.num_envs = num_envs
        self.success_threshold = success_threshold
        self.envs = [
            LowLevelAUV3DEnvV2Ablation(success_threshold=success_threshold, **kwargs)
            for _ in range(num_envs)
        ]

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list

    def step(self, actions):
        obs_list, reward_list, terminated_list, truncated_list, info_list = [], [], [], [], []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, reset_info = env.reset()
                info['reset_info'] = reset_info

            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)

        return (np.array(obs_list), np.array(reward_list),
                np.array(terminated_list), np.array(truncated_list), info_list)
