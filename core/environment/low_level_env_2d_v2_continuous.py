"""
二维环境 V2 连续动作版本（公平对比）
====================================

目标：与 low_level_env_2d_v2 保持同任务、同观测、同奖励、同终止，
仅将动作从离散 6 动作改为连续 3 维动作：
  action = [surge_cmd, sway_cmd, yaw_cmd] ∈ [-1, 1]^3

执行逻辑：
- surge/sway 通过 low_level_config_2d 的 action_distance 映射到每步位移命令
- yaw_cmd 映射到 action_angle_delta.yaw
- 将连续命令经 inner.integrated_controller 直接计算推力，再走 inner.dynamics
- 奖励与终止复用 V2 的 _compute_reward / _check_termination
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env import LowLevelAUVEnv


class LowLevelAUV2DEnvV2Continuous(gym.Env):
    """二维 V2 连续动作环境（公平对照版）。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 curriculum_stage: int = 0,
                 enable_domain_randomization: bool = False,
                 seed: Optional[int] = None,
                 fixed_z: float = 2.0,
                 target_distance: float = 1.0):
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

        self.fixed_z = fixed_z
        self.target_distance = target_distance

        self.action_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )

        obs_high = np.ones(8, dtype=np.float32) * 10.0
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        action_cfg = self.inner.config.get("action_space", {})
        distance_cfg = action_cfg.get("action_distance", {})
        angle_cfg = action_cfg.get("action_angle_delta", {})
        self.surge_step = float(distance_cfg.get("surge", 0.03))
        self.sway_step = float(distance_cfg.get("sway", 0.03))
        self.yaw_delta = float(angle_cfg.get("yaw", 0.0))

        self.episode_count = 0
        self.success_count = 0

    def seed(self, seed: Optional[int] = None):
        return self.inner.seed(seed)

    def _enforce_2d_state(self):
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
        angle = self.inner.np_random.uniform(0, 2 * np.pi)
        distance = self.inner.np_random.uniform(0.5, self.target_distance)

        init_pos = self.inner.dynamics.eta[:3].copy()
        target_x = init_pos[0] + distance * np.cos(angle)
        target_y = init_pos[1] + distance * np.sin(angle)

        self.inner.target_state = np.array([
            target_x, target_y, self.fixed_z,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])

    def _get_observation(self) -> np.ndarray:
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state

        pos_error_world = target[:2] - eta[:2]
        distance = np.linalg.norm(pos_error_world)
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

    def _compute_reward(self) -> Tuple[float, Dict]:
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state

        pos_error = np.linalg.norm(eta[:2] - target[:2])
        pos_error_world = target[:2] - eta[:2]
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))

        reward_info = {}

        position_reward = -0.3 * pos_error
        reward_info['position'] = position_reward

        improvement_reward = 0.0
        if hasattr(self, 'prev_pos_error') and self.prev_pos_error is not None:
            improvement = self.prev_pos_error - pos_error
            improvement_reward = 10.0 * improvement
        self.prev_pos_error = pos_error
        reward_info['improvement'] = improvement_reward

        heading_reward = 0.3 * (1.0 - heading_error / np.pi)
        reward_info['heading'] = heading_reward

        direction_reward = 0.0
        if pos_error > 0.1:
            forward_vel = nu[0]
            if heading_error < 0.5 and forward_vel > 0:
                direction_reward = 0.5 * forward_vel
        reward_info['direction'] = direction_reward

        if pos_error < 0.3:
            proximity_reward = 3.0 * (0.3 - pos_error)
        else:
            proximity_reward = 0.0
        reward_info['proximity'] = proximity_reward

        goal_bonus = 0.0
        if pos_error < 0.2:
            goal_bonus = 100.0
            reward_info['success'] = True
        else:
            reward_info['success'] = False
        reward_info['goal'] = goal_bonus

        time_penalty = -0.02
        reward_info['time'] = time_penalty

        spin_penalty = 0.0
        if abs(nu[5]) > 0.5 and abs(nu[0]) < 0.1:
            spin_penalty = -0.1 * abs(nu[5])
        reward_info['spin_penalty'] = spin_penalty

        total_reward = (
            position_reward + improvement_reward + heading_reward + direction_reward +
            proximity_reward + goal_bonus + time_penalty + spin_penalty
        )
        reward_info['total'] = total_reward
        reward_info['pos_error'] = pos_error
        reward_info['heading_error'] = heading_error
        return total_reward, reward_info

    def _check_termination(self) -> Tuple[bool, bool, Dict]:
        info = {}
        eta = self.inner.dynamics.eta
        target = self.inner.target_state

        pos_error = np.linalg.norm(eta[:2] - target[:2])
        if pos_error < 0.2:
            info['termination_reason'] = 'success'
            return True, False, info

        if abs(eta[0]) > 20 or abs(eta[1]) > 20:
            info['termination_reason'] = 'out_of_bounds'
            return True, False, info

        max_steps = int(90.0 / self.inner.dt)
        if self.inner.current_step >= max_steps:
            info['termination_reason'] = 'timeout'
            return False, True, info

        return False, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        _, _ = self.inner.reset(seed=seed, options=options)
        self._enforce_2d_state()
        self._sample_2d_target()
        self._enforce_2d_state()

        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        self.prev_pos_error = np.linalg.norm(eta[:2] - target[:2])

        obs = self._get_observation()
        info = {
            'simplified_mode': '2d_v2_continuous',
            'target_distance': self.prev_pos_error,
            'target_pos': target[:2].tolist(),
            'init_pos': eta[:2].tolist()
        }
        self.episode_count += 1
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)

        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu

        linear_cmd = np.array([
            float(action[0]) * self.surge_step,
            float(action[1]) * self.sway_step,
            0.0,
        ], dtype=np.float64)
        angle_delta = np.array([float(action[2]) * self.yaw_delta, 0.0, 0.0], dtype=np.float64)

        current_vel_body = nu[:3]
        desired_vel_body = linear_cmd / max(self.inner.dt, 1e-6)

        slowdown_scale = 1.0
        action_cfg = self.inner.config.get('action_space', {})
        slowdown_cfg = action_cfg.get('goal_slowdown', {})
        target_position = self.inner.target_state[:3]
        if slowdown_cfg.get('enabled', True):
            radius = float(slowdown_cfg.get('radius', 1.0))
            min_scale = float(slowdown_cfg.get('min_scale', 0.2))
            pos_error = np.linalg.norm(target_position - eta[:3])
            if radius > 1e-6:
                slowdown_scale = max(min_scale, min(1.0, pos_error / radius))
                desired_vel_body = desired_vel_body * slowdown_scale

        vel_error = desired_vel_body - current_vel_body
        force_cmd = self.inner.integrated_controller.vel_kp * vel_error - self.inner.integrated_controller.vel_kd * current_vel_body
        force_cmd = np.clip(force_cmd, -self.inner.integrated_controller.max_force, self.inner.integrated_controller.max_force)

        roll_current = eta[3]
        pitch_current = eta[4]
        yaw_current = eta[5]
        p = nu[3]
        q = nu[4]
        r = nu[5]

        target_roll = 0.0
        target_pitch = 0.0
        target_yaw = np.arctan2(np.sin(yaw_current + angle_delta[0]), np.cos(yaw_current + angle_delta[0]))

        angle_target = np.array([target_yaw, target_pitch, target_roll], dtype=np.float64)
        angle_current = np.array([yaw_current, pitch_current, roll_current], dtype=np.float64)
        omega_current = np.array([r, q, p], dtype=np.float64)

        tau_attitude, _ = self.inner.integrated_controller.pid_controller.compute(
            angle_target, angle_current, omega_current, self.inner.dt
        )
        thrust_command = self.inner.integrated_controller.thrust_allocator.allocate_with_attitude(force_cmd, tau_attitude)
        self.inner.last_thrust_command = thrust_command.copy()

        external_force = self.inner._compute_external_force()
        eta_new, nu_new = self.inner.dynamics.step(thrust_command, external_force)
        self.inner.water_current.step(self.inner.dt)
        self.inner.current_step += 1

        self._enforce_2d_state()

        reward, reward_info = self._compute_reward()
        terminated, truncated, term_info = self._check_termination()

        obs = self._get_observation()

        if terminated and term_info.get('termination_reason') == 'success':
            self.success_count += 1

        info = {
            **reward_info,
            **term_info,
            'action_continuous': action.tolist(),
            'simplified_mode': '2d_v2_continuous',
            'step': self.inner.current_step,
            'success_rate': self.success_count / max(1, self.episode_count),
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None):
        return self.inner.render()

    def close(self):
        return self.inner.close()

    def get_episode_statistics(self):
        stats = self.inner.get_episode_statistics() if hasattr(self.inner, 'get_episode_statistics') else {}
        stats['success_rate_2d'] = self.success_count / max(1, self.episode_count)
        stats['episode_count_2d'] = self.episode_count
        return stats


class VecLowLevelAUV2DEnvV2Continuous:
    """向量化 2D V2 连续动作环境。"""

    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV2DEnvV2Continuous(**kwargs) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            s = seed + i if seed is not None else None
            obs, info = env.reset(seed=s)
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list

    def step(self, actions):
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
