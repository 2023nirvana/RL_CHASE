"""
二维追逃环境 - 复用完整 LowLevelAUVEnv（与「到达固定目标」同套动力学+PID）
============================================================================

与 LowLevelAUV2DEnvV2 唯一本质区别：目标不是固定点，而是每步更新的逃脱者位置。
- 追击者：inner = LowLevelAUVEnv（完整 AUV 动力学 + IntegratedController + low_level_config_2d）
- 逃脱者：2D 质点，仅设计约束为 v_max > 追击者、a_max < 追击者（来自 robot_config pursuer/evader）
- 每步：(1) 按逃脱者策略更新逃脱者位置/速度（满足 v_max/a_max 上限），(2) inner.target_state[:3] = 逃脱者位置，(3) inner.step(action)
- 观测/动作与 LowLevelAUV2DEnvV2 一致（8 维观测，6 动作），训练脚本可与 train_low_level_2d_v2 同构、同量级（2M 步等），训练时长与到达固定目标相当。
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env import LowLevelAUVEnv


def _clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return x
    n = np.linalg.norm(x)
    if n <= max_norm:
        return x
    return x * (max_norm / n)


class LowLevelAUV2DPursuitEvasionEnv(gym.Env):
    """
    二维追逃：内层与「到达固定目标」完全相同的 LowLevelAUVEnv，目标=逃脱者当前位置。
    设计约束（仅限幅）：追击者 v_max < 逃脱者 v_max，追击者 a_max > 逃脱者 a_max（由 config 给出）。
    """

    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right"]
    EVADER_STATE_PATROL = "PATROL"
    EVADER_STATE_ALERT = "ALERT"
    EVADER_STATE_FLEEING = "FLEEING"
    EVADER_STATE_LOST = "LOST"

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 curriculum_stage: int = 0,
                 enable_domain_randomization: bool = False,
                 seed: Optional[int] = None,
                 fixed_z: float = 2.0,
                 init_distance_range: Tuple[float, float] = (2.0, 6.0),
                 catch_radius: float = 0.5,
                 world_size: float = 10.0,
                 max_episode_seconds: float = 90.0):
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

        robot_config_path = os.path.join(PROJECT_ROOT, "configs/robot_config.yaml")
        with open(robot_config_path, "r", encoding="utf-8") as f:
            robot_config = yaml.safe_load(f)
        pursuer_cfg = robot_config.get("pursuer", {})
        evader_cfg = robot_config.get("evader", {})
        vel_limits = robot_config.get("velocity_limits", {})
        self.pursuer_v_max = float(pursuer_cfg.get("max_speed", vel_limits.get("max_surge", 1.5)))
        self.pursuer_a_max = float(pursuer_cfg.get("max_acceleration", 2.0))
        self.evader_v_max = float(evader_cfg.get("max_speed", 2.5))
        self.evader_a_max = float(evader_cfg.get("max_acceleration", 1.0))
        self.evader_patrol_speed_ratio = float(evader_cfg.get("patrol_speed_ratio", 0.5))
        self.evader_detection_radius = float(evader_cfg.get("detection_radius", 4.0))
        fov_deg = evader_cfg.get("fov_deg", evader_cfg.get("fov_angle", 120.0))
        self.evader_fov_rad = float(fov_deg) * np.pi / 180.0
        self.evader_reaction_delay = float(evader_cfg.get("reaction_delay", 0.5))
        self.evader_lost_memory = float(evader_cfg.get("lost_memory", 2.0))
        self.evader_heading_noise_std = float(evader_cfg.get("heading_noise_std", 0.15))
        self._base_init_distance_range = init_distance_range
        self._base_detection_radius = self.evader_detection_radius
        self._base_reaction_delay = self.evader_reaction_delay

        self.fixed_z = fixed_z
        self.init_distance_range = init_distance_range
        self.catch_radius = catch_radius
        self.world_size = world_size
        self.max_episode_seconds = max_episode_seconds
        self.dt = self.inner.dt
        self.max_steps = int(max_episode_seconds / self.dt)
        self.curriculum_stage = 0
        self.evader_reaction_delay_steps = 0
        self.evader_lost_memory_steps = 0
        self.set_curriculum_stage(curriculum_stage)

        self.action_map = {
            0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8,
        }
        self.action_space = spaces.Discrete(6)
        self.obs_dim = 12
        obs_high = np.ones(self.obs_dim) * 10.0
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.evader_pos = np.zeros(2, dtype=np.float64)
        self.evader_vel = np.zeros(2, dtype=np.float64)
        self.evader_heading = 0.0
        self.evader_state = self.EVADER_STATE_PATROL
        self.evader_state_timer = 0
        self.evader_last_seen_pos = None
        self.evader_visible = False
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

    def _sample_evader_and_target(self):
        """采样逃脱者初始位置（在 init_distance_range 内），并设为 inner 的目标。"""
        eta = self.inner.dynamics.eta
        pursuer_xy = eta[:2]
        angle = self.inner.np_random.uniform(0, 2 * np.pi)
        dist = self.inner.np_random.uniform(self.init_distance_range[0], self.init_distance_range[1])
        self.evader_pos = pursuer_xy + dist * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        self.evader_pos = np.clip(self.evader_pos, -self.world_size, self.world_size)
        self.evader_vel = np.zeros(2, dtype=np.float64)
        self.evader_heading = self.inner.np_random.uniform(-np.pi, np.pi)
        self.evader_state = self.EVADER_STATE_PATROL
        self.evader_state_timer = 0
        self.evader_last_seen_pos = None
        self.evader_visible = False
        self.inner.target_state[:3] = np.array([self.evader_pos[0], self.evader_pos[1], self.fixed_z])

    def set_curriculum_stage(self, stage: int):
        stage = int(np.clip(stage, 0, 2))
        self.curriculum_stage = stage
        if stage == 0:
            scale = 0.6
            delay_scale = 1.4
        elif stage == 1:
            scale = 0.8
            delay_scale = 1.15
        else:
            scale = 1.0
            delay_scale = 1.0
        base_min, base_max = self._base_init_distance_range
        self.init_distance_range = (base_min * scale, base_max * scale)
        self.evader_detection_radius = self._base_detection_radius * scale
        self.evader_reaction_delay = self._base_reaction_delay * delay_scale
        self.evader_reaction_delay_steps = int(
            np.ceil(self.evader_reaction_delay / self.dt)
        ) if self.evader_reaction_delay > 0 else 0
        self.evader_lost_memory_steps = int(
            np.ceil(self.evader_lost_memory / self.dt)
        ) if self.evader_lost_memory > 0 else 0

    def _evader_can_see_pursuer(self, pursuer_xy: np.ndarray) -> Tuple[bool, float, float]:
        rel = pursuer_xy - self.evader_pos
        dist = np.linalg.norm(rel)
        if dist > self.evader_detection_radius:
            return False, dist, 0.0
        if np.linalg.norm(self.evader_vel) > 0.1:
            self.evader_heading = np.arctan2(self.evader_vel[1], self.evader_vel[0])
        rel_angle = np.arctan2(rel[1], rel[0])
        angle_diff = np.arctan2(
            np.sin(rel_angle - self.evader_heading),
            np.cos(rel_angle - self.evader_heading),
        )
        in_fov = abs(angle_diff) <= 0.5 * self.evader_fov_rad
        return in_fov, dist, angle_diff

    def _step_evader(self):
        """逃脱者一步：有限视野 + 反应延迟 + 状态机。"""
        eta = self.inner.dynamics.eta
        pursuer_xy = eta[:2].astype(np.float64)
        visible, _, _ = self._evader_can_see_pursuer(pursuer_xy)
        self.evader_visible = visible
        if visible:
            self.evader_last_seen_pos = pursuer_xy.copy()
            if self.evader_state in (self.EVADER_STATE_PATROL, self.EVADER_STATE_LOST):
                self.evader_state = self.EVADER_STATE_ALERT
                self.evader_state_timer = 0

        if self.evader_state == self.EVADER_STATE_ALERT:
            if not visible:
                self.evader_state = self.EVADER_STATE_PATROL
            else:
                self.evader_state_timer += 1
                if self.evader_state_timer >= self.evader_reaction_delay_steps:
                    self.evader_state = self.EVADER_STATE_FLEEING
                    self.evader_state_timer = 0

        if self.evader_state == self.EVADER_STATE_FLEEING and not visible:
            self.evader_state = self.EVADER_STATE_LOST
            self.evader_state_timer = 0

        if self.evader_state == self.EVADER_STATE_LOST and visible:
            self.evader_state = self.EVADER_STATE_FLEEING
            self.evader_state_timer = 0

        if self.evader_state == self.EVADER_STATE_PATROL:
            if self.evader_heading_noise_std > 0:
                self.evader_heading += self.inner.np_random.normal(0.0, self.evader_heading_noise_std)
            desired_speed = self.evader_v_max * self.evader_patrol_speed_ratio
            desired_vel = desired_speed * np.array([
                np.cos(self.evader_heading),
                np.sin(self.evader_heading),
            ], dtype=np.float64)
        elif self.evader_state == self.EVADER_STATE_ALERT:
            desired_speed = self.evader_v_max * self.evader_patrol_speed_ratio
            desired_vel = desired_speed * np.array([
                np.cos(self.evader_heading),
                np.sin(self.evader_heading),
            ], dtype=np.float64)
        elif self.evader_state == self.EVADER_STATE_FLEEING:
            diff = self.evader_pos - pursuer_xy
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                escape_dir = self.inner.np_random.uniform(-1, 1, 2)
                escape_dir = escape_dir / (np.linalg.norm(escape_dir) + 1e-8)
            else:
                escape_dir = diff / dist
            self.evader_heading = np.arctan2(escape_dir[1], escape_dir[0])
            desired_vel = escape_dir * self.evader_v_max
        else:
            self.evader_state_timer += 1
            if self.evader_state_timer >= self.evader_lost_memory_steps:
                self.evader_state = self.EVADER_STATE_PATROL
                self.evader_state_timer = 0
            if self.evader_last_seen_pos is not None:
                diff = self.evader_pos - self.evader_last_seen_pos
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    escape_dir = np.array([
                        np.cos(self.evader_heading),
                        np.sin(self.evader_heading),
                    ], dtype=np.float64)
                else:
                    escape_dir = diff / dist
            else:
                escape_dir = np.array([
                    np.cos(self.evader_heading),
                    np.sin(self.evader_heading),
                ], dtype=np.float64)
            desired_vel = escape_dir * self.evader_v_max

        acc = (desired_vel - self.evader_vel) / self.dt
        acc = _clip_norm(acc, self.evader_a_max)
        self.evader_vel = self.evader_vel + acc * self.dt
        self.evader_vel = _clip_norm(self.evader_vel, self.evader_v_max)
        self.evader_pos = self.evader_pos + self.evader_vel * self.dt
        hit_boundary = False
        for i in range(2):
            if self.evader_pos[i] < -self.world_size:
                self.evader_pos[i] = -self.world_size
                self.evader_vel[i] = abs(self.evader_vel[i]) * 0.3
                hit_boundary = True
            elif self.evader_pos[i] > self.world_size:
                self.evader_pos[i] = self.world_size
                self.evader_vel[i] = -abs(self.evader_vel[i]) * 0.3
                hit_boundary = True
        if hit_boundary:
            to_center = -self.evader_pos
            if np.linalg.norm(to_center) > 1e-6:
                self.evader_heading = np.arctan2(to_center[1], to_center[0])
        self.inner.target_state[:3] = np.array([self.evader_pos[0], self.evader_pos[1], self.fixed_z])

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
        pursuer_vel = nu[:2]
        evader_speed = np.linalg.norm(self.evader_vel)
        evader_heading = self.evader_heading
        if evader_speed > 0.1:
            evader_heading = np.arctan2(self.evader_vel[1], self.evader_vel[0])
        los_norm = np.linalg.norm(pos_error_world)
        if los_norm < 1e-6 or np.linalg.norm(pursuer_vel) < 1e-6:
            approach_angle = np.pi / 2
        else:
            pv_dir = pursuer_vel / (np.linalg.norm(pursuer_vel) + 1e-8)
            los_dir = pos_error_world / (los_norm + 1e-8)
            dot = float(np.clip(np.dot(pv_dir, los_dir), -1.0, 1.0))
            approach_angle = np.arccos(dot)
        obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            nu[0] / 2.0,
            nu[1] / 2.0,
            nu[5] / 2.0,
            distance / 5.0,
            evader_speed / max(1e-6, self.evader_v_max),
            evader_heading / np.pi,
            approach_angle / np.pi,
            1.0 if self.evader_visible else 0.0,
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
        position_reward = -0.25 * pos_error
        reward_info["position"] = position_reward
        improvement_reward = 0.0
        if hasattr(self, "prev_pos_error") and self.prev_pos_error is not None:
            improvement_reward = 8.0 * (self.prev_pos_error - pos_error)
        self.prev_pos_error = pos_error
        reward_info["improvement"] = improvement_reward
        heading_reward = 0.2 * (1.0 - heading_error / np.pi)
        reward_info["heading"] = heading_reward
        approach_angle = np.pi / 2
        if np.linalg.norm(nu[:2]) > 1e-6 and np.linalg.norm(pos_error_world) > 1e-6:
            pv_dir = nu[:2] / (np.linalg.norm(nu[:2]) + 1e-8)
            los_dir = pos_error_world / (np.linalg.norm(pos_error_world) + 1e-8)
            dot = float(np.clip(np.dot(pv_dir, los_dir), -1.0, 1.0))
            approach_angle = np.arccos(dot)
        approach_reward = 0.35 * (1.0 - approach_angle / np.pi)
        reward_info["approach"] = approach_reward
        stealth_reward = 0.0
        if not self.evader_visible and pos_error < self.evader_detection_radius:
            stealth_reward = 0.4
        reward_info["stealth"] = stealth_reward
        proximity_reward = 3.0 * (0.3 - pos_error) if pos_error < 0.3 else 0.0
        reward_info["proximity"] = proximity_reward
        goal_bonus = 100.0 if pos_error < self.catch_radius else 0.0
        reward_info["success"] = pos_error < self.catch_radius
        reward_info["goal"] = goal_bonus
        reward_info["time"] = -0.02
        spin_penalty = -0.1 * abs(nu[5]) if abs(nu[5]) > 0.5 and abs(nu[0]) < 0.1 else 0.0
        reward_info["spin_penalty"] = spin_penalty
        total_reward = (position_reward + improvement_reward + heading_reward +
                        approach_reward + stealth_reward + proximity_reward +
                        goal_bonus - 0.02 + spin_penalty)
        reward_info["total"] = total_reward
        reward_info["pos_error"] = pos_error
        reward_info["heading_error"] = heading_error
        reward_info["evader_visible"] = self.evader_visible
        reward_info["evader_state"] = self.evader_state
        return total_reward, reward_info

    def _check_termination(self) -> Tuple[bool, bool, Dict]:
        info = {}
        eta = self.inner.dynamics.eta
        pos_error = np.linalg.norm(eta[:2] - self.evader_pos)
        if pos_error < self.catch_radius:
            info["termination_reason"] = "success"
            return True, False, info
        if abs(eta[0]) > self.world_size or abs(eta[1]) > self.world_size:
            info["termination_reason"] = "out_of_bounds"
            return True, False, info
        if self.inner.current_step >= self.max_steps:
            info["termination_reason"] = "timeout"
            return False, True, info
        return False, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.inner.reset(seed=seed, options=options)
        self._enforce_2d_state()
        self._sample_evader_and_target()
        self._enforce_2d_state()
        eta = self.inner.dynamics.eta
        self.prev_pos_error = np.linalg.norm(eta[:2] - self.evader_pos)
        obs = self._get_observation()
        info = {
            "simplified_mode": "2d_pursuit_evasion",
            "target_pos": self.evader_pos.copy(),
            "init_pos": eta[:2].tolist(),
            "distance": float(self.prev_pos_error),
            "evader_state": self.evader_state,
            "evader_visible": self.evader_visible,
        }
        self.episode_count += 1
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = int(action)
        mapped_action = self.action_map.get(action, 1)
        self._step_evader()
        _, _, _, _, _ = self.inner.step(mapped_action)
        self._enforce_2d_state()
        reward, reward_info = self._compute_reward()
        terminated, truncated, term_info = self._check_termination()
        obs = self._get_observation()
        if terminated and term_info.get("termination_reason") == "success":
            self.success_count += 1
        info = {
            **reward_info,
            **term_info,
            "action_2d": action,
            "action_name": self.ACTION_NAMES[action],
            "action_mapped": mapped_action,
            "simplified_mode": "2d_pursuit_evasion",
            "step": self.inner.current_step,
            "success_rate": self.success_count / max(1, self.episode_count),
            "distance": float(np.linalg.norm(self.inner.dynamics.eta[:2] - self.evader_pos)),
            "evader_state": self.evader_state,
            "evader_visible": self.evader_visible,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None):
        return self.inner.render()

    def close(self):
        return self.inner.close()


class VecLowLevelAUV2DPursuitEvasionEnv:
    """向量化 2D 追逃环境，与 VecLowLevelAUV2DEnvV2 用法一致。"""

    def __init__(self, num_envs: int = 4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LowLevelAUV2DPursuitEvasionEnv(**kwargs) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            s = seed + i if seed is not None else None
            obs, info = env.reset(seed=s)
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list

    def step(self, actions):
        obs_list, reward_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)
        return np.array(obs_list), np.array(reward_list), np.array(term_list), np.array(trunc_list), info_list

    def set_curriculum_stage(self, stage: int):
        for env in self.envs:
            env.set_curriculum_stage(stage)

    def close(self):
        for env in self.envs:
            env.close()
