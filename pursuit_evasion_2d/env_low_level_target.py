"""
低层目标跟踪环境（含加速度观测）
=================================

用于训练“底层策略”：给定目标位置，用受限的 v_max / a_max 尽快到达。
观测包含 pos, vel, acc，使策略能利用机动性（加速度）做预测与规划。
与主追逃环境共用相同的双积分器与动作空间，便于 HRL 时直接复用。
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env_asymmetric_pe import (
    _clip_norm,
    _action_to_acceleration,
    NUM_ACCEL_ACTIONS,
)


class LowLevelTarget2DEnv(gym.Env):
    """
    二维目标到达任务：双积分器 + v_max, a_max。
    观测：[pos_error_x, pos_error_y, vel_x, vel_y, acc_x, acc_y] 归一化，
    强调对加速度的关注与观测。
    动作：Discrete(9)，与追逃环境一致。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        v_max: float = 1.0,
        a_max: float = 2.0,
        world_size: float = 10.0,
        dt: float = 0.05,
        max_steps: int = 400,
        goal_tolerance: float = 0.2,
        target_distance_range: Tuple[float, float] = (1.0, 5.0),
        obs_scale_pos: float = 1.0 / 10.0,
        obs_scale_vel: float = 1.0 / 2.0,
        obs_scale_acc: float = 1.0 / 3.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.v_max = v_max
        self.a_max = a_max
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance
        self.target_distance_range = target_distance_range
        self.obs_scale_pos = obs_scale_pos
        self.obs_scale_vel = obs_scale_vel
        self.obs_scale_acc = obs_scale_acc

        self._pos = np.zeros(2)
        self._vel = np.zeros(2)
        self._acc = np.zeros(2)
        self._goal = np.zeros(2)
        self._step_count = 0
        self.np_random = np.random.default_rng(seed)

        # 观测：[pos_error(2), vel(2), acc(2)] = 6 维，含加速度
        obs_dim = 6
        obs_high = np.ones(obs_dim) * 10.0
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACCEL_ACTIONS)

    def _step_dynamics(
        self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        acc_clip = _clip_norm(acc, self.a_max)
        vel_new = vel + acc_clip * self.dt
        vel_new = _clip_norm(vel_new, self.v_max)
        pos_new = pos + vel_new * self.dt
        for i in range(2):
            pos_new[i] = np.clip(pos_new[i], -self.world_size, self.world_size)
        return pos_new, vel_new

    def _get_obs(self) -> np.ndarray:
        pos_error = self._goal - self._pos
        obs = np.concatenate(
            [
                pos_error * self.obs_scale_pos,
                self._vel * self.obs_scale_vel,
                self._acc * self.obs_scale_acc,
            ]
        ).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        super().reset(seed=seed)

        self._pos = self.np_random.uniform(
            -self.world_size * 0.5, self.world_size * 0.5, size=2
        )
        self._vel = np.zeros(2)
        self._acc = np.zeros(2)
        dist = self.np_random.uniform(
            self.target_distance_range[0], self.target_distance_range[1]
        )
        angle = self.np_random.uniform(0, 2 * np.pi)
        self._goal = self._pos + dist * np.array([np.cos(angle), np.sin(angle)])
        self._goal = np.clip(self._goal, -self.world_size, self.world_size)
        self._step_count = 0

        info = {
            "goal": self._goal.copy(),
            "distance": np.linalg.norm(self._goal - self._pos),
        }
        return self._get_obs(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        acc = _action_to_acceleration(int(action), self.a_max)
        self._acc = acc
        self._pos, self._vel = self._step_dynamics(self._pos, self._vel, acc)
        self._step_count += 1

        dist = np.linalg.norm(self._goal - self._pos)
        reward = -dist * 0.2
        if dist < self.goal_tolerance:
            reward = 50.0
        reward -= 0.01  # 时间惩罚

        terminated = dist < self.goal_tolerance
        truncated = self._step_count >= self.max_steps

        info = {
            "distance": float(dist),
            "success": terminated,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
