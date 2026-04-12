"""
阶段一：2D 追击者低层环境（完整动力学，与之前动作空间一致）
============================================================

参考之前 2D 最好的模型，使用 pe_2d_dynamics 的完整动力学（不简化）。
动作空间与 env_asymmetric_pe_v2_dynamics 一致：9 动作 = 0(无推力) + 8 方向满推力。
观测含位置、速度、加速度等，便于利用动力学做规划。
用于分层训练第一阶段：先训练一个会追移动目标的追击者。
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .pe_2d_dynamics import (
    Dynamics2D,
    create_pursuer_dynamics,
    _action_to_thrust_direction,
)

# 与之前追逃环境一致：9 动作 = 0(无推力) + 8 方向满推力
NUM_PURSUER_ACCEL_ACTIONS = 9
PURSUER_V_MAX = 1.2
PURSUER_A_MAX = 2.5
WORLD_SIZE = 10.0
CATCH_RADIUS = 0.5
DT = 0.05
MAX_STEPS = 500


class PursuerLowLevelDynamicsEnv(gym.Env):
    """
    单智能体：追击者用完整动力学追移动目标。
    动力学 M*a = τ - D_l*v - D_q*|v|*v，不简化。
    观测含位置、速度、加速度、相对目标信息等，便于利用加速度做规划。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        pursuer_v_max: float = PURSUER_V_MAX,
        pursuer_a_max: float = PURSUER_A_MAX,
        world_size: float = WORLD_SIZE,
        dt: float = DT,
        max_steps: int = MAX_STEPS,
        catch_radius: float = CATCH_RADIUS,
        target_motion: str = "constant",
        target_speed: float = 0.5,
        init_distance_range: Tuple[float, float] = (2.0, 6.0),
        reward_catch: float = 100.0,
        reward_distance_improve: float = 5.0,
        penalty_time: float = 0.05,
        penalty_boundary: float = 0.1,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.pursuer_v_max = pursuer_v_max
        self.pursuer_a_max = pursuer_a_max
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.catch_radius = catch_radius
        self.target_motion = target_motion
        self.target_speed = target_speed
        self.init_distance_range = init_distance_range
        self.reward_catch = reward_catch
        self.reward_distance_improve = reward_distance_improve
        self.penalty_time = penalty_time
        self.penalty_boundary = penalty_boundary

        self._dynamics = create_pursuer_dynamics(
            v_max=pursuer_v_max, a_max=pursuer_a_max, dt=dt
        )
        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._acc = np.zeros(2, dtype=np.float32)
        self._target_pos = np.zeros(2, dtype=np.float32)
        self._target_vel = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_distance = 0.0
        self.np_random = np.random.default_rng(seed)
        self.render_mode = render_mode

        # 观测：pos(2), vel(2), acc(2), rel_pos(2), rel_vel(2), dist(1), to_boundary(4), my_speed(1), tgt_speed(1), angle_align(1) = 18
        obs_dim = 18
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim) * 10.0,
            high=np.ones(obs_dim) * 10.0,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_PURSUER_ACCEL_ACTIONS)

    def _update_target(self):
        """更新目标位置（恒定速度或简单逃逸）"""
        if self.target_motion == "constant":
            self._target_pos += self._target_vel * self.dt
        elif self.target_motion == "evasive":
            diff = self._target_pos - self._pos
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                escape_dir = diff / dist
                self._target_vel = self.target_speed * min(1.0, 2.0 / (dist + 0.5)) * escape_dir
            self._target_pos += self._target_vel * self.dt

        for i in range(2):
            self._target_pos[i] = np.clip(
                self._target_pos[i], -self.world_size, self.world_size
            )
            if self._target_pos[i] >= self.world_size:
                self._target_vel[i] = -abs(self._target_vel[i])
            elif self._target_pos[i] <= -self.world_size:
                self._target_vel[i] = abs(self._target_vel[i])

    def _step_pursuer(self, action: int) -> bool:
        dir_vec = _action_to_thrust_direction(action)
        thrust = dir_vec * self._dynamics.max_thrust
        pos_new, vel_new, acc = self._dynamics.step(
            self._pos.copy(), self._vel.copy(), thrust, v_max=self.pursuer_v_max
        )
        hit_boundary = False
        for i in range(2):
            if pos_new[i] < -self.world_size:
                pos_new[i] = -self.world_size
                vel_new[i] = abs(vel_new[i]) * 0.3
                hit_boundary = True
            elif pos_new[i] > self.world_size:
                pos_new[i] = self.world_size
                vel_new[i] = -abs(vel_new[i]) * 0.3
                hit_boundary = True
        self._pos = pos_new
        self._vel = vel_new
        self._acc = acc
        return hit_boundary

    def _get_obs(self) -> np.ndarray:
        pos_scale = 1.0 / self.world_size
        vel_scale = 1.0 / max(self.pursuer_v_max, self.target_speed * 2)
        acc_scale = 1.0 / self.pursuer_a_max

        rel_pos = self._target_pos - self._pos
        rel_vel = self._target_vel - self._vel
        distance = np.linalg.norm(rel_pos)
        my_speed = np.linalg.norm(self._vel)
        tgt_speed = np.linalg.norm(self._target_vel)

        angle_align = 0.0
        if my_speed > 1e-6 and distance > 1e-6:
            vel_dir = self._vel / my_speed
            target_dir = rel_pos / distance
            angle_align = np.dot(vel_dir, target_dir)

        to_right = self.world_size - self._pos[0]
        to_left = self._pos[0] + self.world_size
        to_top = self.world_size - self._pos[1]
        to_bottom = self._pos[1] + self.world_size

        obs = np.array(
            [
                self._pos[0] * pos_scale,
                self._pos[1] * pos_scale,
                self._vel[0] * vel_scale,
                self._vel[1] * vel_scale,
                self._acc[0] * acc_scale,
                self._acc[1] * acc_scale,
                rel_pos[0] * pos_scale,
                rel_pos[1] * pos_scale,
                rel_vel[0] * vel_scale,
                rel_vel[1] * vel_scale,
                distance * pos_scale,
                to_right * pos_scale,
                to_left * pos_scale,
                to_top * pos_scale,
                to_bottom * pos_scale,
                my_speed * vel_scale,
                tgt_speed * vel_scale,
                angle_align,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -10.0, 10.0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        super().reset(seed=seed)

        self._pos = self.np_random.uniform(
            -self.world_size * 0.7, self.world_size * 0.7, size=2
        ).astype(np.float32)
        min_d, max_d = self.init_distance_range
        for _ in range(100):
            self._target_pos = self.np_random.uniform(
                -self.world_size * 0.7, self.world_size * 0.7, size=2
            ).astype(np.float32)
            dist = np.linalg.norm(self._target_pos - self._pos)
            if min_d <= dist <= max_d:
                break
        angle = self.np_random.uniform(0, 2 * np.pi)
        self._target_vel = self.target_speed * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        self._vel = np.zeros(2, dtype=np.float32)
        self._acc = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_distance = np.linalg.norm(self._target_pos - self._pos)

        obs = self._get_obs()
        info = {
            "distance": float(self._prev_distance),
            "pursuer_pos": self._pos.copy(),
            "target_pos": self._target_pos.copy(),
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        hit_boundary = self._step_pursuer(int(action))
        self._update_target()

        distance = np.linalg.norm(self._target_pos - self._pos)
        reward = self.reward_distance_improve * (self._prev_distance - distance)
        reward += -self.penalty_time
        if hit_boundary:
            reward += -self.penalty_boundary
        if distance < 2.0:
            reward += 0.2 * (2.0 - distance)
        if distance < self.catch_radius:
            reward += self.reward_catch
        self._prev_distance = distance

        terminated = distance < self.catch_radius
        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()
        info = {
            "distance": float(distance),
            "pursuer_pos": self._pos.copy(),
            "target_pos": self._target_pos.copy(),
            "success": terminated,
        }
        if terminated or truncated:
            info["episode_length"] = self._step_count
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass


def make_pursuer_env(
    target_motion: str = "constant",
    target_speed: float = 0.5,
    init_distance_range: Tuple[float, float] = (2.0, 6.0),
    **kwargs
) -> PursuerLowLevelDynamicsEnv:
    return PursuerLowLevelDynamicsEnv(
        target_motion=target_motion,
        target_speed=target_speed,
        init_distance_range=init_distance_range,
        **kwargs
    )
