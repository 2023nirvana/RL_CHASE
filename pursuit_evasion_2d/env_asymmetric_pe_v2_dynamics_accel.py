"""
阶段二：二维非对称追逃环境 - 与之前动作空间一致（动力学版）
============================================================

与 env_asymmetric_pe_v2_dynamics 相同的动作空间与物理：追击者 9 动作
（0 无推力 + 8 方向满推力），速度劣势、加速度优势。不修改原 env_asymmetric_pe_v2_dynamics，
本文件仅用于分层训练阶段二的独立入口。
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .pe_2d_dynamics import (
    create_pursuer_dynamics,
    create_evader_dynamics,
    _action_to_thrust_direction,
)

# 与之前一致：9 动作
NUM_PURSUER_ACCEL_ACTIONS = 9
NUM_ACCEL_ACTIONS = 9

DEFAULT_PURSUER_V_MAX = 1.2
DEFAULT_PURSUER_A_MAX = 2.5
DEFAULT_EVADER_V_MAX = 1.4
DEFAULT_EVADER_A_MAX = 0.6
DEFAULT_EVADER_TURN_RATE = 2.0
DEFAULT_CATCH_RADIUS = 0.5
DEFAULT_WORLD_SIZE = 10.0
DEFAULT_MAX_STEPS = 500


def _normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def _action_to_acceleration(action: int, a_max: float) -> np.ndarray:
    if action == 0:
        return np.zeros(2, dtype=np.float32)
    angles = np.arange(8) * (2 * np.pi / 8)
    idx = action - 1
    return np.array(
        [a_max * np.cos(angles[idx]), a_max * np.sin(angles[idx])],
        dtype=np.float32,
    )


class PursuitEvasion2DAsymmetricEnvV2DynamicsAccel(gym.Env):
    """二维非对称追逃环境 - 追击者 9 动作（与之前一致），观测 22 维。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        pursuer_v_max: float = DEFAULT_PURSUER_V_MAX,
        pursuer_a_max: float = DEFAULT_PURSUER_A_MAX,
        evader_v_max: float = DEFAULT_EVADER_V_MAX,
        evader_a_max: float = DEFAULT_EVADER_A_MAX,
        evader_turn_rate: float = DEFAULT_EVADER_TURN_RATE,
        catch_radius: float = DEFAULT_CATCH_RADIUS,
        world_size: float = DEFAULT_WORLD_SIZE,
        dt: float = 0.05,
        max_steps: int = DEFAULT_MAX_STEPS,
        difficulty: float = 1.0,
        reward_distance_improve: float = 5.0,
        reward_closing_speed: float = 1.0,
        reward_direction_align: float = 0.5,
        reward_boundary_trap: float = 0.3,
        reward_catch: float = 100.0,
        penalty_time: float = 0.05,
        penalty_boundary: float = 0.1,
        evader_mode: str = "smart",
        evader_reaction_delay: int = 0,
        evader_noise: float = 0.0,
        init_distance_range: Tuple[float, float] = (2.0, 8.0),
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.pursuer_v_max = pursuer_v_max
        self.pursuer_a_max = pursuer_a_max
        self.evader_v_max = evader_v_max
        self.evader_a_max = evader_a_max
        self.evader_turn_rate = evader_turn_rate
        self.catch_radius = catch_radius
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self.reward_distance_improve = reward_distance_improve
        self.reward_closing_speed = reward_closing_speed
        self.reward_direction_align = reward_direction_align
        self.reward_boundary_trap = reward_boundary_trap
        self.reward_catch = reward_catch
        self.penalty_time = penalty_time
        self.penalty_boundary = penalty_boundary
        self.evader_mode = evader_mode
        self.evader_reaction_delay = evader_reaction_delay
        self.evader_noise = evader_noise
        self.init_distance_range = init_distance_range
        self._p_dynamics = create_pursuer_dynamics(
            v_max=pursuer_v_max, a_max=pursuer_a_max, dt=dt
        )
        self._e_dynamics = create_evader_dynamics(
            v_max=evader_v_max, a_max=evader_a_max, dt=dt
        )
        self._apply_difficulty()
        self._p_pos = np.zeros(2, dtype=np.float32)
        self._p_vel = np.zeros(2, dtype=np.float32)
        self._p_acc = np.zeros(2, dtype=np.float32)
        self._e_pos = np.zeros(2, dtype=np.float32)
        self._e_vel = np.zeros(2, dtype=np.float32)
        self._e_heading = 0.0
        self._step_count = 0
        self._prev_distance = 0.0
        self._evader_action_buffer = []
        self.np_random = np.random.default_rng(seed)
        obs_dim = 22
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim) * 10.0, high=np.ones(obs_dim) * 10.0, dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_PURSUER_ACCEL_ACTIONS)
        self.render_mode = render_mode
        self._episode_rewards = []
        self._episode_distance_traveled = 0.0

    def _apply_difficulty(self):
        """难度只在不破坏设计约束的前提下缩放：追击者速度劣、加速度优。"""
        factor = 0.5 + 0.5 * self.difficulty
        # 硬约束：逃脱者最大速度必须始终 ≥ 追击者（否则追击者会“比逃脱者快”）
        self._effective_evader_v_max = max(
            self.evader_v_max * factor,
            self.pursuer_v_max + 0.05,
        )
        # 硬约束：逃脱者最大加速度必须始终 ≤ 追击者
        self._effective_evader_a_max = min(
            self.evader_a_max * factor,
            self.pursuer_a_max - 0.05,
        )

    def set_difficulty(self, difficulty: float):
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self._apply_difficulty()

    def _step_pursuer(self, action: int) -> bool:
        dir_vec = _action_to_thrust_direction(action)
        thrust = dir_vec * self._p_dynamics.max_thrust
        pos_new, vel_new, acc = self._p_dynamics.step(
            self._p_pos.copy(),
            self._p_vel.copy(),
            thrust,
            v_max=self.pursuer_v_max,
            a_max=self.pursuer_a_max,
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
        self._p_pos = pos_new
        self._p_vel = vel_new
        self._p_acc = acc
        return hit_boundary

    def _step_evader(self, desired_acc: np.ndarray):
        if np.linalg.norm(desired_acc) > 1e-6:
            desired_heading = np.arctan2(desired_acc[1], desired_acc[0])
            heading_diff = _normalize_angle(desired_heading - self._e_heading)
            max_turn = self.evader_turn_rate * self.dt
            heading_diff = np.clip(heading_diff, -max_turn, max_turn)
            self._e_heading = _normalize_angle(self._e_heading + heading_diff)
            acc_mag = min(np.linalg.norm(desired_acc), self._effective_evader_a_max)
            acc_desired = acc_mag * np.array(
                [np.cos(self._e_heading), np.sin(self._e_heading)], dtype=np.float32
            )
        else:
            acc_desired = np.zeros(2, dtype=np.float32)
        thrust = self._e_dynamics.thrust_for_desired_acc(self._e_vel, acc_desired)
        pos_new, vel_new, _ = self._e_dynamics.step(
            self._e_pos.copy(),
            self._e_vel.copy(),
            thrust,
            v_max=self._effective_evader_v_max,
            a_max=self._effective_evader_a_max,
        )
        for i in range(2):
            if pos_new[i] < -self.world_size:
                pos_new[i] = -self.world_size
                vel_new[i] = abs(vel_new[i]) * 0.3
            elif pos_new[i] > self.world_size:
                pos_new[i] = self.world_size
                vel_new[i] = -abs(vel_new[i]) * 0.3
        self._e_pos = pos_new
        self._e_vel = vel_new
        if np.linalg.norm(vel_new) > 0.1:
            self._e_heading = np.arctan2(vel_new[1], vel_new[0])

    def _get_evader_action(self) -> np.ndarray:
        if self.evader_mode == "random":
            action = self.np_random.integers(0, NUM_ACCEL_ACTIONS)
            return _action_to_acceleration(action, self._effective_evader_a_max)
        elif self.evader_mode == "simple":
            diff = self._e_pos - self._p_pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                return np.zeros(2)
            return (diff / dist) * self._effective_evader_a_max
        elif self.evader_mode == "smart":
            return self._smart_evader_policy()
        elif self.evader_mode == "expert":
            return self._expert_evader_policy()
        return np.zeros(2)

    def _smart_evader_policy(self) -> np.ndarray:
        diff = self._e_pos - self._p_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            angle = self.np_random.uniform(0, 2 * np.pi)
            return self._effective_evader_a_max * np.array(
                [np.cos(angle), np.sin(angle)]
            )
        escape_dir = diff / dist
        boundary_penalty = np.zeros(2)
        margin = 2.0
        if self._e_pos[0] > self.world_size - margin:
            boundary_penalty[0] -= 1.0
        if self._e_pos[0] < -self.world_size + margin:
            boundary_penalty[0] += 1.0
        if self._e_pos[1] > self.world_size - margin:
            boundary_penalty[1] -= 1.0
        if self._e_pos[1] < -self.world_size + margin:
            boundary_penalty[1] += 1.0
        combined = escape_dir + 0.5 * boundary_penalty
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-6:
            combined = combined / combined_norm
        if self.evader_noise > 0:
            noise = self.np_random.normal(0, self.evader_noise, 2)
            combined = combined + noise
            combined = combined / (np.linalg.norm(combined) + 1e-6)
        return combined * self._effective_evader_a_max

    def _expert_evader_policy(self) -> np.ndarray:
        diff = self._e_pos - self._p_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            angle = self.np_random.uniform(0, 2 * np.pi)
            return self._effective_evader_a_max * np.array(
                [np.cos(angle), np.sin(angle)]
            )
        escape_dir = diff / dist
        predicted_p_pos = self._p_pos + self._p_vel * 0.5
        predicted_diff = self._e_pos - predicted_p_pos
        predicted_dist = np.linalg.norm(predicted_diff)
        predicted_escape = (
            predicted_diff / predicted_dist if predicted_dist > 1e-6 else escape_dir
        )
        boundary_penalty = np.zeros(2)
        margin = 2.5
        if self._e_pos[0] > self.world_size - margin:
            boundary_penalty[0] -= (self._e_pos[0] - (self.world_size - margin)) / margin
        if self._e_pos[0] < -self.world_size + margin:
            boundary_penalty[0] += ((-self.world_size + margin) - self._e_pos[0]) / margin
        if self._e_pos[1] > self.world_size - margin:
            boundary_penalty[1] -= (self._e_pos[1] - (self.world_size - margin)) / margin
        if self._e_pos[1] < -self.world_size + margin:
            boundary_penalty[1] += ((-self.world_size + margin) - self._e_pos[1]) / margin
        combined = 0.4 * escape_dir + 0.4 * predicted_escape + 0.2 * boundary_penalty
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 1e-6:
            combined = combined / combined_norm
        return combined * self._effective_evader_a_max

    def _get_observation(self) -> np.ndarray:
        pos_scale = 1.0 / self.world_size
        vel_scale = 1.0 / max(self.pursuer_v_max, self.evader_v_max)
        acc_scale = 1.0 / max(self.pursuer_a_max, self.evader_a_max)
        rel_pos = self._e_pos - self._p_pos
        rel_vel = self._e_vel - self._p_vel
        distance = np.linalg.norm(rel_pos)
        closing_speed = (
            -np.dot(rel_vel, rel_pos) / distance if distance > 1e-6 else 0.0
        )
        evader_speed = np.linalg.norm(self._e_vel)
        evader_vel_dir = (
            self._e_vel / evader_speed if evader_speed > 1e-6 else np.zeros(2)
        )
        pursuer_speed = np.linalg.norm(self._p_vel)
        angle_alignment = 0.0
        if pursuer_speed > 1e-6 and distance > 1e-6:
            p_vel_dir = self._p_vel / pursuer_speed
            target_dir = rel_pos / distance
            angle_alignment = np.dot(p_vel_dir, target_dir)
        obs = np.array(
            [
                self._p_pos[0] * pos_scale,
                self._p_pos[1] * pos_scale,
                self._p_vel[0] * vel_scale,
                self._p_vel[1] * vel_scale,
                rel_pos[0] * pos_scale,
                rel_pos[1] * pos_scale,
                rel_vel[0] * vel_scale,
                rel_vel[1] * vel_scale,
                distance * pos_scale,
                closing_speed * vel_scale,
                evader_vel_dir[0],
                evader_vel_dir[1],
                (self.world_size - self._p_pos[1]) * pos_scale,
                (self._p_pos[1] + self.world_size) * pos_scale,
                (self._p_pos[0] + self.world_size) * pos_scale,
                (self.world_size - self._p_pos[0]) * pos_scale,
                min(self.world_size - abs(self._e_pos[0]), self.world_size - abs(self._e_pos[1])) * pos_scale,
                pursuer_speed * vel_scale,
                evader_speed * vel_scale,
                angle_alignment,
                self._p_acc[0] * acc_scale,
                self._p_acc[1] * acc_scale,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -10.0, 10.0)

    def _compute_reward(self, hit_boundary: bool) -> Tuple[float, Dict[str, float]]:
        reward, reward_info = 0.0, {}
        distance = np.linalg.norm(self._e_pos - self._p_pos)
        r_improve = self.reward_distance_improve * (self._prev_distance - distance)
        reward += r_improve
        reward_info["r_improve"] = r_improve
        rel_pos = self._e_pos - self._p_pos
        rel_vel = self._e_vel - self._p_vel
        r_closing = 0.0
        if distance > 1e-6:
            closing_speed = -np.dot(rel_vel, rel_pos) / distance
            r_closing = self.reward_closing_speed * np.clip(closing_speed, -1.0, 1.0)
        reward += r_closing
        reward_info["r_closing"] = r_closing
        pursuer_speed = np.linalg.norm(self._p_vel)
        r_align = 0.0
        if pursuer_speed > 0.1 and distance > 1e-6:
            vel_dir = self._p_vel / pursuer_speed
            target_dir = rel_pos / distance
            r_align = self.reward_direction_align * np.dot(vel_dir, target_dir)
        reward += r_align
        reward_info["r_align"] = r_align
        evader_to_boundary = min(
            self.world_size - abs(self._e_pos[0]),
            self.world_size - abs(self._e_pos[1]),
        )
        r_boundary_trap = (
            self.reward_boundary_trap * (2.0 - evader_to_boundary) / 2.0
            if evader_to_boundary < 2.0
            else 0.0
        )
        reward += r_boundary_trap
        reward_info["r_boundary_trap"] = r_boundary_trap
        reward += -self.penalty_time
        reward_info["r_time"] = -self.penalty_time
        if hit_boundary:
            reward += -self.penalty_boundary
            reward_info["r_boundary_hit"] = -self.penalty_boundary
        if distance < 2.0:
            r_proximity = 0.2 * (2.0 - distance)
            reward += r_proximity
            reward_info["r_proximity"] = r_proximity
        if distance < self.catch_radius:
            reward += self.reward_catch
            reward_info["r_catch"] = self.reward_catch
        self._prev_distance = distance
        return reward, reward_info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        super().reset(seed=seed)
        if options:
            if "difficulty" in options:
                self.set_difficulty(options["difficulty"])
            if "evader_mode" in options:
                self.evader_mode = options["evader_mode"]
        min_dist, max_dist = self.init_distance_range
        self._p_pos = self.np_random.uniform(
            -self.world_size * 0.7, self.world_size * 0.7, size=2
        ).astype(np.float32)
        for _ in range(100):
            self._e_pos = self.np_random.uniform(
                -self.world_size * 0.7, self.world_size * 0.7, size=2
            ).astype(np.float32)
            dist = np.linalg.norm(self._e_pos - self._p_pos)
            if min_dist <= dist <= max_dist:
                break
        self._p_vel = np.zeros(2, dtype=np.float32)
        self._p_acc = np.zeros(2, dtype=np.float32)
        self._e_vel = np.zeros(2, dtype=np.float32)
        diff = self._e_pos - self._p_pos
        self._e_heading = np.arctan2(diff[1], diff[0])
        self._step_count = 0
        self._prev_distance = np.linalg.norm(self._e_pos - self._p_pos)
        self._evader_action_buffer = []
        self._episode_rewards = []
        self._episode_distance_traveled = 0.0
        obs = self._get_observation()
        info = {
            "distance": float(self._prev_distance),
            "pursuer_pos": self._p_pos.copy(),
            "evader_pos": self._e_pos.copy(),
            "difficulty": self.difficulty,
            "evader_mode": self.evader_mode,
        }
        return obs, info

    def step(
        self, action: int, evader_action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        evader_action = evader_action if evader_action is not None else self._get_evader_action()
        if self.evader_reaction_delay > 0:
            self._evader_action_buffer.append(evader_action)
            if len(self._evader_action_buffer) > self.evader_reaction_delay:
                evader_action = self._evader_action_buffer.pop(0)
            else:
                evader_action = np.zeros(2)
        old_p_pos = self._p_pos.copy()
        hit_boundary = self._step_pursuer(int(action))
        self._step_evader(evader_action)
        self._episode_distance_traveled += np.linalg.norm(self._p_pos - old_p_pos)
        reward, reward_info = self._compute_reward(hit_boundary)
        self._episode_rewards.append(reward)
        distance = np.linalg.norm(self._e_pos - self._p_pos)
        caught = distance < self.catch_radius
        terminated = caught
        truncated = self._step_count >= self.max_steps
        obs = self._get_observation()
        info = {
            "distance": float(distance),
            "pursuer_pos": self._p_pos.copy(),
            "evader_pos": self._e_pos.copy(),
            "pursuer_vel": self._p_vel.copy(),
            "evader_vel": self._e_vel.copy(),
            "caught": caught,
            "step": self._step_count,
            "reward_info": reward_info,
        }
        if terminated or truncated:
            info["episode_reward"] = sum(self._episode_rewards)
            info["episode_length"] = self._step_count
            info["distance_traveled"] = self._episode_distance_traveled
            info["success"] = caught
        return obs, float(reward), terminated, truncated, info

    def get_evader_obs(self) -> np.ndarray:
        pos_scale = 1.0 / self.world_size
        vel_scale = 1.0 / max(self.pursuer_v_max, self.evader_v_max)
        rel_pos = self._p_pos - self._e_pos
        rel_vel = self._p_vel - self._e_vel
        distance = np.linalg.norm(rel_pos)
        closing_speed = -np.dot(rel_vel, rel_pos) / distance if distance > 1e-6 else 0.0
        pursuer_speed = np.linalg.norm(self._p_vel)
        pursuer_vel_dir = self._p_vel / pursuer_speed if pursuer_speed > 1e-6 else np.zeros(2)
        evader_speed = np.linalg.norm(self._e_vel)
        angle_alignment = 0.0
        if evader_speed > 1e-6 and distance > 1e-6:
            angle_alignment = np.dot(self._e_vel / evader_speed, rel_pos / distance)
        pursuer_to_boundary = min(
            self.world_size - abs(self._p_pos[0]),
            self.world_size - abs(self._p_pos[1]),
        )
        obs = np.array(
            [
                self._e_pos[0] * pos_scale,
                self._e_pos[1] * pos_scale,
                self._e_vel[0] * vel_scale,
                self._e_vel[1] * vel_scale,
                rel_pos[0] * pos_scale,
                rel_pos[1] * pos_scale,
                rel_vel[0] * vel_scale,
                rel_vel[1] * vel_scale,
                distance * pos_scale,
                closing_speed * vel_scale,
                pursuer_vel_dir[0],
                pursuer_vel_dir[1],
                (self.world_size - self._e_pos[1]) * pos_scale,
                (self._e_pos[1] + self.world_size) * pos_scale,
                (self._e_pos[0] + self.world_size) * pos_scale,
                (self.world_size - self._e_pos[0]) * pos_scale,
                pursuer_to_boundary * pos_scale,
                evader_speed * vel_scale,
                pursuer_speed * vel_scale,
                angle_alignment,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -10.0, 10.0)

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "pursuer_pos": self._p_pos.copy(),
            "pursuer_vel": self._p_vel.copy(),
            "pursuer_acc": self._p_acc.copy(),
            "evader_pos": self._e_pos.copy(),
            "evader_vel": self._e_vel.copy(),
            "evader_heading": np.array([self._e_heading]),
        }

    def render(self):
        return None

    def close(self):
        pass


def make_easy_env(**kwargs):
    kw = {"difficulty": 0.3, "evader_mode": "simple", "evader_reaction_delay": 2, "init_distance_range": (1.5, 4.0)}
    kw.update(kwargs)
    return PursuitEvasion2DAsymmetricEnvV2DynamicsAccel(**kw)


def make_medium_env(**kwargs):
    kw = {"difficulty": 0.6, "evader_mode": "smart", "evader_reaction_delay": 1, "evader_noise": 0.1, "init_distance_range": (2.0, 6.0)}
    kw.update(kwargs)
    return PursuitEvasion2DAsymmetricEnvV2DynamicsAccel(**kw)


def make_hard_env(**kwargs):
    kw = {"difficulty": 1.0, "evader_mode": "smart", "evader_reaction_delay": 0, "evader_noise": 0.05, "init_distance_range": (3.0, 8.0)}
    kw.update(kwargs)
    return PursuitEvasion2DAsymmetricEnvV2DynamicsAccel(**kw)


def make_expert_env(**kwargs):
    kw = {"difficulty": 1.0, "evader_mode": "expert", "evader_reaction_delay": 0, "init_distance_range": (4.0, 8.0)}
    kw.update(kwargs)
    return PursuitEvasion2DAsymmetricEnvV2DynamicsAccel(**kw)
