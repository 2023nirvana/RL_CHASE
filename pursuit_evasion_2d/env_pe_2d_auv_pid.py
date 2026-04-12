"""
二维追逃 - AUV 配置 + 双环 PID+RL（与 low_level 一致）
======================================================

- 动力学参数从 configs/robot_config.yaml 读取；v_max/a_max 硬约束。
- 控制流程采用与 core/control/dual_loop_pid IntegratedController 相同的双环 PID：
  - 从 configs/low_level_config_2d.yaml 读取 velocity_controller、goal_slowdown、action_distance；
  - 外环：动作 -> 每步位移 linear_cmd -> desired_vel = linear_cmd/Ts（近目标时 goal_slowdown）；
  - 内环：force = Kp_vel*(desired_vel - vel) - Kd_vel*vel，按 max_force 限幅；
  - 推力 = force_2d 送入 Dynamics2D.step(τ, v_max, a_max)。
- 默认 world_size=5 便于小范围试验。
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .pe_2d_auv_config import (
    create_pursuer_dynamics_from_config,
    create_evader_dynamics_from_config,
    get_pursuer_evader_limits_from_config,
)
from .pe_2d_dual_loop_pid import DualLoopPID2D

DEFAULT_WORLD_SIZE = 5.0  # 小范围试验
DEFAULT_CATCH_RADIUS = 0.4
DEFAULT_MAX_STEPS = 400
NUM_ACTIONS = 9


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


class PursuitEvasion2DAUVPIDEnv(gym.Env):
    """
    二维追逃：AUV 配置动力学 + PID+RL，硬约束 v_max/a_max。
    默认 5x5 世界，适合先做小范围验证。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config_path: Optional[str] = None,
        pursuer_v_max: Optional[float] = None,
        pursuer_a_max: Optional[float] = None,
        evader_v_max: Optional[float] = None,
        evader_a_max: Optional[float] = None,
        evader_turn_rate: float = 2.0,
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
        init_distance_range: Tuple[float, float] = (1.0, 3.5),
        low_level_config_path: Optional[str] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config_path = config_path
        pv, pa, ev, ea = get_pursuer_evader_limits_from_config(config_path)
        self.pursuer_v_max = pursuer_v_max if pursuer_v_max is not None else pv
        self.pursuer_a_max = pursuer_a_max if pursuer_a_max is not None else pa
        self.evader_v_max = evader_v_max if evader_v_max is not None else ev
        self.evader_a_max = evader_a_max if evader_a_max is not None else ea
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
        self.low_level_config_path = low_level_config_path

        self._p_dynamics = create_pursuer_dynamics_from_config(
            config_path,
            dt=dt,
            v_max_override=self.pursuer_v_max,
            a_max_override=self.pursuer_a_max,
        )
        self._e_dynamics = create_evader_dynamics_from_config(
            config_path,
            dt=dt,
            v_max_override=self.evader_v_max,
            a_max_override=self.evader_a_max,
        )
        self._p_dual_loop = DualLoopPID2D(config_path=low_level_config_path)
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
            low=-np.ones(obs_dim) * 10.0,
            high=np.ones(obs_dim) * 10.0,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.render_mode = render_mode
        self._episode_rewards = []
        self._episode_distance_traveled = 0.0

    def _apply_difficulty(self):
        factor = 0.5 + 0.5 * self.difficulty
        self._effective_evader_v_max = max(
            self.evader_v_max * factor,
            self.pursuer_v_max + 0.05,
        )
        self._effective_evader_a_max = min(
            self.evader_a_max * factor,
            self.pursuer_a_max - 0.05,
        )

    def set_difficulty(self, difficulty: float):
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self._apply_difficulty()

    def _step_pursuer(self, action: int) -> bool:
        thrust, _ = self._p_dual_loop.compute_thrust(
            action,
            self._p_pos.copy(),
            self._p_vel.copy(),
            self.dt,
            target_position=self._e_pos.copy(),
        )
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
                [np.cos(self._e_heading), np.sin(self._e_heading)],
                dtype=np.float32,
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
            action = self.np_random.integers(0, NUM_ACTIONS)
            return _action_to_acceleration(action, self._effective_evader_a_max)
        if self.evader_mode == "simple":
            diff = self._e_pos - self._p_pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                return np.zeros(2)
            return (diff / dist) * self._effective_evader_a_max
        if self.evader_mode == "smart":
            return self._smart_evader_policy()
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
        margin = 1.2
        if self._e_pos[0] > self.world_size - margin:
            boundary_penalty[0] -= 1.0
        if self._e_pos[0] < -self.world_size + margin:
            boundary_penalty[0] += 1.0
        if self._e_pos[1] > self.world_size - margin:
            boundary_penalty[1] -= 1.0
        if self._e_pos[1] < -self.world_size + margin:
            boundary_penalty[1] += 1.0
        combined = escape_dir + 0.5 * boundary_penalty
        n = np.linalg.norm(combined)
        if n > 1e-6:
            combined = combined / n
        if self.evader_noise > 0:
            noise = self.np_random.normal(0, self.evader_noise, 2)
            combined = combined + noise
            combined = combined / (np.linalg.norm(combined) + 1e-6)
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
                min(
                    self.world_size - abs(self._e_pos[0]),
                    self.world_size - abs(self._e_pos[1]),
                )
                * pos_scale,
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
        evader_action = (
            evader_action
            if evader_action is not None
            else self._get_evader_action()
        )
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

    def get_state(self) -> Dict[str, np.ndarray]:
        """与 env_asymmetric_pe_v2_dynamics_accel 一致，供可视化用。"""
        return {
            "pursuer_pos": self._p_pos.copy(),
            "pursuer_vel": self._p_vel.copy(),
            "pursuer_acc": self._p_acc.copy(),
            "evader_pos": self._e_pos.copy(),
            "evader_vel": self._e_vel.copy(),
        }

    def render(self):
        return None

    def close(self):
        pass
