"""
Option A 环境：高层离散宏动作 + 底层连续 PID 执行
================================================

基于 PursuitEvasion2DAsymmetricEnvV2PID 扩展：
- 高层动作每 K 步决策一次（离散 9 动作）
- 底层不再使用离散推力动作，而是连续 waypoint -> PID -> thrust
- 保留原环境的动力学、奖励、课程难度与终止判据
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

from .env_asymmetric_pe_v2_pid import PursuitEvasion2DAsymmetricEnvV2PID

NUM_HIGH_ACTIONS = 9
DEFAULT_HIGH_WAYPOINT_DIST = 2.0


class PursuitEvasion2DAsymmetricEnvV2PIDOptionA(PursuitEvasion2DAsymmetricEnvV2PID):
    """Option A: 高层离散 + 底层连续 PID 追踪子目标。"""

    def __init__(
        self,
        high_waypoint_dist: float = DEFAULT_HIGH_WAYPOINT_DIST,
        include_acc_in_high_obs: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.high_waypoint_dist = float(high_waypoint_dist)
        self.include_acc_in_high_obs = bool(include_acc_in_high_obs)

    def _step_pursuer_to_waypoint(self, waypoint: np.ndarray) -> bool:
        """底层连续执行：PID 跟踪 waypoint 并通过动力学推进一步。"""
        waypoint = np.asarray(waypoint, dtype=np.float32).reshape(2)
        thrust, _ = self._p_pid.compute_thrust(
            waypoint, self._p_pos.copy(), self._p_vel.copy()
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

    def high_action_to_waypoint(self, high_action: int) -> np.ndarray:
        """高层动作 -> waypoint（0 表示直接追当前 evader 位置）。"""
        if int(high_action) == 0:
            return self._e_pos.copy()
        angle = (int(high_action) - 1) * (2 * np.pi / 8)
        offset = self.high_waypoint_dist * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        waypoint = self._p_pos + offset
        return np.clip(waypoint, -self.world_size, self.world_size)

    def get_high_level_obs(self) -> np.ndarray:
        """高层观测：相对位置、双方速度，可选追击者加速度。"""
        pos_scale = 1.0 / self.world_size
        vel_scale = 1.0 / max(self.pursuer_v_max, self.evader_v_max)
        acc_scale = 1.0 / max(self.pursuer_a_max, self.evader_a_max)

        rel_pos = (self._e_pos - self._p_pos) * pos_scale
        p_vel = self._p_vel * vel_scale
        e_vel = self._e_vel * vel_scale

        if self.include_acc_in_high_obs:
            p_acc = self._p_acc * acc_scale
            obs = np.concatenate([rel_pos, p_vel, e_vel, p_acc]).astype(np.float32)
        else:
            obs = np.concatenate([rel_pos, p_vel, e_vel]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def step_with_waypoint(
        self,
        waypoint: np.ndarray,
        evader_action: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行 1 个原子步：连续 waypoint -> PID -> dynamics。"""
        self._step_count += 1

        evader_action = evader_action if evader_action is not None else self._get_evader_action()
        if self.evader_reaction_delay > 0:
            self._evader_action_buffer.append(evader_action)
            if len(self._evader_action_buffer) > self.evader_reaction_delay:
                evader_action = self._evader_action_buffer.pop(0)
            else:
                evader_action = np.zeros(2)

        old_p_pos = self._p_pos.copy()
        hit_boundary = self._step_pursuer_to_waypoint(waypoint)
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
            "pursuer_acc": self._p_acc.copy(),
            "caught": caught,
            "step": self._step_count,
            "reward_info": reward_info,
            "waypoint": np.asarray(waypoint, dtype=np.float32).copy(),
        }
        if terminated or truncated:
            info["episode_reward"] = float(sum(self._episode_rewards))
            info["episode_length"] = self._step_count
            info["distance_traveled"] = self._episode_distance_traveled
            info["success"] = caught

        return obs, float(reward), terminated, truncated, info

    def step_macro(
        self,
        high_action: int,
        macro_steps: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一个高层宏动作，内部运行 macro_steps 个底层连续控制步。"""
        macro_steps = int(max(1, macro_steps))
        high_action = int(high_action)

        # 0 号动作表示“追当前目标”，每个原子步动态更新 waypoint；其余动作固定 waypoint
        fixed_waypoint = None if high_action == 0 else self.high_action_to_waypoint(high_action)

        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = self._get_observation()
        last_info: Dict[str, Any] = {}

        for k in range(macro_steps):
            waypoint = self._e_pos.copy() if high_action == 0 else fixed_waypoint
            last_obs, reward, terminated, truncated, info = self.step_with_waypoint(waypoint)
            total_reward += reward
            last_info = info
            if terminated or truncated:
                break

        last_info = dict(last_info)
        last_info["macro_action"] = high_action
        last_info["macro_steps_executed"] = k + 1
        last_info["macro_reward"] = float(total_reward)
        last_info["high_obs"] = self.get_high_level_obs().copy()

        return last_obs, float(total_reward), terminated, truncated, last_info
