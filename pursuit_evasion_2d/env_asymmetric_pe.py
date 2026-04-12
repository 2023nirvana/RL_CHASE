"""
二维非对称追逃博弈环境（速度 vs 加速度）
=========================================

研究思路：追击者最大速度 < 逃脱者，但最大加速度 > 逃脱者；
借助地形或预测策略，利用短时机动性完成追捕。

动力学：双积分器，速度与加速度约束分离。
- 追击者 (P): v_max 较小, a_max 较大
- 逃脱者 (E): v_max 较大, a_max 较小
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 默认非对称参数：P 速度劣势、加速度优势
DEFAULT_PURSUER_V_MAX = 1.0
DEFAULT_PURSUER_A_MAX = 2.0
DEFAULT_EVADER_V_MAX = 1.5
DEFAULT_EVADER_A_MAX = 0.8

# 追捕成功距离
DEFAULT_CATCH_RADIUS = 0.4
# 世界边界
DEFAULT_WORLD_SIZE = 10.0
# 离散动作：无加速度 + 8 方向 (a_max 大小)
NUM_ACCEL_ACTIONS = 9  # 0=no_acc, 1~8=8 directions


def _clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(x)
    if n <= max_norm or max_norm <= 0:
        return x
    return x * (max_norm / n)


def _action_to_acceleration(action: int, a_max: float) -> np.ndarray:
    """离散动作 -> 2D 加速度向量 (世界坐标系)"""
    if action == 0:
        return np.zeros(2)
    # 1~8: 上/右上/右/右下/下/左下/左/左上
    angles = (np.arange(8) + 1) * (2 * np.pi / 8)  # 从 45° 起
    idx = action - 1
    ax = a_max * np.cos(angles[idx])
    ay = a_max * np.sin(angles[idx])
    return np.array([ax, ay], dtype=np.float32)


class PursuitEvasion2DAsymmetricEnv(gym.Env):
    """
    二维追逃环境：双积分器 + 非对称 v_max / a_max。

    状态：
        pursuer: pos(2), vel(2), acc(2)  # 上一步施加的加速度
        evader:  pos(2), vel(2), acc(2)
    观测（对追击者）：
        [self_pos, self_vel, self_acc, other_pos, other_vel] 归一化
    动作：
        Discrete(9): 0=不加速, 1~8=8 方向以 a_max 加速
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        pursuer_v_max: float = DEFAULT_PURSUER_V_MAX,
        pursuer_a_max: float = DEFAULT_PURSUER_A_MAX,
        evader_v_max: float = DEFAULT_EVADER_V_MAX,
        evader_a_max: float = DEFAULT_EVADER_A_MAX,
        catch_radius: float = DEFAULT_CATCH_RADIUS,
        world_size: float = DEFAULT_WORLD_SIZE,
        dt: float = 0.05,
        max_steps: int = 1000,
        obs_scale_pos: float = 1.0 / 10.0,
        obs_scale_vel: float = 1.0 / 2.0,
        obs_scale_acc: float = 1.0 / 3.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        obstacles: Optional[List[Tuple[float, float, float, float]]] = None,
    ):
        """
        obstacles: 矩形障碍物列表 [(x_min, y_min, x_max, y_max), ...]，可选
        """
        super().__init__()
        self.pursuer_v_max = pursuer_v_max
        self.pursuer_a_max = pursuer_a_max
        self.evader_v_max = evader_v_max
        self.evader_a_max = evader_a_max
        self.catch_radius = catch_radius
        self.world_size = world_size
        self.dt = dt
        self.max_steps = max_steps
        self.obs_scale_pos = obs_scale_pos
        self.obs_scale_vel = obs_scale_vel
        self.obs_scale_acc = obs_scale_acc
        self.obstacles = obstacles or []

        # 状态: P 与 E 的 pos(2), vel(2), acc(2)
        self._p_pos = np.zeros(2)
        self._p_vel = np.zeros(2)
        self._p_acc = np.zeros(2)
        self._e_pos = np.zeros(2)
        self._e_vel = np.zeros(2)
        self._e_acc = np.zeros(2)

        self._step_count = 0
        self.np_random = np.random.default_rng(seed)

        # 观测：对追击者 [self_pos(2), self_vel(2), self_acc(2), other_pos(2), other_vel(2)] = 10
        obs_dim = 10
        obs_high = np.ones(obs_dim) * 10.0
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACCEL_ACTIONS)

        self.render_mode = render_mode

    def _step_dynamics(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        v_max: float,
        a_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        acc_clip = _clip_norm(acc, a_max)
        vel_new = vel + acc_clip * self.dt
        vel_new = _clip_norm(vel_new, v_max)
        pos_new = pos + vel_new * self.dt
        # 边界：反弹或截断
        for i in range(2):
            if pos_new[i] < -self.world_size:
                pos_new[i] = -self.world_size
                vel_new[i] = max(0.0, vel_new[i])
            elif pos_new[i] > self.world_size:
                pos_new[i] = self.world_size
                vel_new[i] = min(0.0, vel_new[i])
        return pos_new, vel_new

    def _check_obstacle(self, pos: np.ndarray) -> bool:
        for (x0, y0, x1, y1) in self.obstacles:
            if x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1:
                return True
        return False

    def _get_pursuer_obs(self) -> np.ndarray:
        # [self_pos, self_vel, self_acc, other_pos, other_vel] 归一化
        obs = np.concatenate(
            [
                self._p_pos * self.obs_scale_pos,
                self._p_vel * self.obs_scale_vel,
                self._p_acc * self.obs_scale_acc,
                self._e_pos * self.obs_scale_pos,
                self._e_vel * self.obs_scale_vel,
            ]
        ).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _get_evader_obs(self) -> np.ndarray:
        # 逃脱者观测：己方 + 追击者
        obs = np.concatenate(
            [
                self._e_pos * self.obs_scale_pos,
                self._e_vel * self.obs_scale_vel,
                self._e_acc * self.obs_scale_acc,
                self._p_pos * self.obs_scale_pos,
                self._p_vel * self.obs_scale_vel,
            ]
        ).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        super().reset(seed=seed)

        # 随机初始位置（避免重叠）
        self._p_pos = self.np_random.uniform(
            -self.world_size * 0.8, self.world_size * 0.8, size=2
        )
        self._e_pos = self.np_random.uniform(
            -self.world_size * 0.8, self.world_size * 0.8, size=2
        )
        while np.linalg.norm(self._e_pos - self._p_pos) < self.catch_radius * 3:
            self._e_pos = self.np_random.uniform(
                -self.world_size * 0.8, self.world_size * 0.8, size=2
            )
        self._p_vel = np.zeros(2)
        self._p_acc = np.zeros(2)
        self._e_vel = np.zeros(2)
        self._e_acc = np.zeros(2)
        self._step_count = 0

        # 默认返回追击者视角（用于单智能体训练追击策略）
        obs = self._get_pursuer_obs()
        info = {
            "pursuer_pos": self._p_pos.copy(),
            "evader_pos": self._e_pos.copy(),
            "distance": np.linalg.norm(self._p_pos - self._e_pos),
        }
        return obs, info

    def step(
        self, action: int, evader_action: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 追击者动作
        p_acc = _action_to_acceleration(int(action), self.pursuer_a_max)
        # 逃脱者：若未传入则随机
        if evader_action is None:
            evader_action = self.np_random.integers(0, NUM_ACCEL_ACTIONS)
        e_acc = _action_to_acceleration(int(evader_action), self.evader_a_max)

        self._p_acc = p_acc
        self._e_acc = e_acc

        # 先更新追击者，再逃脱者（可改为同时更新，此处顺序无本质影响）
        self._p_pos, self._p_vel = self._step_dynamics(
            self._p_pos, self._p_vel, p_acc,
            self.pursuer_v_max, self.pursuer_a_max,
        )
        self._e_pos, self._e_vel = self._step_dynamics(
            self._e_pos, self._e_vel, e_acc,
            self.evader_v_max, self.evader_a_max,
        )

        # 障碍物：若新位置在障碍内则回退到旧位置（简化）
        # 可改为反弹，这里仅防止穿障
        if self._check_obstacle(self._p_pos):
            self._p_pos = self._p_pos - self._p_vel * self.dt
            self._p_vel = 0.9 * self._p_vel
        if self._check_obstacle(self._e_pos):
            self._e_pos = self._e_pos - self._e_vel * self.dt
            self._e_vel = 0.9 * self._e_vel

        self._step_count += 1
        dist = np.linalg.norm(self._p_pos - self._e_pos)

        # 奖励：追击者希望减小距离
        reward = -dist * 0.1
        terminated = False
        if dist < self.catch_radius:
            reward = 100.0
            terminated = True
        truncated = self._step_count >= self.max_steps

        obs = self._get_pursuer_obs()
        info = {
            "distance": float(dist),
            "pursuer_pos": self._p_pos.copy(),
            "evader_pos": self._e_pos.copy(),
            "caught": terminated,
        }
        return obs, float(reward), terminated, truncated, info

    def get_evader_obs(self) -> np.ndarray:
        return self._get_evader_obs()

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "pursuer_pos": self._p_pos.copy(),
            "pursuer_vel": self._p_vel.copy(),
            "evader_pos": self._e_pos.copy(),
            "evader_vel": self._e_vel.copy(),
        }

    def set_evader_action(self, evader_action: int):
        """用于固定 evader 策略或外部控制"""
        self._evader_action_override = evader_action

    def render(self):
        if self.render_mode is None:
            return None
        # 简单 ASCII 或返回 RGB 需自己实现
        return None

    def close(self):
        pass
