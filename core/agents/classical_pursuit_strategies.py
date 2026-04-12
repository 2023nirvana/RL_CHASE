"""
经典追击策略模块
================
实现多种经典制导/追击策略，输出格式与 HRL 高层完全兼容：
  输入: pursuer state + evader state + subgoal_radius
  输出: subgoal_world (2D world-frame position)

所有策略继承 `BaseStrategy`，对外统一接口 `compute_subgoal()`。
支持的策略：
  - PurePursuit  : 直接朝逃脱者当前位置
  - LeadPursuit  : 线性预测拦截点
  - ProNav (PN)  : 比例导引
  - AugProNav (APN): 增强比例导引（考虑目标加速度）
  - ConstantBearing (CB): 常方位角碰撞航向
  - DeviatedPursuit (DP): 偏置追击（PP + 固定偏角）

与 train_pe_high_level_continuous_hrl.py 中的接口对齐：
  subgoal_world = strategy.compute_subgoal(
      pursuer_pos, pursuer_vel, pursuer_yaw,
      evader_pos, evader_vel,
      subgoal_radius
  )
  随后由底层 NN 执行 _apply_continuous_low_level_action(env, low_act, subgoal_world)
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


def _angle_wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _unit_vec(angle: float) -> np.ndarray:
    """Unit vector from angle."""
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


# ─────────────────────── Base Class ───────────────────────

class BaseStrategy(ABC):
    """所有经典策略的基类。"""

    def __init__(self, name: str = "base"):
        self.name = name
        # 用于需要上一步状态差分的策略
        self._prev_los_angle: Optional[float] = None
        self._prev_evader_vel: Optional[np.ndarray] = None
        self._dt: float = 0.02  # env dt, will be overridden if needed

    def reset(self):
        """Episode/场景重置时调用。"""
        self._prev_los_angle = None
        self._prev_evader_vel = None

    @abstractmethod
    def _compute_desired_direction(
        self,
        pursuer_pos: np.ndarray,
        pursuer_vel: np.ndarray,
        pursuer_yaw: float,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
    ) -> np.ndarray:
        """
        返回期望运动方向的 *单位向量* (world frame, 2D)。
        子类只需实现此方法。
        """
        ...

    def compute_subgoal(
        self,
        pursuer_pos: np.ndarray,
        pursuer_vel: np.ndarray,
        pursuer_yaw: float,
        evader_pos: np.ndarray,
        evader_vel: np.ndarray,
        subgoal_radius: float,
        world_size: float = 10.0,
    ) -> np.ndarray:
        """
        统一接口 → 返回 world-frame subgoal 位置 (shape=(2,))。
        subgoal = pursuer_pos + direction * subgoal_radius, clipped to world.
        """
        direction = self._compute_desired_direction(
            np.asarray(pursuer_pos, dtype=np.float64)[:2],
            np.asarray(pursuer_vel, dtype=np.float64)[:2],
            float(pursuer_yaw),
            np.asarray(evader_pos, dtype=np.float64)[:2],
            np.asarray(evader_vel, dtype=np.float64)[:2],
        )
        direction = _safe_normalize(direction)
        if np.linalg.norm(direction) < 0.5:
            # fallback: head toward evader
            direction = _safe_normalize(
                np.asarray(evader_pos[:2]) - np.asarray(pursuer_pos[:2])
            )
            if np.linalg.norm(direction) < 0.5:
                direction = _unit_vec(pursuer_yaw)  # last resort

        subgoal = np.asarray(pursuer_pos[:2], dtype=np.float64) + direction * subgoal_radius
        subgoal = np.clip(subgoal, -world_size, world_size)
        return subgoal

    def set_dt(self, dt: float):
        self._dt = dt

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


# ─────────────────────── 1. Pure Pursuit ───────────────────────

class PurePursuit(BaseStrategy):
    """
    纯追踪 (PP): 永远朝逃脱者 *当前位置* 前进。
    最简单的基准策略，不考虑目标运动。
    """

    def __init__(self):
        super().__init__(name="PurePursuit")

    def _compute_desired_direction(self, pursuer_pos, pursuer_vel, pursuer_yaw,
                                   evader_pos, evader_vel):
        return evader_pos - pursuer_pos


# ─────────────────────── 2. Lead Pursuit (Intercept) ───────────────────────

class LeadPursuit(BaseStrategy):
    """
    前置追踪 (Lead/Intercept): 线性预测逃脱者 τ 秒后的位置并朝该点。
    τ = dist / pursuer_speed，clamp 到 [tau_min, tau_max]。
    """

    def __init__(self, pursuer_max_speed: float = 2.0,
                 tau_min: float = 0.2, tau_max: float = 3.0):
        super().__init__(name="LeadPursuit")
        self.v_max = pursuer_max_speed
        self.tau_min = tau_min
        self.tau_max = tau_max

    def _compute_desired_direction(self, pursuer_pos, pursuer_vel, pursuer_yaw,
                                   evader_pos, evader_vel):
        rel = evader_pos - pursuer_pos
        dist = np.linalg.norm(rel)
        pursuer_speed = max(np.linalg.norm(pursuer_vel), 0.1)
        tau = np.clip(dist / pursuer_speed, self.tau_min, self.tau_max)
        intercept = evader_pos + evader_vel * tau
        return intercept - pursuer_pos


# ─────────────────────── 3. Proportional Navigation (PN) ───────────────────────

class ProNav(BaseStrategy):
    """
    比例导引 (Proportional Navigation):
      a_cmd ⊥ LOS = N * V_c * dλ/dt
    其中 N 为导引系数 (典型 3~5)，λ 为 LOS 角，V_c 为接近速度。

    实现: 每步估算 LOS 角速率 dλ/dt，据此偏转追击方向。
    """

    def __init__(self, N: float = 4.0):
        super().__init__(name=f"ProNav(N={N})")
        self.N = N

    def _compute_desired_direction(self, pursuer_pos, pursuer_vel, pursuer_yaw,
                                   evader_pos, evader_vel):
        rel_pos = evader_pos - pursuer_pos
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-6:
            return _unit_vec(pursuer_yaw)

        los_angle = np.arctan2(rel_pos[1], rel_pos[0])

        # Estimate LOS rate (dλ/dt)
        if self._prev_los_angle is not None:
            d_los = _angle_wrap(los_angle - self._prev_los_angle)
            los_rate = d_los / self._dt
        else:
            los_rate = 0.0
        self._prev_los_angle = los_angle

        # PN lateral acceleration direction:
        # desired heading = LOS angle + bias proportional to LOS rate
        # higher LOS rate → larger lead angle
        # Clamp the correction to avoid instability
        correction = np.clip(self.N * los_rate * self._dt, -np.pi / 3, np.pi / 3)
        desired_heading = los_angle + correction

        return _unit_vec(desired_heading)


# ─────────────────────── 4. Augmented PN (APN) ───────────────────────

class AugProNav(BaseStrategy):
    """
    增强比例导引 (Augmented PN):
      a_cmd = N * V_c * dλ/dt + (N/2) * a_T⊥
    其中 a_T⊥ 是目标加速度的 LOS 法向分量。
    对机动目标（如变向逃脱者）比 PN 更鲁棒。
    """

    def __init__(self, N: float = 4.0):
        super().__init__(name=f"AugProNav(N={N})")
        self.N = N

    def _compute_desired_direction(self, pursuer_pos, pursuer_vel, pursuer_yaw,
                                   evader_pos, evader_vel):
        rel_pos = evader_pos - pursuer_pos
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-6:
            return _unit_vec(pursuer_yaw)

        los_angle = np.arctan2(rel_pos[1], rel_pos[0])

        # LOS rate
        if self._prev_los_angle is not None:
            d_los = _angle_wrap(los_angle - self._prev_los_angle)
            los_rate = d_los / self._dt
        else:
            los_rate = 0.0
        self._prev_los_angle = los_angle

        # Target lateral acceleration (estimate from velocity change)
        if self._prev_evader_vel is not None:
            evader_accel = (evader_vel - self._prev_evader_vel) / self._dt
            # Project accel onto LOS-normal direction
            los_dir = _unit_vec(los_angle)
            los_normal = np.array([-los_dir[1], los_dir[0]])
            a_t_perp = float(np.dot(evader_accel, los_normal))
        else:
            a_t_perp = 0.0
        self._prev_evader_vel = evader_vel.copy()

        # APN correction: PN term + augmentation term
        # Convert acceleration-level to heading correction
        pursuer_speed = max(np.linalg.norm(pursuer_vel), 0.3)
        pn_term = self.N * los_rate
        aug_term = (self.N / 2.0) * a_t_perp / pursuer_speed
        correction = np.clip((pn_term + aug_term) * self._dt, -np.pi / 3, np.pi / 3)
        desired_heading = los_angle + correction

        return _unit_vec(desired_heading)


# ─────────────────────── 5. Constant Bearing (CB) ───────────────────────

class ConstantBearing(BaseStrategy):
    """
    常方位角策略 (Constant Bearing / Collision Course):
    保持 LOS 角不变即可实现碰撞。
    当 dλ/dt ≠ 0 时，调整航向使其趋向零。
    实质是 PN 的极限情况 (N → ∞ 使 los_rate → 0)，
    这里用增益 K 的 P 控制器来消除 LOS rate。
    """

    def __init__(self, K: float = 8.0):
        super().__init__(name=f"ConstantBearing(K={K})")
        self.K = K

    def _compute_desired_direction(self, pursuer_pos, pursuer_vel, pursuer_yaw,
                                   evader_pos, evader_vel):
        rel_pos = evader_pos - pursuer_pos
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-6:
            return _unit_vec(pursuer_yaw)

        los_angle = np.arctan2(rel_pos[1], rel_pos[0])

        if self._prev_los_angle is not None:
            d_los = _angle_wrap(los_angle - self._prev_los_angle)
            los_rate = d_los / self._dt
        else:
            los_rate = 0.0
        self._prev_los_angle = los_angle

        # Aggressively cancel LOS rate
        correction = np.clip(self.K * los_rate * self._dt, -np.pi / 2, np.pi / 2)
        desired_heading = los_angle + correction

        return _unit_vec(desired_heading)


# ─────────────────────── 6. Deviated Pursuit (DP) ───────────────────────

class DeviatedPursuit(BaseStrategy):
    """
    偏置追踪 (Deviated Pursuit):
    在纯追踪方向上增加一个固定偏角 δ（朝逃脱者运动方向侧），
    实质是 PP 的带预判变体，较简单但对直线逃脱有效。
    δ > 0 表示向逃脱者运动方向偏转。
    """

    def __init__(self, deviation_deg: float = 20.0):
        super().__init__(name=f"DeviatedPursuit(δ={deviation_deg}°)")
        self.delta = np.deg2rad(deviation_deg)

    def _compute_desired_direction(self, pursuer_pos, pursuer_vel, pursuer_yaw,
                                   evader_pos, evader_vel):
        rel_pos = evader_pos - pursuer_pos
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-6:
            return _unit_vec(pursuer_yaw)

        los_angle = np.arctan2(rel_pos[1], rel_pos[0])

        # Determine which side to deviate: toward evader's velocity direction
        evader_speed = np.linalg.norm(evader_vel)
        if evader_speed > 0.05:
            evader_heading = np.arctan2(evader_vel[1], evader_vel[0])
            # Cross product sign to decide left/right deviation
            cross = np.cos(los_angle) * evader_vel[1] - np.sin(los_angle) * evader_vel[0]
            sign = 1.0 if cross > 0 else -1.0
        else:
            sign = 0.0  # target nearly stationary, no deviation

        desired_heading = los_angle + sign * self.delta
        return _unit_vec(desired_heading)


# ─────────────────────── Registry & Factory ───────────────────────

STRATEGY_REGISTRY = {
    "pp":   PurePursuit,
    "lead": LeadPursuit,
    "pn":   ProNav,
    "apn":  AugProNav,
    "cb":   ConstantBearing,
    "dp":   DeviatedPursuit,
}


def make_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    工厂函数: 根据名称创建策略实例。
    
    >>> s = make_strategy("pn", N=5)
    >>> s = make_strategy("lead", pursuer_max_speed=2.0)
    """
    key = name.lower().strip()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[key](**kwargs)


def list_strategies() -> list[str]:
    """Return available strategy names."""
    return list(STRATEGY_REGISTRY.keys())
