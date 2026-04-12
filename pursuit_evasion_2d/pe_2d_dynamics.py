"""
二维动力学模块 - 基于 low_level_env 的 AUV 动力学
==================================================

仿照 core/dynamics/auv_dynamics.py 的物理模型，提取为2D平面版本：
- 质量 + 附加质量
- 线性阻尼 D_l * v
- 二次阻尼 D_q * |v| * v
- 推力 τ 作为控制输入

动力学方程: M * a = τ - D_l * v - D_q * |v| * v

与 pe_v2 简化运动学的区别：
- 原 pe_v2: 直接 acc -> vel += acc*dt, pos += vel*dt (理想双积分器)
- 本模块: 推力 -> 加速度由动力学方程计算，含阻尼、惯性
"""

from typing import Tuple, Optional
import numpy as np


def _clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    """裁剪向量到最大范数"""
    n = np.linalg.norm(x)
    if n <= max_norm or max_norm <= 0:
        return x
    return x * (max_norm / n)


def _action_to_thrust_direction(action: int) -> np.ndarray:
    """
    离散动作 -> 2D 推力方向 (世界坐标系，单位向量)
    0: 零推力
    1~8: 8个方向，从0°（右/东）开始逆时针
    """
    if action == 0:
        return np.zeros(2, dtype=np.float32)
    angles = np.arange(8) * (2 * np.pi / 8)
    idx = action - 1
    return np.array([np.cos(angles[idx]), np.sin(angles[idx])], dtype=np.float32)


class Dynamics2D:
    """
    二维动力学模型 - 仿照 low_level AUV 动力学
    
    动力学: M * a = τ - D_linear * v - D_quadratic * |v| * v
    
    参数参考 robot_config.yaml / auv_dynamics.py:
    - mass: 15, added_mass ~ 3 -> M_eff ~ 18
    - linear_damping X_u: -5
    - quadratic_damping X_uu: -15
    """

    def __init__(
        self,
        mass: float = 18.0,           # 有效质量 (mass + added_mass)
        linear_damping: float = 5.0,  # D_linear (正值，力 = -D*v)
        quadratic_damping: float = 15.0,  # D_quadratic (力 = -D*|v|*v)
        max_thrust: float = 50.0,     # 最大推力 (N)，决定 a_max
        dt: float = 0.05,
        use_rk4: bool = True,         # 使用 RK4 积分
    ):
        self.mass = mass
        self.linear_damping = linear_damping
        self.quadratic_damping = quadratic_damping
        self.max_thrust = max_thrust
        self.dt = dt
        self.use_rk4 = use_rk4

        # 理论最大加速度 (无阻尼时): a_max = max_thrust / mass
        self.theoretical_a_max = max_thrust / mass

        # 状态
        self._pos = np.zeros(2, dtype=np.float32)
        self._vel = np.zeros(2, dtype=np.float32)
        self._acc = np.zeros(2, dtype=np.float32)

    def _damping_force(self, vel: np.ndarray) -> np.ndarray:
        """阻尼力: F_damp = -D_l * v - D_q * |v| * v"""
        v_norm = np.linalg.norm(vel)
        if v_norm < 1e-10:
            return np.zeros(2)
        F_linear = -self.linear_damping * vel
        F_quadratic = -self.quadratic_damping * v_norm * vel
        return F_linear + F_quadratic

    def compute_acceleration(
        self, pos: np.ndarray, vel: np.ndarray, thrust: np.ndarray
    ) -> np.ndarray:
        """
        计算加速度: a = (τ + F_damp) / M
        其中 F_damp = -D_l*v - D_q*|v|*v
        """
        F_damp = self._damping_force(vel)
        F_total = thrust + F_damp
        acc = F_total / self.mass
        return acc.astype(np.float32)

    def _euler_step(
        self, pos: np.ndarray, vel: np.ndarray, thrust: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """欧拉积分"""
        acc = self.compute_acceleration(pos, vel, thrust)
        vel_new = vel + acc * self.dt
        pos_new = pos + vel_new * self.dt
        return pos_new, vel_new

    def _rk4_step(
        self, pos: np.ndarray, vel: np.ndarray, thrust: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """4阶 Runge-Kutta 积分（与 auv_dynamics 一致）"""
        dt = self.dt

        def deriv(v):
            a = self.compute_acceleration(pos, v, thrust)
            return a

        k1 = deriv(vel)
        v2 = vel + 0.5 * dt * k1
        k2 = deriv(v2)
        v3 = vel + 0.5 * dt * k2
        k3 = deriv(v3)
        v4 = vel + dt * k3
        k4 = deriv(v4)

        acc_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        vel_new = vel + dt * acc_avg
        pos_new = pos + dt * (vel + vel_new) / 2.0  # 梯形积分位置

        return pos_new, vel_new

    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        thrust: np.ndarray,
        v_max: Optional[float] = None,
        a_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行一个仿真步。

        当提供 a_max 或需要严格满足 v_max 时，使用「硬约束」路径：
        先算加速度并裁剪到 a_max，再积分，最后将速度裁剪到 v_max，
        保证 ‖v‖ ≤ v_max 且 ‖a‖ ≤ a_max 在每步后成立。
        """
        if a_max is not None and a_max > 0:
            # 硬约束路径：加速度裁剪 -> 积分 -> 速度裁剪
            acc = self.compute_acceleration(pos, vel, thrust)
            acc_norm = np.linalg.norm(acc)
            if acc_norm > a_max:
                acc = acc * (a_max / acc_norm)
            vel_new = vel + acc * self.dt
            if v_max is not None and v_max > 0:
                v_norm = np.linalg.norm(vel_new)
                if v_norm > v_max:
                    vel_new = vel_new * (v_max / v_norm)
            pos_new = pos + 0.5 * (vel + vel_new) * self.dt
            return (
                pos_new.astype(np.float32),
                vel_new.astype(np.float32),
                acc.astype(np.float32),
            )
        # 原有路径：RK4/欧拉 + 可选 v_max 裁剪
        if self.use_rk4:
            pos_new, vel_new = self._rk4_step(pos, vel, thrust)
        else:
            pos_new, vel_new = self._euler_step(pos, vel, thrust)
        acc = self.compute_acceleration(pos, vel, thrust)
        if v_max is not None and v_max > 0:
            v_norm = np.linalg.norm(vel_new)
            if v_norm > v_max:
                vel_new = vel_new * (v_max / v_norm)
        return pos_new.astype(np.float32), vel_new.astype(np.float32), acc.astype(np.float32)

    def thrust_for_desired_acc(
        self, vel: np.ndarray, desired_acc: np.ndarray
    ) -> np.ndarray:
        """
        给定期望加速度，反推所需推力
        a = (thrust + F_damp) / m => thrust = m * a - F_damp
        """
        F_damp = self._damping_force(vel)
        thrust = self.mass * desired_acc - F_damp
        return _clip_norm(thrust, self.max_thrust).astype(np.float32)

    def reset(self, pos: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None):
        """重置状态"""
        self._pos = pos.copy() if pos is not None else np.zeros(2, dtype=np.float32)
        self._vel = vel.copy() if vel is not None else np.zeros(2, dtype=np.float32)
        self._acc = np.zeros(2, dtype=np.float32)


def create_pursuer_dynamics(
    v_max: float = 1.2,
    a_max: float = 2.5,
    dt: float = 0.05,
) -> Dynamics2D:
    """
    创建追击者动力学。
    物理设定：追击者 最大速度 < 逃脱者，最大加速度 > 逃脱者（速度劣势、加速度优势）。
    本函数 v_max=1.2, a_max=2.5；create_evader_dynamics 为 v_max=1.4, a_max=0.6。
    通过调节 mass 和 max_thrust 实现: a_max = max_thrust / mass。
    """
    mass = 15.0
    max_thrust = a_max * mass
    return Dynamics2D(
        mass=mass,
        linear_damping=4.0,
        quadratic_damping=12.0,
        max_thrust=max_thrust,
        dt=dt,
        use_rk4=True,
    )


def create_evader_dynamics(
    v_max: float = 1.4,
    a_max: float = 0.6,
    dt: float = 0.05,
) -> Dynamics2D:
    """
    创建逃脱者动力学
    逃脱者：速度略高，加速度低（低机动性）
    """
    mass = 20.0  # 质量更大，加速慢
    max_thrust = a_max * mass
    return Dynamics2D(
        mass=mass,
        linear_damping=6.0,
        quadratic_damping=18.0,
        max_thrust=max_thrust,
        dt=dt,
        use_rk4=True,
    )
