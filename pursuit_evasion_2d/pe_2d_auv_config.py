"""
2D 追逃用 AUV 动力学 - 从 robot_config 加载物理参数
====================================================

- 从 configs/robot_config.yaml 读取质量、阻尼、推力上限及 velocity_limits / pursuer / evader。
- 2D 平面取 surge/sway 对应项（X_udot, Y_vdot, X_u, Y_v, X_uu, Y_vv），
  与 thruster.max_thrust，构成与 core/dynamics/auv_dynamics 一致的 2D 切片。
- 速度与加速度上限作为硬约束（step 内 a_max、v_max 裁剪），
  并保证设计约束：追击者 v_max < 逃脱者 v_max，追击者 a_max > 逃脱者 a_max。
"""

from typing import Optional, Tuple
import os
import numpy as np
import yaml

from .pe_2d_dynamics import Dynamics2D


def _load_robot_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        # configs 在 UnderwaterPursuitEvasion 下
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(proj_root, "configs", "robot_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _params_2d_from_config(config: dict) -> Tuple[float, float, float, float]:
    """从完整配置中提取 2D 动力学参数：mass, linear_damping, quadratic_damping, max_thrust。"""
    robot = config["robot"]
    hydro = config["hydrodynamics"]
    thruster = config["thruster"]

    mass = float(robot["mass"])
    # 2D 平面取 x-y (surge-sway)：附加质量取 X_udot, Y_vdot 的负值（配置里为负）
    added_x = -float(hydro["added_mass"]["X_udot"])
    added_y = -float(hydro["added_mass"]["Y_vdot"])
    mass_eff = mass + 0.5 * (added_x + added_y)

    # 线性阻尼：配置为负，力 = -D*v，这里取绝对值作为 D_linear
    d_x = abs(float(hydro["linear_damping"]["X_u"]))
    d_y = abs(float(hydro["linear_damping"]["Y_v"]))
    linear_damping = 0.5 * (d_x + d_y)

    q_x = abs(float(hydro["quadratic_damping"]["X_uu"]))
    q_y = abs(float(hydro["quadratic_damping"]["Y_vv"]))
    quadratic_damping = 0.5 * (q_x + q_y)

    max_thrust = float(thruster["max_thrust"])
    return mass_eff, linear_damping, quadratic_damping, max_thrust


def create_pursuer_dynamics_from_config(
    config_path: Optional[str] = None,
    dt: float = 0.05,
    v_max_override: Optional[float] = None,
    a_max_override: Optional[float] = None,
) -> Dynamics2D:
    """
    从 robot_config 创建追击者 2D 动力学。
    设计约束：v_max 小于逃脱者、a_max 大于逃脱者；若 config 中 pursuer 已满足则直接使用，
    否则可用 v_max_override / a_max_override 覆盖（例如强制 pursuer_v_max=1.2, a_max=2.5）。
    """
    config = _load_robot_config(config_path)
    mass_eff, linear_damping, quadratic_damping, max_thrust = _params_2d_from_config(config)
    pursuer_cfg = config.get("pursuer", {})
    vel_limits = config.get("velocity_limits", {})

    v_max = v_max_override if v_max_override is not None else float(
        pursuer_cfg.get("max_speed", vel_limits.get("max_surge", 1.5))
    )
    a_max = a_max_override if a_max_override is not None else float(
        pursuer_cfg.get("max_acceleration", 2.0)
    )
    # 推力上限需能支撑 a_max：max_thrust >= mass_eff * a_max
    thrust_cap = mass_eff * a_max
    thrust = min(max_thrust, thrust_cap) if thrust_cap > 0 else max_thrust

    return Dynamics2D(
        mass=mass_eff,
        linear_damping=linear_damping,
        quadratic_damping=quadratic_damping,
        max_thrust=thrust,
        dt=dt,
        use_rk4=True,
    )


def create_evader_dynamics_from_config(
    config_path: Optional[str] = None,
    dt: float = 0.05,
    v_max_override: Optional[float] = None,
    a_max_override: Optional[float] = None,
) -> Dynamics2D:
    """
    从 robot_config 创建逃脱者 2D 动力学。
    设计约束：v_max 大于追击者、a_max 小于追击者。
    """
    config = _load_robot_config(config_path)
    mass_eff, linear_damping, quadratic_damping, max_thrust = _params_2d_from_config(config)
    evader_cfg = config.get("evader", {})
    vel_limits = config.get("velocity_limits", {})

    v_max = v_max_override if v_max_override is not None else float(
        evader_cfg.get("max_speed", vel_limits.get("max_surge", 2.5))
    )
    a_max = a_max_override if a_max_override is not None else float(
        evader_cfg.get("max_acceleration", 1.0)
    )
    thrust_cap = mass_eff * a_max
    thrust = min(max_thrust, thrust_cap) if thrust_cap > 0 else max_thrust

    return Dynamics2D(
        mass=mass_eff,
        linear_damping=linear_damping,
        quadratic_damping=quadratic_damping,
        max_thrust=thrust,
        dt=dt,
        use_rk4=True,
    )


def get_pursuer_evader_limits_from_config(
    config_path: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """返回 (pursuer_v_max, pursuer_a_max, evader_v_max, evader_a_max)。"""
    config = _load_robot_config(config_path)
    vel = config.get("velocity_limits", {})
    p = config.get("pursuer", {})
    e = config.get("evader", {})
    pursuer_v = float(p.get("max_speed", vel.get("max_surge", 1.5)))
    pursuer_a = float(p.get("max_acceleration", 2.0))
    evader_v = float(e.get("max_speed", vel.get("max_surge", 2.5)))
    evader_a = float(e.get("max_acceleration", 1.0))
    return pursuer_v, pursuer_a, evader_v, evader_a
