# 多维非对称追逃博弈 2D 模块
# 研究思路：利用多维度优势（机动性/地形）弥补速度劣势
# 追击者：v_max 小于逃脱者，a_max 大于逃脱者

from .env_asymmetric_pe import (
    PursuitEvasion2DAsymmetricEnv,
    DEFAULT_PURSUER_V_MAX,
    DEFAULT_PURSUER_A_MAX,
    DEFAULT_EVADER_V_MAX,
    DEFAULT_EVADER_A_MAX,
)
from .env_low_level_target import LowLevelTarget2DEnv
from .env_low_level_target_continuous import LowLevelTarget2DContinuousEnv

# 动力学版（基于 low_level_env 动力学，含加速度，不修改原 env_asymmetric_pe_v2）
from .pe_2d_dynamics import Dynamics2D, create_pursuer_dynamics, create_evader_dynamics
from .env_asymmetric_pe_v2_dynamics import (
    PursuitEvasion2DAsymmetricEnvV2Dynamics,
    make_easy_env as make_easy_env_dynamics,
    make_medium_env as make_medium_env_dynamics,
    make_hard_env as make_hard_env_dynamics,
    make_expert_env as make_expert_env_dynamics,
)
# 分层训练：阶段一追击者（动力学+加速度选项）+ 阶段二追逃策略
from .env_pursuer_low_level_dynamics import (
    PursuerLowLevelDynamicsEnv,
    NUM_PURSUER_ACCEL_ACTIONS,
    make_pursuer_env,
)
from .env_asymmetric_pe_v2_dynamics_accel import (
    PursuitEvasion2DAsymmetricEnvV2DynamicsAccel,
    make_easy_env as make_easy_env_accel,
    make_medium_env as make_medium_env_accel,
    make_hard_env as make_hard_env_accel,
    make_expert_env as make_expert_env_accel,
)
# RL 目标点 + 2D 双环 PID 算 τ（与 low_level 输出逻辑一致）
from .pe_2d_pid_controller import PIDController2D, action_to_target_position_2d
# AUV 配置动力学 + 双环 PID+RL（与 low_level 一致），硬约束 v_max/a_max，默认 5x5 小世界
from .pe_2d_auv_config import (
    create_pursuer_dynamics_from_config,
    create_evader_dynamics_from_config,
    get_pursuer_evader_limits_from_config,
)
from .pe_2d_dual_loop_pid import DualLoopPID2D
from .env_pe_2d_auv_pid import PursuitEvasion2DAUVPIDEnv
from .env_asymmetric_pe_v2_pid import (
    PursuitEvasion2DAsymmetricEnvV2PID,
    make_easy_env as make_easy_env_pid,
    make_medium_env as make_medium_env_pid,
    make_hard_env as make_hard_env_pid,
    make_expert_env as make_expert_env_pid,
)
from .env_asymmetric_pe_v2_pid_option_a import (
    PursuitEvasion2DAsymmetricEnvV2PIDOptionA,
)

__all__ = [
    "PursuitEvasion2DAsymmetricEnv",
    "LowLevelTarget2DEnv",
    "LowLevelTarget2DContinuousEnv",
    "DEFAULT_PURSUER_V_MAX",
    "DEFAULT_PURSUER_A_MAX",
    "DEFAULT_EVADER_V_MAX",
    "DEFAULT_EVADER_A_MAX",
    # 动力学版
    "Dynamics2D",
    "create_pursuer_dynamics",
    "create_evader_dynamics",
    "PursuitEvasion2DAsymmetricEnvV2Dynamics",
    "make_easy_env_dynamics",
    "make_medium_env_dynamics",
    "make_hard_env_dynamics",
    "make_expert_env_dynamics",
    # RL 目标 + PID 版
    "PIDController2D",
    "action_to_target_position_2d",
    "PursuitEvasion2DAsymmetricEnvV2PID",
    "make_easy_env_pid",
    "make_medium_env_pid",
    "make_hard_env_pid",
    "make_expert_env_pid",
    "PursuitEvasion2DAsymmetricEnvV2PIDOptionA",
    # 分层训练（加速度选项）
    "PursuerLowLevelDynamicsEnv",
    "NUM_PURSUER_ACCEL_ACTIONS",
    "make_pursuer_env",
    "PursuitEvasion2DAsymmetricEnvV2DynamicsAccel",
    "make_easy_env_accel",
    "make_medium_env_accel",
    "make_hard_env_accel",
    "make_expert_env_accel",
    # AUV 配置 + 双环 PID（与 low_level 一致），硬约束，默认 5x5
    "create_pursuer_dynamics_from_config",
    "create_evader_dynamics_from_config",
    "get_pursuer_evader_limits_from_config",
    "DualLoopPID2D",
    "PursuitEvasion2DAUVPIDEnv",
]
