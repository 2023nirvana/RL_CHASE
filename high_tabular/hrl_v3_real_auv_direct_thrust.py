#!/usr/bin/env python3
"""
直接推力版追逃环境
==================
与 hrl_v3_real_auv_fixed.py 完全一致，但底层控制器替换为
v5 直接推力（8 推进器）版本。

使用方式与 PursuitEnvRealAUVFixed 完全兼容。
"""

import os, sys, math
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v3_direct_thrust import LowLevelAUV2DEnvV3DirectThrust
from high_tabular.hrl_v3_real_auv_fixed import PursuitEnvRealAUVFixed


# ==================== 速度钳位封装 ====================

class VelocityCappedDirectThrustEnv(LowLevelAUV2DEnvV3DirectThrust):
    """在直接推力环境基础上增加最大速度和最大加速度限制。

    在世界坐标系中钳位，确保：
    1. |v_world| <= v_max_pursuer
    2. |Δv_world/dt| <= a_max_pursuer
    """

    def __init__(self, v_max_pursuer: float = 0.65,
                 a_max_pursuer: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.v_max_pursuer = v_max_pursuer
        self.a_max_pursuer = a_max_pursuer
        self._prev_vx_w = 0.0
        self._prev_vy_w = 0.0

    def _body_to_world(self, vx_b, vy_b, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return vx_b * c - vy_b * s, vx_b * s + vy_b * c

    def _world_to_body(self, vx_w, vy_w, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return vx_w * c + vy_w * s, -vx_w * s + vy_w * c

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        nu = self.inner.dynamics.nu
        yaw = self.inner.dynamics.eta[5]
        self._prev_vx_w, self._prev_vy_w = self._body_to_world(nu[0], nu[1], yaw)
        return result

    def step(self, action):
        result = super().step(action)
        nu = self.inner.dynamics.nu
        eta = self.inner.dynamics.eta
        yaw = eta[5]
        dt = self.inner.dt

        # 当前世界坐标系速度
        vx_w, vy_w = self._body_to_world(nu[0], nu[1], yaw)

        # 1) 钳位加速度（世界坐标系）
        dvx = vx_w - self._prev_vx_w
        dvy = vy_w - self._prev_vy_w
        dv_mag = math.sqrt(dvx * dvx + dvy * dvy)
        max_dv = self.a_max_pursuer * dt
        if dv_mag > max_dv:
            scale = max_dv / dv_mag
            vx_w = self._prev_vx_w + dvx * scale
            vy_w = self._prev_vy_w + dvy * scale

        # 2) 钳位速度（世界坐标系）
        speed = math.sqrt(vx_w * vx_w + vy_w * vy_w)
        if speed > self.v_max_pursuer:
            scale = self.v_max_pursuer / speed
            vx_w *= scale
            vy_w *= scale

        # 写回 body frame
        nu[0], nu[1] = self._world_to_body(vx_w, vy_w, yaw)

        self._prev_vx_w = vx_w
        self._prev_vy_w = vy_w
        return result

    def sync_prev_velocity(self):
        """碰撞解算后调用，同步 _prev 为当前 nu 的世界坐标速度。"""
        nu = self.inner.dynamics.nu
        yaw = self.inner.dynamics.eta[5]
        self._prev_vx_w, self._prev_vy_w = self._body_to_world(
            nu[0], nu[1], yaw)


# ==================== v5 直接推力 NN 控制器加载 ====================

class DirectThrustActor(nn.Module):
    """与 v5 训练脚本中 GaussianActorCritic 的 actor 部分一致。"""
    def __init__(self, obs_dim=8, act_dim=8, hidden=(64, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.actor_body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)

    def forward(self, x):
        return torch.tanh(self.mu_head(self.actor_body(x)))


def load_v5_nn(checkpoint_path: str, device: torch.device):
    """加载 v5 直接推力模型。"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    obs_dim = ckpt.get("obs_dim", 8)
    act_dim = ckpt.get("act_dim", 8)

    config = ckpt.get("config", {})
    network_cfg = config.get("network", {})
    feature_cfg = network_cfg.get("feature_extractor", {})
    hidden = tuple(feature_cfg.get("hidden_sizes", [64, 64]))

    model = DirectThrustActor(obs_dim, act_dim, hidden)

    state_dict = ckpt["model"]
    model_dict = {}
    for k, v in state_dict.items():
        if k.startswith("actor_body") or k.startswith("mu_head"):
            model_dict[k] = v
    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded v5 direct thrust NN: obs={obs_dim}, act={act_dim}, hidden={hidden}")
    return model


# ==================== 追逃环境 (直接推力版) ====================

class PursuitEnvDirectThrust(PursuitEnvRealAUVFixed):
    """
    直接推力版追逃环境。

    与 PursuitEnvRealAUVFixed 完全一致，唯一区别：
    - 底层 AUV 环境替换为 LowLevelAUV2DEnvV3DirectThrust
    - 底层策略网络输出 8 维推进器动作（而非 3 维速度指令）
    
    高层接口（观测、动作、奖励等）完全不变。
    """

    def __init__(self, nn_policy, device,
                 world_size=30.0, catch_radius=1.0, max_steps=6000,
                 subgoal_range=0.5, low_steps=50, seed=42,
                 evader_mode="medium",
                 thrust_scale=40.0,
                 v_max_pursuer=0.65,
                 a_max_pursuer=0.5,
                 v_max_evader=None,
                 a_max_evader=None):
        # 先调用父类 __init__ 完成所有初始化
        super().__init__(
            nn_policy=nn_policy, device=device,
            world_size=world_size, catch_radius=catch_radius,
            max_steps=max_steps, subgoal_range=subgoal_range,
            low_steps=low_steps, seed=seed,
            evader_mode=evader_mode,
        )

        self.v_max_pursuer = v_max_pursuer
        self.a_max_pursuer = a_max_pursuer

        # 可选覆盖逃脱者运动学参数
        if v_max_evader is not None:
            self.v_max_e = v_max_evader
        if a_max_evader is not None:
            self.a_max_e = a_max_evader

        # 替换底层 AUV 环境为带速度/加速度钳位的直接推力版本
        self.auv_env.close()
        self.auv_env = VelocityCappedDirectThrustEnv(
            v_max_pursuer=v_max_pursuer,
            a_max_pursuer=a_max_pursuer,
            target_distance_max=subgoal_range,
            target_distance_min=0.1,
            reach_threshold=0.05,
            max_episode_seconds=60.0,
            thrust_scale=thrust_scale,
            init_velocity_range=0.0,
            seed=seed,
        )
