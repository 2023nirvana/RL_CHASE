"""
二维动态目标环境 - 6个离散动作
================================
方案B：目标会移动
"""
from typing import Optional, Tuple, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from core.environment.low_level_env import LowLevelAUVEnv


class LowLevelAUV2DDynamicEnv(gym.Env):
    """
    二维动态目标环境 - 6个离散动作
    
    改进：
    1. 目标到达边界时"反弹"而不是停止
    2. 可配置追击者速度优势
    3. 支持多种目标运动模式
    """
    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right"]
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path=None, render_mode=None, enable_domain_randomization=False,
                 seed=None, fixed_z=2.0, target_distance=3.0, target_speed=0.3, target_motion='constant',
                 boundary_behavior='stop', boundary_limit=15.0):
        """
        Args:
            target_speed: 目标移动速度 (m/s)，默认0.3
            target_motion: 目标运动模式 ('constant', 'random_walk', 'circular', 'evasive')
            boundary_behavior: 边界行为 ('bounce'=反弹, 'stop'=停止, 'wrap'=环绕)
            boundary_limit: 边界范围 (m)
        """
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")
        
        self.inner = LowLevelAUVEnv(config_path=config_path, render_mode=render_mode,
                                     enable_domain_randomization=enable_domain_randomization, seed=seed)
        self.fixed_z, self.target_distance = fixed_z, target_distance
        self.target_speed, self.target_motion = target_speed, target_motion
        self.boundary_behavior = boundary_behavior
        self.boundary_limit = boundary_limit
        self.target_pos, self.target_vel = np.zeros(2), np.zeros(2)
        self.target_angle, self.circle_center, self.circle_radius = 0.0, np.zeros(2), 1.0
        
        self.action_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8}
        self.action_space = spaces.Discrete(6)
        self.obs_dim = 12
        self.observation_space = spaces.Box(low=-10*np.ones(self.obs_dim), high=10*np.ones(self.obs_dim), dtype=np.float32)
        self.episode_count, self.success_count, self.prev_pos_error, self.dt = 0, 0, None, 0.05
        self.step_count = 0

    def _enforce_2d(self):
        try:
            self.inner.dynamics.eta[2], self.inner.dynamics.eta[3:5] = self.fixed_z, 0.0
            self.inner.dynamics.nu[2:5] = 0.0
        except: pass

    def _sample_target(self):
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(self.target_distance*0.5, self.target_distance)
        self.target_pos = dist * np.array([np.cos(angle), np.sin(angle)])
        if self.target_motion == 'constant':
            va = np.random.uniform(0, 2*np.pi)
            self.target_vel = self.target_speed * np.array([np.cos(va), np.sin(va)])
        elif self.target_motion == 'circular':
            self.target_angle, self.circle_center = np.random.uniform(0, 2*np.pi), self.target_pos.copy()
        else: self.target_vel = np.zeros(2)

    def _update_target(self):
        """更新目标位置，支持边界反弹/停止/环绕"""
        if self.target_motion == 'constant':
            self.target_pos += self.target_vel * self.dt
        elif self.target_motion == 'random_walk':
            self.target_vel = np.clip(self.target_vel + np.random.randn(2)*0.1, -self.target_speed, self.target_speed)
            self.target_pos += self.target_vel * self.dt
        elif self.target_motion == 'circular':
            w = self.target_speed / self.circle_radius
            self.target_angle += w * self.dt
            self.target_pos = self.circle_center + self.circle_radius * np.array([np.cos(self.target_angle), np.sin(self.target_angle)])
            self.target_vel = self.target_speed * np.array([-np.sin(self.target_angle), np.cos(self.target_angle)])
        elif self.target_motion == 'evasive':
            d = self.target_pos - self.inner.dynamics.eta[:2]
            dn = np.linalg.norm(d)
            if dn > 0.1: self.target_vel = self.target_speed * min(1, 2/(dn+0.5)) * d/dn
            self.target_pos += self.target_vel * self.dt
        
        # 边界处理
        self._handle_boundary()
        self.inner.target_state[:2] = self.target_pos
    
    def _handle_boundary(self):
        """处理目标到达边界的行为"""
        limit = self.boundary_limit
        
        if self.boundary_behavior == 'bounce':
            # 反弹：碰到边界时速度反向
            for i in range(2):
                if self.target_pos[i] > limit:
                    self.target_pos[i] = limit
                    self.target_vel[i] = -abs(self.target_vel[i])  # 反弹
                elif self.target_pos[i] < -limit:
                    self.target_pos[i] = -limit
                    self.target_vel[i] = abs(self.target_vel[i])  # 反弹
                    
        elif self.boundary_behavior == 'wrap':
            # 环绕：从一边消失，从另一边出现
            for i in range(2):
                if self.target_pos[i] > limit:
                    self.target_pos[i] = -limit + (self.target_pos[i] - limit)
                elif self.target_pos[i] < -limit:
                    self.target_pos[i] = limit + (self.target_pos[i] + limit)
                    
        else:  # 'stop'
            # 停止：碰到边界就停下（原行为，但明确设置速度为0）
            for i in range(2):
                if abs(self.target_pos[i]) >= limit:
                    self.target_pos[i] = np.clip(self.target_pos[i], -limit, limit)
                    self.target_vel[i] = 0

    def reset(self, seed=None, options=None):
        self.inner.reset(seed=seed, options=options)
        self.inner.dynamics.eta[:3] = [0, 0, self.fixed_z]
        self.inner.dynamics.eta[3:] = [0, 0, np.random.uniform(-np.pi, np.pi)]
        self.inner.dynamics.nu[:] = 0
        self._sample_target()
        self.inner.target_state = np.zeros(6)
        self.inner.target_state[:2] = self.target_pos
        self._enforce_2d()
        self.episode_count += 1
        self.prev_pos_error = None
        self.step_count = 0
        return self._get_obs(), {'target_pos': self.target_pos.tolist(), 'target_vel': self.target_vel.tolist()}

    def step(self, action):
        self._update_target()
        self.inner.step(self.action_map[action])
        self._enforce_2d()
        self.step_count += 1
        obs = self._get_obs()
        reward, ri = self._compute_reward()
        term, trunc, ti = self._check_term()
        return obs, reward, term, trunc, {**ri, **ti, 'target_pos': self.target_pos.tolist()}

    def _get_obs(self):
        eta, nu = self.inner.dynamics.eta, self.inner.dynamics.nu
        pe = self.target_pos - eta[:2]
        d = np.linalg.norm(pe)
        ta = np.arctan2(pe[1], pe[0])
        he = np.arctan2(np.sin(ta-eta[5]), np.cos(ta-eta[5]))
        c, s = np.cos(eta[5]), np.sin(eta[5])
        return np.clip(np.array([c*pe[0]+s*pe[1], -s*pe[0]+c*pe[1], he, ta, nu[0], nu[1], nu[5], d,
                                  self.target_vel[0], self.target_vel[1], self.target_vel[0]-nu[0], self.target_vel[1]-nu[1]], 
                                 dtype=np.float32) / np.array([5,5,np.pi,np.pi,2,2,2,5,1,1,2,2]), -10, 10)

    def _compute_reward(self):
        eta, nu = self.inner.dynamics.eta, self.inner.dynamics.nu
        pe = np.linalg.norm(eta[:2] - self.target_pos)
        r = -0.3*pe + (10*(self.prev_pos_error-pe) if self.prev_pos_error else 0) - 0.02
        self.prev_pos_error = pe
        d = self.target_pos - eta[:2]
        he = abs(np.arctan2(np.sin(np.arctan2(d[1],d[0])-eta[5]), np.cos(np.arctan2(d[1],d[0])-eta[5])))
        r += 0.3*(1-he/np.pi) + (5*(0.5-pe) if pe<0.5 else 0) + (100 if pe<0.3 else 0)
        return r, {'pos_error': pe, 'success': pe<0.3}

    def _check_term(self):
        eta = self.inner.dynamics.eta
        pe = np.linalg.norm(eta[:2] - self.target_pos)
        if pe < 0.3: self.success_count += 1; return True, False, {'termination_reason': 'success'}
        if abs(eta[0]) > 20 or abs(eta[1]) > 20: return True, False, {'termination_reason': 'out_of_bounds'}
        if self.step_count >= 3000: return False, True, {'termination_reason': 'timeout'}
        return False, False, {'termination_reason': None, 'pos_error': pe}


class VecLowLevelAUV2DDynamicEnv:
    def __init__(self, num_envs=4, **kw):
        self.envs = [LowLevelAUV2DDynamicEnv(**kw) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.observation_space, self.action_space = self.envs[0].observation_space, self.envs[0].action_space
    
    def reset(self, seed=None):
        r = [e.reset(seed=seed+i if seed else None) for i,e in enumerate(self.envs)]
        return np.array([x[0] for x in r]), [x[1] for x in r]
    
    def step(self, actions):
        r = []
        for e, a in zip(self.envs, actions):
            o, rw, t, tr, i = e.step(a)
            if t or tr: o, _ = e.reset()
            r.append((o, rw, t, tr, i))
        return np.array([x[0] for x in r]), np.array([x[1] for x in r]), np.array([x[2] for x in r]), np.array([x[3] for x in r]), [x[4] for x in r]
