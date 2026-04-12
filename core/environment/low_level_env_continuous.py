"""
连续动作底层控制环境 - 真正的 RL 底层控制
==========================================

动作空间：Box(-1, 1, shape=(6,)) -> 6DOF 力/力矩
观测空间：14 维（位置误差、姿态、速度等）
"""
from typing import Optional, Tuple, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from core.dynamics.auv_dynamics import AUVDynamics


class LowLevelContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config_path=None, target_distance=2.0, max_force=20.0, 
                 max_torque=5.0, fixed_z=2.0, enable_3d=False, seed=None):
        super().__init__()
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs/robot_config.yaml")
        
        self.dynamics = AUVDynamics(config_path=config_path, randomize=False)
        self.dynamics.set_dt(0.02)  # 设置仿真时间步长 50Hz
        self.target_distance = target_distance
        self.max_force, self.max_torque = max_force, max_torque
        self.fixed_z, self.enable_3d = fixed_z, enable_3d
        
        # 连续动作空间：6DOF 力/力矩
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.obs_dim = 14
        self.observation_space = spaces.Box(low=-10*np.ones(self.obs_dim), high=10*np.ones(self.obs_dim), dtype=np.float32)
        
        self.target_pos = np.zeros(3)
        self.step_count = 0
        self.episode_count = 0
        self.success_count = 0
        self.prev_pos_error = None
        self.prev_action = np.zeros(6)
        if seed: np.random.seed(seed)

    def _sample_target(self):
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(self.target_distance*0.5, self.target_distance)
        self.target_pos[:2] = [dist*np.cos(angle), dist*np.sin(angle)]
        self.target_pos[2] = np.random.uniform(1, 4) if self.enable_3d else self.fixed_z

    def _enforce_constraints(self):
        if not self.enable_3d:
            self.dynamics.eta[2] = self.fixed_z
            self.dynamics.eta[3:5] = 0
            self.dynamics.nu[2:5] = 0

    def reset(self, seed=None, options=None):
        if seed: np.random.seed(seed)
        init_eta = np.array([0, 0, self.fixed_z, 0, 0, np.random.uniform(-np.pi, np.pi)])
        self.dynamics.reset(eta=init_eta, nu=np.zeros(6))
        self._sample_target()
        self._enforce_constraints()
        self.step_count = 0
        self.episode_count += 1
        self.prev_pos_error = None
        self.prev_action = np.zeros(6)
        return self._get_obs(), {'target_pos': self.target_pos.tolist()}

    def step(self, action):
        tau = np.zeros(6)
        tau[:3] = action[:3] * self.max_force
        tau[3:] = action[3:] * self.max_torque
        if not self.enable_3d:
            tau[2:5] = 0  # heave, roll, pitch
        
        # 直接应用力/力矩（核心：不使用PID，RL直接控制）
        eta_dot, nu_dot = self.dynamics.compute_derivatives(self.dynamics.eta, self.dynamics.nu, tau, None)
        dt = self.dynamics.dt
        self.dynamics.eta += dt * eta_dot
        self.dynamics.nu += dt * nu_dot
        self.dynamics.eta[3:6] = ((self.dynamics.eta[3:6] + np.pi) % (2*np.pi)) - np.pi
        self._enforce_constraints()
        
        self.step_count += 1
        obs = self._get_obs()
        reward, ri = self._compute_reward(action)
        term, trunc, ti = self._check_term()
        self.prev_action = action.copy()
        return obs, reward, term, trunc, {**ri, **ti}

    def _get_obs(self):
        eta, nu = self.dynamics.eta, self.dynamics.nu
        pe = self.target_pos - eta[:3]
        dist = np.linalg.norm(pe)
        ta = np.arctan2(pe[1], pe[0])
        c, s = np.cos(eta[5]), np.sin(eta[5])
        pe_body = [c*pe[0]+s*pe[1], -s*pe[0]+c*pe[1], pe[2]]
        he = np.arctan2(np.sin(ta-eta[5]), np.cos(ta-eta[5]))
        return np.clip(np.array([pe_body[0]/3, pe_body[1]/3, pe_body[2]/3,
                                  eta[3]/np.pi, eta[4]/np.pi, eta[5]/np.pi,
                                  nu[0]/2, nu[1]/2, nu[2]/2, nu[3]/2, nu[4]/2, nu[5]/2,
                                  dist/3, he/np.pi], dtype=np.float32), -10, 10)

    def _compute_reward(self, action):
        eta, nu = self.dynamics.eta, self.dynamics.nu
        pe = np.linalg.norm(eta[:3] - self.target_pos)
        
        r = -0.5*pe  # 位置惩罚
        if self.prev_pos_error: r += 15*(self.prev_pos_error - pe)  # 改进奖励
        self.prev_pos_error = pe
        r -= 0.01*np.linalg.norm(nu[:3])  # 速度惩罚
        r -= 0.1*(abs(eta[3])+abs(eta[4]))  # 姿态惩罚
        r -= 0.05*np.linalg.norm(action - self.prev_action)  # 平滑惩罚
        r -= 0.001*np.sum(action**2)  # 能耗惩罚
        if pe < 0.5: r += 5*(0.5-pe)  # 接近奖励
        success = pe < 0.3
        if success: r += 100  # 成功奖励
        r -= 0.02  # 时间惩罚
        return r, {'pos_error': pe, 'success': success}

    def _check_term(self):
        eta = self.dynamics.eta
        pe = np.linalg.norm(eta[:3] - self.target_pos)
        if pe < 0.3: self.success_count += 1; return True, False, {'termination_reason': 'success'}
        if abs(eta[0])>15 or abs(eta[1])>15 or eta[2]<0 or eta[2]>10: return True, False, {'termination_reason': 'out_of_bounds'}
        if abs(eta[3])>np.pi/2 or abs(eta[4])>np.pi/2: return True, False, {'termination_reason': 'flip'}
        if self.step_count >= 2000: return False, True, {'termination_reason': 'timeout'}
        return False, False, {'termination_reason': None, 'pos_error': pe}


class VecLowLevelContinuousEnv:
    def __init__(self, num_envs=4, **kw):
        self.envs = [LowLevelContinuousEnv(**kw) for _ in range(num_envs)]
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
