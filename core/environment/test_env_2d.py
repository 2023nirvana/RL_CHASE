"""
2D 环境测试脚本
===============

验证 2D 简化环境能正常工作

使用方法：
    cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion
    python core/environment/test_env_2d.py
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d import LowLevelAUV2DEnv


def test_basic_env():
    """测试基本环境功能"""
    print("="*60)
    print("Test 1: Basic Environment")
    print("="*60)
    
    env = LowLevelAUV2DEnv(target_distance=1.0, use_simplified_obs=True)
    
    print(f"Action space: {env.action_space}")
    print(f"Action names: {env.ACTION_NAMES}")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation dim: {env.obs_dim}")
    
    obs, info = env.reset()
    print(f"\nInitial observation: {obs}")
    print(f"Initial info: {info}")
    
    # 检查 2D 约束
    eta = env.inner.dynamics.eta
    print(f"\nAUV state (eta): x={eta[0]:.3f}, y={eta[1]:.3f}, z={eta[2]:.3f}")
    print(f"              : roll={eta[3]:.3f}, pitch={eta[4]:.3f}, yaw={eta[5]:.3f}")
    
    target = env.inner.target_state
    print(f"Target state: x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f}")
    
    env.close()
    print("\n[PASS] Basic environment test")


def test_actions():
    """测试各个动作"""
    print("\n" + "="*60)
    print("Test 2: Action Effects")
    print("="*60)
    
    env = LowLevelAUV2DEnv(target_distance=2.0, use_simplified_obs=True)
    
    for action, name in enumerate(env.ACTION_NAMES):
        obs, _ = env.reset()
        initial_pos = env.inner.dynamics.eta[:2].copy()
        
        # 执行同一动作 50 步
        total_reward = 0
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        final_pos = env.inner.dynamics.eta[:2]
        displacement = final_pos - initial_pos
        
        print(f"\nAction {action} ({name}):")
        print(f"  Displacement: dx={displacement[0]:.4f}, dy={displacement[1]:.4f}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Position error: {info.get('pos_error', 'N/A'):.3f}m")
    
    env.close()
    print("\n[PASS] Action test")


def test_episode():
    """测试完整 episode"""
    print("\n" + "="*60)
    print("Test 3: Full Episode")
    print("="*60)
    
    env = LowLevelAUV2DEnv(target_distance=0.5, use_simplified_obs=True)
    
    obs, info = env.reset()
    initial_distance = info['target_distance']
    print(f"Initial target distance: {initial_distance:.3f}m")
    
    total_reward = 0
    step = 0
    done = False
    
    while not done:
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated
        
        # 每 20 步打印一次
        if step % 20 == 0:
            print(f"  Step {step}: pos_error={info.get('pos_error', 0):.3f}m, reward={reward:.2f}")
    
    print(f"\nEpisode finished:")
    print(f"  Steps: {step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Termination reason: {info.get('termination_reason', 'unknown')}")
    print(f"  Final position error: {info.get('pos_error', 0):.3f}m")
    
    env.close()
    print("\n[PASS] Episode test")


def test_simple_policy():
    """测试简单启发式策略"""
    print("\n" + "="*60)
    print("Test 4: Heuristic Policy")
    print("="*60)
    
    env = LowLevelAUV2DEnv(target_distance=0.5, use_simplified_obs=True)
    
    successes = 0
    n_episodes = 10
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            # 简单启发式：基于位置误差选择动作
            # obs = [pos_error_body_x, pos_error_body_y, yaw_error, vx, vy, yaw_rate]
            ex, ey = obs[0], obs[1]  # 归一化后的误差
            
            # 选择最大误差方向的动作
            if abs(ex) > abs(ey):
                action = 0 if ex > 0 else 1  # forward/backward
            else:
                action = 2 if ey > 0 else 3  # left/right
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        success = info.get('termination_reason') == 'success'
        if success:
            successes += 1
        
        print(f"  Episode {ep+1}: steps={steps}, success={success}, pos_err={info.get('pos_error', 0):.3f}m")
    
    print(f"\nHeuristic policy success rate: {successes}/{n_episodes} = {successes/n_episodes:.1%}")
    
    env.close()
    print("\n[PASS] Heuristic policy test")


def test_reward_signal():
    """测试奖励信号是否合理"""
    print("\n" + "="*60)
    print("Test 5: Reward Signal")
    print("="*60)
    
    env = LowLevelAUV2DEnv(target_distance=0.5, use_simplified_obs=True)
    
    obs, info = env.reset()
    print(f"Initial distance: {info['target_distance']:.3f}m")
    
    # 测试朝向目标移动 vs 远离目标
    rewards_toward = []
    rewards_away = []
    
    for _ in range(20):
        env.reset()
        env.inner.target_state[0] = env.inner.dynamics.eta[0] + 0.5  # 目标在前方
        env.inner.target_state[1] = env.inner.dynamics.eta[1]
        env.prev_pos_error = 0.5
        
        # 向前（朝向目标）
        obs, reward, _, _, _ = env.step(0)  # forward
        rewards_toward.append(reward)
    
    env.reset()
    for _ in range(20):
        env.reset()
        env.inner.target_state[0] = env.inner.dynamics.eta[0] + 0.5
        env.inner.target_state[1] = env.inner.dynamics.eta[1]
        env.prev_pos_error = 0.5
        
        # 向后（远离目标）
        obs, reward, _, _, _ = env.step(1)  # backward
        rewards_away.append(reward)
    
    print(f"Reward moving toward target: {np.mean(rewards_toward):.3f} +/- {np.std(rewards_toward):.3f}")
    print(f"Reward moving away from target: {np.mean(rewards_away):.3f} +/- {np.std(rewards_away):.3f}")
    print(f"Difference: {np.mean(rewards_toward) - np.mean(rewards_away):.3f}")
    
    if np.mean(rewards_toward) > np.mean(rewards_away):
        print("[PASS] Reward signal correctly encourages moving toward target")
    else:
        print("[WARNING] Reward signal may not correctly guide learning")
    
    env.close()


if __name__ == "__main__":
    test_basic_env()
    test_actions()
    test_episode()
    test_simple_policy()
    test_reward_signal()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
