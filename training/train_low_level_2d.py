"""
2D 简化环境训练脚本
==================

目的：在最简单的 2D 场景下验证 PID+RL 架构能否收敛

使用方法：
    python training/train_low_level_2d.py
    
    # 指定目标距离
    python training/train_low_level_2d.py --target-distance 0.5
    
    # 更多步数
    python training/train_low_level_2d.py --total-timesteps 5000000
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional

# 强制输出刷新
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d import LowLevelAUV2DEnv, VecLowLevelAUV2DEnv


class Simple2DTrainer:
    """简化的 2D 训练器（使用简单的 PPO 实现）"""
    
    def __init__(self, config: Dict, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 创建实验目录
        self.experiment_name = args.experiment_name or f"2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 初始化环境
        self._init_environments()
        
        # 初始化网络和优化器
        self._init_network()
        
        # 统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        
    def _init_environments(self):
        """初始化 2D 环境"""
        training_config = self.config['training']
        self.num_envs = training_config.get('n_envs', 4)
        
        # 计算最小目标距离
        min_dist = self.args.min_target_distance
        
        # 向量化环境
        self.train_env = VecLowLevelAUV2DEnv(
            num_envs=self.num_envs,
            target_distance=self.args.target_distance,
            min_target_distance=min_dist,
            use_simplified_obs=True,
            enable_domain_randomization=False
        )
        
        # 单个评估环境
        self.eval_env = LowLevelAUV2DEnv(
            target_distance=self.args.target_distance,
            min_target_distance=min_dist,
            use_simplified_obs=True,
            enable_domain_randomization=False
        )
        
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        
        # 获取实际的最小距离
        actual_min_dist = self.eval_env.min_target_distance
        
        print(f"2D Environment initialized:")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.num_actions}")
        print(f"  Target distance: {actual_min_dist}m ~ {self.args.target_distance}m")
        print(f"  Success threshold: 0.15m")
        print(f"  Num envs: {self.num_envs}")
        
    def _init_network(self):
        """初始化简单的策略网络"""
        network_config = self.config.get('network', {})
        
        # 简单的 MLP 策略网络（不用 LSTM 先试试）
        hidden_sizes = network_config.get('feature_extractor', {}).get('hidden_sizes', [64, 64])
        
        # Actor 网络
        actor_layers = []
        prev_dim = self.obs_dim
        for hidden_size in hidden_sizes:
            actor_layers.append(torch.nn.Linear(prev_dim, hidden_size))
            actor_layers.append(torch.nn.ReLU())
            prev_dim = hidden_size
        actor_layers.append(torch.nn.Linear(prev_dim, self.num_actions))
        self.actor = torch.nn.Sequential(*actor_layers).to(self.device)
        
        # Critic 网络
        critic_layers = []
        prev_dim = self.obs_dim
        for hidden_size in hidden_sizes:
            critic_layers.append(torch.nn.Linear(prev_dim, hidden_size))
            critic_layers.append(torch.nn.ReLU())
            prev_dim = hidden_size
        critic_layers.append(torch.nn.Linear(prev_dim, 1))
        self.critic = torch.nn.Sequential(*critic_layers).to(self.device)
        
        # 优化器
        training_config = self.config['training']
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=training_config.get('learning_rate', 3e-4)
        )
        
        # PPO 参数
        self.gamma = training_config.get('gamma', 0.99)
        self.gae_lambda = training_config.get('gae_lambda', 0.95)
        self.clip_range = training_config.get('clip_range', 0.2)
        self.ent_coef = training_config.get('ent_coef', 0.05)
        self.vf_coef = training_config.get('vf_coef', 0.5)
        self.max_grad_norm = training_config.get('max_grad_norm', 0.5)
        self.n_steps = training_config.get('n_steps', 1024)
        self.batch_size = training_config.get('batch_size', 256)
        self.n_epochs = training_config.get('n_epochs', 4)
        
        total_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        print(f"Network parameters: {total_params:,}")
        
    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """获取动作"""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            logits = self.actor(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                actions = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                
            return actions.cpu().numpy()
    
    def collect_rollouts(self):
        """收集 rollout 数据"""
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []
        logprob_buffer = []
        
        obs, _ = self.train_env.reset()
        
        for step in range(self.n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                logits = self.actor(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                values = self.critic(obs_tensor).squeeze(-1)
            
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminateds, truncateds, infos = self.train_env.step(actions_np)
            dones = np.logical_or(terminateds, truncateds)
            
            obs_buffer.append(obs)
            action_buffer.append(actions_np)
            reward_buffer.append(rewards)
            done_buffer.append(dones)
            value_buffer.append(values.cpu().numpy())
            logprob_buffer.append(logprobs.cpu().numpy())
            
            # 统计
            for i, info in enumerate(infos):
                if info.get('termination_reason') == 'success':
                    self.success_count += 1
                if dones[i]:
                    self.episode_count += 1
            
            obs = next_obs
        
        # 计算最后一个状态的 value
        with torch.no_grad():
            last_values = self.critic(torch.tensor(obs, dtype=torch.float32, device=self.device)).squeeze(-1).cpu().numpy()
        
        # 计算 GAE
        advantages, returns = self._compute_gae(
            np.array(reward_buffer),
            np.array(value_buffer),
            np.array(done_buffer),
            last_values
        )
        
        return {
            'obs': np.array(obs_buffer),
            'actions': np.array(action_buffer),
            'logprobs': np.array(logprob_buffer),
            'advantages': advantages,
            'returns': returns,
            'mean_reward': np.mean(reward_buffer)
        }
    
    def _compute_gae(self, rewards, values, dones, last_values):
        """计算 GAE"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def train_step(self, rollout_data: Dict):
        """执行一次 PPO 更新"""
        obs = torch.tensor(rollout_data['obs'].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout_data['actions'].reshape(-1), dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(rollout_data['logprobs'].reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout_data['advantages'].reshape(-1), dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout_data['returns'].reshape(-1), dtype=torch.float32, device=self.device)
        
        # 归一化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for epoch in range(self.n_epochs):
            # 随机打乱
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                logits = self.actor(batch_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                values = self.critic(batch_obs).squeeze(-1)
                
                # PPO loss
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                
                # Total loss
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return {
            'loss': total_loss / (self.n_epochs * (len(obs) // self.batch_size + 1)),
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def evaluate(self, n_episodes: int = 10):
        """评估策略"""
        rewards = []
        successes = 0
        
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.get_action(obs[np.newaxis, :], deterministic=True)[0]
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                if info.get('termination_reason') == 'success':
                    successes += 1
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': successes / n_episodes
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def train(self, total_timesteps: int = None):
        """执行训练"""
        if total_timesteps is None:
            total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        
        training_config = self.config['training']
        eval_freq = training_config.get('eval_freq', 5000)
        save_freq = training_config.get('save_freq', 20000)
        
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*60}")
        print(f"Starting 2D training")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Steps per update: {steps_per_update:,}")
        print(f"Number of updates: {num_updates:,}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        timesteps = 0
        best_success_rate = 0.0
        
        for update in range(1, num_updates + 1):
            # 收集数据
            rollout_data = self.collect_rollouts()
            timesteps += steps_per_update
            
            # 训练
            train_info = self.train_step(rollout_data)
            
            # 评估
            if timesteps % eval_freq < steps_per_update:
                eval_info = self.evaluate(n_episodes=10)
                
                elapsed = time.time() - start_time
                fps = timesteps / elapsed
                success_rate = self.success_count / max(1, self.episode_count)
                
                print(f"\nUpdate {update}/{num_updates} | Timesteps: {timesteps:,}", flush=True)
                print(f"  FPS: {fps:.1f}", flush=True)
                print(f"  Mean Reward: {rollout_data['mean_reward']:.3f}", flush=True)
                print(f"  Eval Reward: {eval_info['mean_reward']:.3f}", flush=True)
                print(f"  Eval Success Rate: {eval_info['success_rate']:.1%}", flush=True)
                print(f"  Train Success Rate: {success_rate:.1%}", flush=True)
                print(f"  Loss: {train_info['loss']:.4f}", flush=True)
                print(f"  Entropy: {train_info['entropy']:.4f}", flush=True)
                
                # 保存最佳模型
                if eval_info['success_rate'] > best_success_rate:
                    best_success_rate = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best model! Success rate: {best_success_rate:.1%}")
            
            # 定期保存
            if timesteps % save_freq < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        # 保存最终模型
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\nTraining completed!")
        print(f"Best success rate: {best_success_rate:.1%}")
        print(f"Final model saved to {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="2D Simplified Environment Training")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--target-distance", type=float, default=1.0, help="Max target distance")
    parser.add_argument("--min-target-distance", type=float, default=None, help="Min target distance (default: target_distance * 0.5)")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载配置
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖总步数
    if args.total_timesteps is not None:
        config['training']['total_timesteps'] = args.total_timesteps
    
    # 创建训练器并训练
    trainer = Simple2DTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
