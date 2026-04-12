"""
2D V2 环境训练脚本 - 6 动作（含转向）
=====================================

Stage 1: 增加 yaw_left/yaw_right 转向动作

使用方法：
    python training/train_low_level_2d_v2.py --target-distance 2.0 --experiment-name 2d_v2_dist_2m
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

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_v2 import LowLevelAUV2DEnvV2, VecLowLevelAUV2DEnvV2


class Simple2DTrainerV2:
    """2D V2 训练器（6 动作）"""
    
    def __init__(self, config: Dict, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.experiment_name = args.experiment_name or f"2d_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self._init_environments()
        self._init_network()
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        
    def _init_environments(self):
        """初始化 2D V2 环境"""
        training_config = self.config['training']
        self.num_envs = training_config.get('n_envs', 4)
        
        self.train_env = VecLowLevelAUV2DEnvV2(
            num_envs=self.num_envs,
            target_distance=self.args.target_distance,
            enable_domain_randomization=False
        )
        
        self.eval_env = LowLevelAUV2DEnvV2(
            target_distance=self.args.target_distance,
            enable_domain_randomization=False
        )
        
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        
        print(f"2D V2 Environment initialized:")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.num_actions} (with yaw control)")
        print(f"  Actions: {LowLevelAUV2DEnvV2.ACTION_NAMES}")
        print(f"  Target distance: {self.args.target_distance}m")
        print(f"  Num envs: {self.num_envs}")
        
    def _init_network(self):
        """初始化策略网络 - 稍微增大网络"""
        network_config = self.config.get('network', {})
        
        # 稍微增大网络以处理更复杂的任务
        hidden_sizes = network_config.get('feature_extractor', {}).get('hidden_sizes', [128, 64])
        
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
        
        training_config = self.config['training']
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=training_config.get('learning_rate', 3e-4)
        )
        
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
            
            for i, info in enumerate(infos):
                if info.get('termination_reason') == 'success':
                    self.success_count += 1
                if dones[i]:
                    self.episode_count += 1
            
            obs = next_obs
        
        with torch.no_grad():
            last_values = self.critic(torch.tensor(obs, dtype=torch.float32, device=self.device)).squeeze(-1).cpu().numpy()
        
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
        obs = torch.tensor(rollout_data['obs'].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout_data['actions'].reshape(-1), dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(rollout_data['logprobs'].reshape(-1), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout_data['advantages'].reshape(-1), dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout_data['returns'].reshape(-1), dtype=torch.float32, device=self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for epoch in range(self.n_epochs):
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                logits = self.actor(batch_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                values = self.critic(batch_obs).squeeze(-1)
                
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
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
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def train(self, total_timesteps: int = None):
        if total_timesteps is None:
            total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        
        training_config = self.config['training']
        eval_freq = training_config.get('eval_freq', 5000)
        save_freq = training_config.get('save_freq', 20000)
        
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*60}")
        print(f"Starting 2D V2 training (6 actions with yaw)")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Steps per update: {steps_per_update:,}")
        print(f"Number of updates: {num_updates:,}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        timesteps = 0
        best_success_rate = 0.0
        
        for update in range(1, num_updates + 1):
            rollout_data = self.collect_rollouts()
            timesteps += steps_per_update
            
            train_info = self.train_step(rollout_data)
            
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
                
                if eval_info['success_rate'] > best_success_rate:
                    best_success_rate = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best model! Success rate: {best_success_rate:.1%}", flush=True)
            
            if timesteps % save_freq < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\nTraining completed!")
        print(f"Best success rate: {best_success_rate:.1%}")
        print(f"Final model saved to {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="2D V2 Environment Training (6 actions)")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--target-distance", type=float, default=2.0, help="Target distance")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # V2 默认使用更大的网络
    if 'network' not in config:
        config['network'] = {}
    config['network']['feature_extractor'] = {'hidden_sizes': [128, 64]}
    
    if args.total_timesteps is not None:
        config['training']['total_timesteps'] = args.total_timesteps
    
    trainer = Simple2DTrainerV2(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
