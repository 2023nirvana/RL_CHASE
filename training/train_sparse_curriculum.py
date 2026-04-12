"""
稀疏奖励 + 朝向 + 课程学习训练脚本
===================================

策略：
1. 奖励设计：纯稀疏（到达目标100分）+ 少量朝向奖励（权重10:1）
2. 课程学习：从近距离（1m）开始，逐步增加到目标距离（10m）
3. 成功阈值：连续成功率达标后升级难度

朝向奖励设计：
- 到达目标: +100 (基础)
- 朝向正确（<30度）: +0.1/step
- 朝向基本正确（<60度）: +0.05/step
- 朝向错误（>150度）: -0.05/step

权重比例：goal_reward : heading_reward ≈ 10:1
（假设平均500步到达目标，朝向奖励约为0.1*500=50，所以goal=100, heading*steps≈10）

使用方法：
    # 基础版本
    python training/train_sparse_curriculum.py --experiment-name sparse_cur_v1
    
    # 变体：更细粒度朝向奖励
    python training/train_sparse_curriculum.py --experiment-name sparse_cur_v2 --heading-bonus 0.05
    
    # 变体：更激进的课程学习
    python training/train_sparse_curriculum.py --experiment-name sparse_cur_v3 --curriculum-threshold 0.6
"""
import os, sys, argparse, yaml, time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, Dict

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env import LowLevelAUVEnv
from utils.visualizer import TrainingVisualizer
import gymnasium as gym
from gymnasium import spaces


# ==================== 稀疏+朝向奖励环境 ====================
class SparseCurriculumEnv(gym.Env):
    """
    稀疏奖励 + 朝向 + 课程学习环境
    
    奖励设计（10:1权重）：
    - 到达目标: +100
    - 朝向正确: +0.1/step（约500步*0.1=50，与goal_reward的10:1关系）
    """
    
    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right"]
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None,
                 fixed_z: float = 2.0,
                 target_distance: float = 1.0,
                 goal_reward: float = 100.0,
                 heading_bonus: float = 0.1,
                 heading_penalty: float = -0.05,
                 success_threshold: float = 0.3,
                 max_steps: int = 2000):
        
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")
        
        self.inner = LowLevelAUVEnv(config_path=config_path, render_mode=render_mode, seed=seed)
        
        self.fixed_z = fixed_z
        self.target_distance = target_distance
        self.goal_reward = goal_reward
        self.heading_bonus = heading_bonus
        self.heading_penalty = heading_penalty
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        
        self.action_map = {
            0: 1,  # forward
            1: 2,  # backward
            2: 3,  # left
            3: 4,  # right
            4: 7,  # yaw_left
            5: 8,  # yaw_right
        }
        
        self.action_space = spaces.Discrete(6)
        self.obs_dim = 8
        self.observation_space = spaces.Box(
            low=-10 * np.ones(self.obs_dim),
            high=10 * np.ones(self.obs_dim),
            dtype=np.float32
        )
        
        self.step_count = 0
        self.episode_count = 0
        self.success_count = 0
    
    def _enforce_2d(self):
        try:
            self.inner.dynamics.eta[2] = self.fixed_z
            self.inner.dynamics.eta[3:5] = 0.0
            self.inner.dynamics.nu[2:5] = 0.0
        except:
            pass
    
    def _sample_target(self):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(self.target_distance * 0.7, self.target_distance)
        
        init_pos = self.inner.dynamics.eta[:2]
        target_x = init_pos[0] + dist * np.cos(angle)
        target_y = init_pos[1] + dist * np.sin(angle)
        
        self.inner.target_state = np.zeros(12)
        self.inner.target_state[:2] = [target_x, target_y]
        self.inner.target_state[2] = self.fixed_z
    
    def _get_obs(self) -> np.ndarray:
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        pos_error = target[:2] - eta[:2]
        distance = np.linalg.norm(pos_error)
        target_angle = np.arctan2(pos_error[1], pos_error[0])
        
        yaw = eta[5]
        heading_error = np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw))
        
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        pos_error_body_x = cos_yaw * pos_error[0] + sin_yaw * pos_error[1]
        pos_error_body_y = -sin_yaw * pos_error[0] + cos_yaw * pos_error[1]
        
        obs = np.array([
            pos_error_body_x / 5.0,
            pos_error_body_y / 5.0,
            heading_error / np.pi,
            target_angle / np.pi,
            nu[0] / 2.0,
            nu[1] / 2.0,
            nu[5] / 2.0,
            distance / 5.0
        ], dtype=np.float32)
        
        return np.clip(obs, -10.0, 10.0)
    
    def _compute_reward(self) -> Tuple[float, Dict]:
        """
        稀疏 + 朝向奖励（10:1权重）
        """
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        
        pos_error = target[:2] - eta[:2]
        distance = np.linalg.norm(pos_error)
        
        target_angle = np.arctan2(pos_error[1], pos_error[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))
        
        reward_info = {'pos_error': distance, 'heading_error': heading_error}
        
        # 成功奖励
        success = distance < self.success_threshold
        goal_bonus = self.goal_reward if success else 0.0
        reward_info['success'] = success
        reward_info['goal'] = goal_bonus
        
        # 朝向奖励（稀疏，仅在特定条件下给予）
        heading_reward = 0.0
        if heading_error < np.pi / 6:  # <30度
            heading_reward = self.heading_bonus
        elif heading_error < np.pi / 3:  # <60度
            heading_reward = self.heading_bonus * 0.5
        elif heading_error > np.pi * 5 / 6:  # >150度
            heading_reward = self.heading_penalty
        
        reward_info['heading'] = heading_reward
        
        reward = goal_bonus + heading_reward
        return reward, reward_info
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.inner.reset()
        self._enforce_2d()
        self._sample_target()
        
        self.step_count = 0
        self.episode_count += 1
        
        return self._get_obs(), {'target_pos': self.inner.target_state[:2].tolist()}
    
    def step(self, action: int):
        inner_action = self.action_map[action]
        _, _, _, _, _ = self.inner.step(inner_action)
        self._enforce_2d()
        
        self.step_count += 1
        
        obs = self._get_obs()
        reward, reward_info = self._compute_reward()
        
        terminated = reward_info['success']
        truncated = self.step_count >= self.max_steps
        
        if terminated:
            self.success_count += 1
            termination_reason = 'success'
        elif truncated:
            termination_reason = 'timeout'
        else:
            termination_reason = None
        
        info = {
            'pos_error': reward_info['pos_error'],
            'heading_error': reward_info['heading_error'],
            'termination_reason': termination_reason,
            'target_pos': self.inner.target_state[:2].tolist()
        }
        
        return obs, reward, terminated, truncated, info
    
    def set_target_distance(self, distance: float):
        """课程学习：设置目标距离"""
        self.target_distance = distance


class VecSparseCurriculumEnv:
    """向量化环境"""
    
    def __init__(self, num_envs: int = 4, **kwargs):
        self.envs = [SparseCurriculumEnv(**kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed=None):
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed + i if seed else None)
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list
    
    def step(self, actions):
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, rew, term, trunc, info = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return np.stack(obs_list), np.array(rew_list), np.array(term_list), np.array(trunc_list), info_list
    
    def set_target_distance(self, distance: float):
        """课程学习：设置所有环境的目标距离"""
        for env in self.envs:
            env.set_target_distance(distance)


# ==================== 训练器 ====================
class SparseCurriculumTrainer:
    """稀疏奖励 + 课程学习训练器"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.experiment_name = args.experiment_name or f"sparse_cur_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.visualizer = TrainingVisualizer(save_dir=os.path.join(self.log_dir, "plots"))
        
        # 课程学习状态
        self.current_distance = args.initial_distance
        self.curriculum_success_window = deque(maxlen=args.curriculum_window)
        
        print(f"Device: {self.device}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Distance Curriculum: {args.initial_distance}m -> {args.max_distance}m")
        print(f"Heading Bonus: {args.heading_bonus} (ratio ≈ {args.goal_reward/args.heading_bonus/500:.1f}:1)")
        
        self._init_envs()
        self._init_network()
        
        self.success_count = 0
        self.episode_count = 0
    
    def _init_envs(self):
        tc = self.config['training']
        self.num_envs = tc.get('n_envs', 4)
        
        env_kwargs = {
            'target_distance': self.current_distance,
            'goal_reward': self.args.goal_reward,
            'heading_bonus': self.args.heading_bonus,
            'heading_penalty': self.args.heading_penalty,
            'success_threshold': self.args.success_threshold,
            'max_steps': self.args.max_steps
        }
        
        self.train_env = VecSparseCurriculumEnv(num_envs=self.num_envs, **env_kwargs)
        self.eval_env = SparseCurriculumEnv(**env_kwargs)
        
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        
        print(f"Env: obs={self.obs_dim}, actions={self.num_actions}")
        print(f"Current distance: {self.current_distance}m")
    
    def _init_network(self):
        hidden_sizes = [256, 128]
        
        actor_layers = []
        prev = self.obs_dim
        for h in hidden_sizes:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        actor_layers.append(nn.Linear(prev, self.num_actions))
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        
        critic_layers = []
        prev = self.obs_dim
        for h in hidden_sizes:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        
        tc = self.config['training']
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=tc.get('learning_rate', 3e-4)
        )
        
        self.gamma = tc.get('gamma', 0.99)
        self.gae_lambda = tc.get('gae_lambda', 0.95)
        self.clip_range = tc.get('clip_range', 0.2)
        self.ent_coef = tc.get('ent_coef', 0.05)
        self.vf_coef = tc.get('vf_coef', 0.5)
        self.max_grad_norm = tc.get('max_grad_norm', 0.5)
        self.n_steps = tc.get('n_steps', 1024)
        self.batch_size = tc.get('batch_size', 256)
        self.n_epochs = tc.get('n_epochs', 4)
        
        total_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        print(f"Network params: {total_params:,}")
    
    def _update_curriculum(self, success: bool):
        """更新课程学习状态"""
        self.curriculum_success_window.append(1 if success else 0)
        
        if len(self.curriculum_success_window) >= self.args.curriculum_window:
            success_rate = sum(self.curriculum_success_window) / len(self.curriculum_success_window)
            
            if success_rate >= self.args.curriculum_threshold and self.current_distance < self.args.max_distance:
                # 增加距离
                self.current_distance = min(
                    self.current_distance + self.args.distance_increment,
                    self.args.max_distance
                )
                self.curriculum_success_window.clear()
                
                # 更新环境
                self.train_env.set_target_distance(self.current_distance)
                self.eval_env.set_target_distance(self.current_distance)
                
                print(f"\n>>> Curriculum Update: Target distance = {self.current_distance:.1f}m (success rate: {success_rate:.1%})")
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits = self.actor(torch.tensor(obs, dtype=torch.float32, device=self.device))
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                return torch.argmax(probs, dim=-1).cpu().numpy()
            return torch.distributions.Categorical(probs).sample().cpu().numpy()
    
    def collect_rollouts(self):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
        obs, _ = self.train_env.reset()
        
        for _ in range(self.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.actor(obs_t)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
                values = self.critic(obs_t).squeeze(-1)
            
            next_obs, rewards, terms, truncs, infos = self.train_env.step(actions.cpu().numpy())
            dones = np.logical_or(terms, truncs)
            
            obs_buf.append(obs)
            act_buf.append(actions.cpu().numpy())
            rew_buf.append(rewards)
            done_buf.append(dones)
            val_buf.append(values.cpu().numpy())
            logp_buf.append(logprobs.cpu().numpy())
            
            for i, info in enumerate(infos):
                if info.get('termination_reason') == 'success':
                    self.success_count += 1
                    self._update_curriculum(True)
                elif dones[i]:
                    self._update_curriculum(False)
                if dones[i]:
                    self.episode_count += 1
            
            obs = next_obs
        
        with torch.no_grad():
            last_val = self.critic(torch.tensor(obs, dtype=torch.float32, device=self.device)).squeeze(-1).cpu().numpy()
        
        adv, ret = self._compute_gae(np.array(rew_buf), np.array(val_buf), np.array(done_buf), last_val)
        
        return {
            'obs': np.array(obs_buf),
            'actions': np.array(act_buf),
            'logprobs': np.array(logp_buf),
            'advantages': adv,
            'returns': ret,
            'mean_reward': np.mean(rew_buf)
        }
    
    def _compute_gae(self, rewards, values, dones, last_val):
        adv = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            nv = last_val if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            adv[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        return adv, adv + values
    
    def train_step(self, data):
        obs = torch.tensor(data['obs'].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(data['actions'].reshape(-1), dtype=torch.long, device=self.device)
        old_logp = torch.tensor(data['logprobs'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = torch.tensor(data['advantages'].reshape(-1), dtype=torch.float32, device=self.device)
        ret = torch.tensor(data['returns'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        total_actor_loss, total_value_loss, total_entropy, num_batches = 0, 0, 0, 0
        
        for _ in range(self.n_epochs):
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), self.batch_size):
                bi = idx[start:start + self.batch_size]
                logits = self.actor(obs[bi])
                dist = torch.distributions.Categorical(torch.softmax(logits, -1))
                new_logp = dist.log_prob(actions[bi])
                entropy = dist.entropy().mean()
                values = self.critic(obs[bi]).squeeze(-1)
                
                ratio = torch.exp(new_logp - old_logp[bi])
                actor_loss = -torch.min(ratio * adv[bi], torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv[bi]).mean()
                value_loss = nn.functional.mse_loss(values, ret[bi])
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1
        
        return {
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_value_loss / num_batches,
            'entropy': total_entropy / num_batches
        }
    
    def evaluate(self, n_episodes=10):
        rewards, successes = [], 0
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            ep_r, done = 0, False
            while not done:
                action = self.get_action(obs[np.newaxis, :], deterministic=True)[0]
                obs, r, term, trunc, info = self.eval_env.step(action)
                ep_r += r
                done = term or trunc
                if info.get('termination_reason') == 'success':
                    successes += 1
            rewards.append(ep_r)
        return {'mean_reward': np.mean(rewards), 'success_rate': successes / n_episodes}
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_distance': self.current_distance
        }, path)
    
    def train(self, total_timesteps=None):
        if total_timesteps is None:
            total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*70}")
        print(f"Sparse + Heading + Curriculum Training")
        print(f"Total: {total_timesteps:,}, Updates: {num_updates}")
        print(f"Reward: goal={self.args.goal_reward}, heading={self.args.heading_bonus}")
        print(f"Curriculum: {self.args.initial_distance}m -> {self.args.max_distance}m (step={self.args.distance_increment}m)")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        timesteps = 0
        best_sr = 0.0
        last_plot_step = 0
        
        for update in range(1, num_updates + 1):
            data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(data)
            
            # 记录指标
            self.visualizer.add_scalar('train/actor_loss', train_info['actor_loss'], timesteps)
            self.visualizer.add_scalar('train/critic_loss', train_info['critic_loss'], timesteps)
            self.visualizer.add_scalar('train/entropy', train_info['entropy'], timesteps)
            self.visualizer.add_scalar('rollout/mean_reward', data['mean_reward'], timesteps)
            self.visualizer.add_scalar('curriculum/target_distance', self.current_distance, timesteps)
            
            sr = self.success_count / max(1, self.episode_count)
            self.visualizer.add_scalar('train/success_rate', sr, timesteps)
            
            if timesteps % 5000 < steps_per_update:
                eval_info = self.evaluate(10)
                self.visualizer.add_scalar('eval/success_rate', eval_info['success_rate'], timesteps)
                self.visualizer.add_scalar('eval/mean_reward', eval_info['mean_reward'], timesteps)
                
                fps = timesteps / (time.time() - start_time)
                print(f"\nUpdate {update}/{num_updates} | Steps: {timesteps:,} | FPS: {fps:.0f}", flush=True)
                print(f"  Distance: {self.current_distance:.1f}m | Train SR: {sr:.1%}", flush=True)
                print(f"  Eval SR: {eval_info['success_rate']:.1%} | Reward: {eval_info['mean_reward']:.2f}", flush=True)
                print(f"  Actor Loss: {train_info['actor_loss']:.4f} | Entropy: {train_info['entropy']:.4f}", flush=True)
                
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {best_sr:.1%}", flush=True)
            
            if timesteps - last_plot_step >= 5000:
                self.visualizer.plot_training_curves(
                    metrics=['train/actor_loss', 'train/critic_loss', 'train/entropy',
                            'train/success_rate', 'eval/success_rate', 'curriculum/target_distance'],
                    save_name='training_curves.png'
                )
                last_plot_step = timesteps
            
            if timesteps % 20000 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        self.visualizer.plot_training_curves(
            metrics=['train/actor_loss', 'train/critic_loss', 'train/entropy',
                    'train/success_rate', 'eval/success_rate', 'curriculum/target_distance'],
            save_name='training_curves_final.png'
        )
        self.visualizer.save_data('training_data.json')
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        
        print(f"\n{'='*70}")
        print(f"Training done! Best SR: {best_sr:.1%}")
        print(f"Final distance: {self.current_distance:.1f}m")
        print(f"Plots saved to: {self.visualizer.save_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Sparse + Heading + Curriculum Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # 奖励参数
    parser.add_argument("--goal-reward", type=float, default=100.0,
                       help="Reward for reaching goal")
    parser.add_argument("--heading-bonus", type=float, default=0.1,
                       help="Bonus for correct heading (10:1 ratio with goal)")
    parser.add_argument("--heading-penalty", type=float, default=-0.05,
                       help="Penalty for wrong heading")
    
    # 课程学习参数
    parser.add_argument("--initial-distance", type=float, default=1.0,
                       help="Initial target distance")
    parser.add_argument("--max-distance", type=float, default=10.0,
                       help="Maximum target distance")
    parser.add_argument("--distance-increment", type=float, default=1.0,
                       help="Distance increment per curriculum stage")
    parser.add_argument("--curriculum-threshold", type=float, default=0.7,
                       help="Success rate threshold to advance curriculum")
    parser.add_argument("--curriculum-window", type=int, default=100,
                       help="Window size for computing success rate")
    
    # 环境参数
    parser.add_argument("--success-threshold", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=2000)
    
    # 训练参数
    parser.add_argument("--total-timesteps", type=int, default=3000000)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if args.total_timesteps:
        config['training']['total_timesteps'] = args.total_timesteps
    
    trainer = SparseCurriculumTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
