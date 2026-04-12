"""
高级2D训练脚本 - 支持多种改进方案
=====================================

支持的训练模式：
1. sparse_heading  - 改进版稀疏奖励（稀疏+朝向）
2. curriculum      - 课程学习（从近到远）
3. obstacle        - 避障训练

使用方法：
    # 改进版稀疏奖励训练
    python training/train_2d_advanced.py --mode sparse_heading --experiment-name sparse_heading_5m --target-distance 5.0
    
    # 课程学习训练
    python training/train_2d_advanced.py --mode curriculum --experiment-name curriculum_1to10m
    
    # 避障训练
    python training/train_2d_advanced.py --mode obstacle --experiment-name obstacle_5m --num-obstacles 5
"""
import os, sys, argparse, yaml, time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_simple_reward import (
    LowLevelAUV2DSimpleRewardEnv, 
    VecLowLevelAUV2DSimpleRewardEnv,
    CurriculumLowLevelAUV2DEnv
)
from core.environment.low_level_env_2d_obstacle import (
    LowLevelAUV2DObstacleEnv,
    VecLowLevelAUV2DObstacleEnv
)


class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.mode = args.mode
        self.experiment_name = args.experiment_name or f"{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Mode: {self.mode}")
        print(f"Experiment: {self.experiment_name}")
        
        self._init_envs()
        self._init_network()
        self.success_count = 0
        self.episode_count = 0

    def _init_envs(self):
        tc = self.config['training']
        self.num_envs = tc.get('n_envs', 4)
        
        if self.mode == 'sparse_heading':
            # 改进版稀疏奖励
            env_kwargs = {
                'reward_type': 'sparse_heading',
                'target_distance': self.args.target_distance,
                'goal_reward': self.args.goal_reward,
                'success_threshold': self.args.success_threshold,
                'max_steps': self.args.max_steps
            }
            self.train_env = VecLowLevelAUV2DSimpleRewardEnv(num_envs=self.num_envs, **env_kwargs)
            self.eval_env = LowLevelAUV2DSimpleRewardEnv(**env_kwargs)
            
        elif self.mode == 'curriculum':
            # 课程学习（不使用向量化，因为需要统一追踪进度）
            self.train_env = CurriculumWrapper(
                num_envs=self.num_envs,
                initial_distance=self.args.initial_distance,
                max_distance=self.args.max_distance,
                distance_increment=self.args.distance_increment,
                reward_type=self.args.reward_type,
                goal_reward=self.args.goal_reward,
                success_threshold=self.args.success_threshold,
                max_steps=self.args.max_steps
            )
            self.eval_env = LowLevelAUV2DSimpleRewardEnv(
                reward_type=self.args.reward_type,
                target_distance=self.args.max_distance,  # 评估用最大距离
                goal_reward=self.args.goal_reward,
                success_threshold=self.args.success_threshold,
                max_steps=self.args.max_steps
            )
            
        elif self.mode == 'obstacle':
            # 避障训练（支持课程学习）
            env_kwargs = {
                'target_distance': self.args.target_distance,
                'num_obstacles': self.args.num_obstacles,
                'goal_reward': self.args.goal_reward,
                'collision_penalty': self.args.collision_penalty,
                'obstacle_avoidance_coef': self.args.obstacle_avoidance_coef,
                'success_threshold': self.args.success_threshold,
                'max_steps': self.args.max_steps
            }
            
            if self.args.obstacle_curriculum:
                # 课程学习：从0个障碍物开始
                self.train_env = ObstacleCurriculumWrapper(
                    num_envs=self.num_envs,
                    initial_num_obstacles=0,
                    max_num_obstacles=self.args.num_obstacles,
                    obstacle_increment=1,
                    success_threshold_to_advance=self.args.obstacle_curriculum_threshold,
                    window_size=100,
                    **env_kwargs
                )
            else:
                # 直接使用固定障碍物数量
                self.train_env = VecLowLevelAUV2DObstacleEnv(num_envs=self.num_envs, **env_kwargs)
            
            self.eval_env = LowLevelAUV2DObstacleEnv(**env_kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        print(f"Env: obs={self.obs_dim}, actions={self.num_actions}")

    def _init_network(self):
        hidden_sizes = [256, 128]
        
        # Actor
        actor_layers = []
        prev = self.obs_dim
        for h in hidden_sizes:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        actor_layers.append(nn.Linear(prev, self.num_actions))
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        
        # Critic
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
        
        # 加载预训练模型（如果指定）
        if self.args.pretrain_checkpoint:
            self._load_pretrain(self.args.pretrain_checkpoint)
    
    def _load_pretrain(self, checkpoint_path: str):
        """加载预训练模型权重"""
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Pretrain checkpoint not found: {checkpoint_path}")
            return
        
        print(f"\n>>> Loading pretrain model from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # 尝试加载actor和critic
        loaded_actor = False
        loaded_critic = False
        
        if 'actor' in ckpt:
            try:
                # 检查维度是否匹配
                pretrain_actor = ckpt['actor']
                if isinstance(pretrain_actor, dict):
                    # 检查第一层输入维度是否匹配
                    first_key = list(pretrain_actor.keys())[0]
                    if 'weight' in first_key:
                        pretrain_input_dim = pretrain_actor[first_key].shape[1]
                        if pretrain_input_dim == self.obs_dim:
                            self.actor.load_state_dict(pretrain_actor, strict=False)
                            loaded_actor = True
                            print(f"  ✓ Loaded actor weights (input dim: {pretrain_input_dim})")
                        else:
                            print(f"  ✗ Actor input dim mismatch: pretrain={pretrain_input_dim}, current={self.obs_dim}")
                            print(f"     Will train from scratch for actor")
                else:
                    self.actor.load_state_dict(pretrain_actor, strict=False)
                    loaded_actor = True
                    print(f"  ✓ Loaded actor weights")
            except Exception as e:
                print(f"  ✗ Failed to load actor: {e}")
        
        if 'critic' in ckpt:
            try:
                pretrain_critic = ckpt['critic']
                if isinstance(pretrain_critic, dict):
                    first_key = list(pretrain_critic.keys())[0]
                    if 'weight' in first_key:
                        pretrain_input_dim = pretrain_critic[first_key].shape[1]
                        if pretrain_input_dim == self.obs_dim:
                            self.critic.load_state_dict(pretrain_critic, strict=False)
                            loaded_critic = True
                            print(f"  ✓ Loaded critic weights (input dim: {pretrain_input_dim})")
                        else:
                            print(f"  ✗ Critic input dim mismatch: pretrain={pretrain_input_dim}, current={self.obs_dim}")
                            print(f"     Will train from scratch for critic")
                else:
                    self.critic.load_state_dict(pretrain_critic, strict=False)
                    loaded_critic = True
                    print(f"  ✓ Loaded critic weights")
            except Exception as e:
                print(f"  ✗ Failed to load critic: {e}")
        
        # 加载optimizer状态（如果存在且维度匹配）
        if 'optimizer' in ckpt and loaded_actor and loaded_critic:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                print(f"  ✓ Loaded optimizer state")
            except Exception as e:
                print(f"  ✗ Failed to load optimizer: {e}")
        
        if not loaded_actor and not loaded_critic:
            print(f"  ✗ No compatible weights found, training from scratch")
        else:
            print(f"  → Fine-tuning from pretrain model")

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
            next_val = last_val if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            adv[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        return adv, adv + values

    def train_step(self, data):
        obs = torch.tensor(data['obs'].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(data['actions'].reshape(-1), dtype=torch.long, device=self.device)
        old_logp = torch.tensor(data['logprobs'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = torch.tensor(data['advantages'].reshape(-1), dtype=torch.float32, device=self.device)
        ret = torch.tensor(data['returns'].reshape(-1), dtype=torch.float32, device=self.device)
        
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
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
                actor_loss = -torch.min(
                    ratio * adv[bi],
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv[bi]
                ).mean()
                
                value_loss = nn.functional.mse_loss(values, ret[bi])
                
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
        
        return {'loss': loss.item(), 'entropy': entropy.item()}

    def evaluate(self, n_episodes=10):
        rewards, successes, collisions = [], 0, 0
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
                if info.get('termination_reason') == 'collision':
                    collisions += 1
            rewards.append(ep_r)
        
        result = {'mean_reward': np.mean(rewards), 'success_rate': successes / n_episodes}
        if self.mode == 'obstacle':
            result['collision_rate'] = collisions / n_episodes
        return result

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def train(self, total_timesteps=None):
        if total_timesteps is None:
            total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*60}")
        print(f"Advanced Training: {self.mode}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        timesteps = 0
        best_sr = 0.0
        
        for update in range(1, num_updates + 1):
            data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(data)
            
            if timesteps % 10000 < steps_per_update:
                eval_info = self.evaluate(10)
                train_sr = self.success_count / max(1, self.episode_count)
                fps = timesteps / (time.time() - start_time)
                
                print(f"\nUpdate {update}/{num_updates} | Steps: {timesteps:,} | FPS: {fps:.0f}", flush=True)
                print(f"  Eval SR: {eval_info['success_rate']:.1%} | Train SR: {train_sr:.1%} | Entropy: {train_info['entropy']:.4f}", flush=True)
                
                # 模式特定信息
                if self.mode == 'curriculum' and hasattr(self.train_env, 'get_curriculum_info'):
                    curr_info = self.train_env.get_curriculum_info()
                    print(f"  Curriculum Level: {curr_info['level']}, Distance: {curr_info['current_distance']:.1f}m", flush=True)
                elif self.mode == 'obstacle':
                    print(f"  Collision Rate: {eval_info.get('collision_rate', 0):.1%}", flush=True)
                    if hasattr(self.train_env, 'get_curriculum_info'):
                        obs_info = self.train_env.get_curriculum_info()
                        print(f"  Obstacle Curriculum: Level {obs_info['level']}, "
                              f"Obstacles: {obs_info['current_obstacles']}/{obs_info['max_obstacles']}, "
                              f"Train SR: {obs_info['recent_success_rate']:.1%}, "
                              f"Collision: {obs_info['recent_collision_rate']:.1%}", flush=True)
                
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {best_sr:.1%}", flush=True)
            
            if timesteps % 50000 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\nTraining done! Best SR: {best_sr:.1%}")
        
        return best_sr


class CurriculumWrapper:
    """课程学习向量化环境包装器
    
    安全处理课程进阶：
    - 不销毁重建环境，而是直接更新 target_distance 参数
    - 进阶后重置所有环境
    """
    
    def __init__(self, num_envs: int = 4, 
                 initial_distance: float = 1.0,
                 max_distance: float = 10.0,
                 distance_increment: float = 0.5,
                 success_threshold_to_advance: float = 0.8,
                 window_size: int = 100,
                 **kwargs):
        self.num_envs = num_envs
        self.current_distance = initial_distance
        self.max_distance = max_distance
        self.distance_increment = distance_increment
        self.success_threshold_to_advance = success_threshold_to_advance
        self.window_size = window_size
        self.kwargs = kwargs
        
        # 创建环境（只在初始化时创建一次）
        self.envs = [LowLevelAUV2DSimpleRewardEnv(target_distance=initial_distance, **kwargs) 
                     for _ in range(num_envs)]
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # 课程追踪
        self.recent_successes = []
        self.curriculum_level = 0
        
        # 是否需要在下一次step前重置所有环境
        self._need_reset = False
    
    def _update_target_distance(self, new_distance: float):
        """更新所有环境的目标距离（不销毁环境）"""
        for env in self.envs:
            env.target_distance = new_distance
    
    def _check_and_advance(self):
        """检查是否需要进阶"""
        if len(self.recent_successes) < self.window_size:
            return False
        
        success_rate = sum(self.recent_successes[-self.window_size:]) / self.window_size
        
        if success_rate >= self.success_threshold_to_advance and self.current_distance < self.max_distance:
            old_distance = self.current_distance
            self.current_distance = min(self.current_distance + self.distance_increment, self.max_distance)
            self.curriculum_level += 1
            self.recent_successes = []
            
            print(f"\n>>> Curriculum Advanced! Level {self.curriculum_level}, "
                  f"Distance: {old_distance:.1f}m -> {self.current_distance:.1f}m")
            
            # 更新目标距离并标记需要重置
            self._update_target_distance(self.current_distance)
            self._need_reset = True
            
            return True
        return False
    
    def reset(self, seed=None):
        """重置所有环境"""
        if seed is not None:
            results = [env.reset(seed=seed+i) for i, env in enumerate(self.envs)]
        else:
            results = [env.reset() for env in self.envs]
        
        self._need_reset = False
        return np.array([r[0] for r in results]), [r[1] for r in results]
    
    def step(self, actions):
        """执行一步"""
        # 如果需要重置（课程进阶后），先重置所有环境
        if self._need_reset:
            obs, infos = self.reset()
            return (obs, 
                    np.zeros(self.num_envs), 
                    np.zeros(self.num_envs, dtype=bool), 
                    np.zeros(self.num_envs, dtype=bool),
                    infos)
        
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            
            if term or trunc:
                success = info.get('termination_reason') == 'success'
                self.recent_successes.append(1 if success else 0)
                obs, _ = env.reset()
            
            results.append((obs, reward, term, trunc, info))
        
        # 检查是否需要进阶（下次step时会处理重置）
        self._check_and_advance()
        
        return (np.array([r[0] for r in results]), 
                np.array([r[1] for r in results]),
                np.array([r[2] for r in results]), 
                np.array([r[3] for r in results]),
                [r[4] for r in results])
    
    def get_curriculum_info(self):
        """获取课程学习状态信息"""
        recent_sr = 0.0
        if len(self.recent_successes) > 0:
            recent_sr = sum(self.recent_successes[-100:]) / min(100, len(self.recent_successes))
        
        return {
            'level': self.curriculum_level,
            'current_distance': self.current_distance,
            'max_distance': self.max_distance,
            'recent_success_rate': recent_sr,
            'samples_in_window': len(self.recent_successes)
        }
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            try:
                env.close()
            except:
                pass
        self.envs = []


class ObstacleCurriculumWrapper:
    """避障课程学习包装器：从0个障碍物开始，逐步增加"""
    
    def __init__(self, num_envs: int = 4,
                 initial_num_obstacles: int = 0,
                 max_num_obstacles: int = 5,
                 obstacle_increment: int = 1,
                 success_threshold_to_advance: float = 0.7,
                 window_size: int = 100,
                 **env_kwargs):
        self.num_envs = num_envs
        self.current_num_obstacles = initial_num_obstacles
        self.max_num_obstacles = max_num_obstacles
        self.obstacle_increment = obstacle_increment
        self.success_threshold_to_advance = success_threshold_to_advance
        self.window_size = window_size
        self.env_kwargs = env_kwargs
        
        # 创建环境
        self.envs = []
        self._create_envs()
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # 课程追踪
        self.recent_successes = []
        self.recent_collisions = []
        self.curriculum_level = 0
        self._need_reset = False
    
    def _create_envs(self):
        """创建环境（使用当前障碍物数量）"""
        for env in self.envs:
            try:
                env.close()
            except:
                pass
        
        kwargs = self.env_kwargs.copy()
        kwargs['current_num_obstacles'] = self.current_num_obstacles
        self.envs = [LowLevelAUV2DObstacleEnv(**kwargs) for _ in range(self.num_envs)]
    
    def _check_and_advance(self):
        """检查是否需要进阶"""
        if len(self.recent_successes) < self.window_size:
            return False
        
        success_rate = sum(self.recent_successes[-self.window_size:]) / self.window_size
        collision_rate = sum(self.recent_collisions[-self.window_size:]) / self.window_size
        
        # 成功率足够高且碰撞率足够低时进阶
        if (success_rate >= self.success_threshold_to_advance and 
            collision_rate < 0.2 and 
            self.current_num_obstacles < self.max_num_obstacles):
            
            old_num = self.current_num_obstacles
            self.current_num_obstacles = min(
                self.current_num_obstacles + self.obstacle_increment,
                self.max_num_obstacles
            )
            self.curriculum_level += 1
            self.recent_successes = []
            self.recent_collisions = []
            
            print(f"\n>>> Obstacle Curriculum Advanced! Level {self.curriculum_level}, "
                  f"Obstacles: {old_num} -> {self.current_num_obstacles}")
            
            # 重建环境
            self._create_envs()
            self._need_reset = True
            return True
        return False
    
    def reset(self, seed=None):
        """重置所有环境"""
        if seed is not None:
            results = [env.reset(seed=seed+i) for i, env in enumerate(self.envs)]
        else:
            results = [env.reset() for env in self.envs]
        
        self._need_reset = False
        return np.array([r[0] for r in results]), [r[1] for r in results]
    
    def step(self, actions):
        """执行一步"""
        if self._need_reset:
            obs, infos = self.reset()
            return (obs, 
                    np.zeros(self.num_envs), 
                    np.zeros(self.num_envs, dtype=bool), 
                    np.zeros(self.num_envs, dtype=bool),
                    infos)
        
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            
            if term or trunc:
                success = info.get('termination_reason') == 'success'
                collision = info.get('termination_reason') == 'collision'
                self.recent_successes.append(1 if success else 0)
                self.recent_collisions.append(1 if collision else 0)
                obs, _ = env.reset()
            
            results.append((obs, reward, term, trunc, info))
        
        # 检查是否需要进阶
        self._check_and_advance()
        
        return (np.array([r[0] for r in results]), 
                np.array([r[1] for r in results]),
                np.array([r[2] for r in results]), 
                np.array([r[3] for r in results]),
                [r[4] for r in results])
    
    def get_curriculum_info(self):
        """获取课程学习状态"""
        recent_sr = 0.0
        recent_cr = 0.0
        if len(self.recent_successes) > 0:
            recent_sr = sum(self.recent_successes[-100:]) / min(100, len(self.recent_successes))
        if len(self.recent_collisions) > 0:
            recent_cr = sum(self.recent_collisions[-100:]) / min(100, len(self.recent_collisions))
        
        return {
            'level': self.curriculum_level,
            'current_obstacles': self.current_num_obstacles,
            'max_obstacles': self.max_num_obstacles,
            'recent_success_rate': recent_sr,
            'recent_collision_rate': recent_cr
        }
    
    def close(self):
        for env in self.envs:
            try:
                env.close()
            except:
                pass
        self.envs = []


def main():
    parser = argparse.ArgumentParser(description="Advanced 2D Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--mode", type=str, default='sparse_heading',
                        choices=['sparse_heading', 'curriculum', 'obstacle'])
    
    # 通用参数
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--goal-reward", type=float, default=100.0)
    parser.add_argument("--success-threshold", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-checkpoint", type=str, default=None,
                       help="Path to pretrain checkpoint (relative to PROJECT_ROOT), e.g., 'checkpoints/shaping_5m/best_model.pth'")
    
    # 课程学习参数
    parser.add_argument("--initial-distance", type=float, default=1.0)
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--distance-increment", type=float, default=0.5)
    parser.add_argument("--reward-type", type=str, default='shaping',
                        choices=['sparse_only', 'sparse_heading', 'distance_linear', 'minimal', 'shaping'])
    
    # 避障参数
    parser.add_argument("--num-obstacles", type=int, default=5)
    parser.add_argument("--collision-penalty", type=float, default=-50.0)
    parser.add_argument("--obstacle-avoidance-coef", type=float, default=0.5)
    parser.add_argument("--obstacle-curriculum", action="store_true", 
                       help="Use curriculum learning for obstacles (start from 0)")
    parser.add_argument("--obstacle-curriculum-threshold", type=float, default=0.7,
                       help="Success rate threshold to advance obstacle curriculum")
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if args.total_timesteps:
        config['training']['total_timesteps'] = args.total_timesteps
    
    trainer = AdvancedTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
