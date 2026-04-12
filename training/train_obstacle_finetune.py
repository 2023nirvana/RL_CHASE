"""
避障微调训练脚本 - 基于预训练模型学习避障
============================================

策略：
1. 使用shaping_10m预训练模型作为基础
2. 观测空间扩展：8维 -> 10维（添加障碍物信息）
3. 课程学习：从0个障碍物开始，逐步增加到目标数量
4. 奖励设计：保持原有导航奖励 + 避障奖励

使用方法：
    # 基础避障训练
    python training/train_obstacle_finetune.py --experiment-name obs_finetune_v1

    # 变体：更多障碍物
    python training/train_obstacle_finetune.py --experiment-name obs_finetune_v2 --max-obstacles 8
    
    # 变体：更大碰撞惩罚
    python training/train_obstacle_finetune.py --experiment-name obs_finetune_v3 --collision-penalty -100
"""
import os, sys, argparse, yaml, time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from collections import deque

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_obstacle import LowLevelAUV2DObstacleEnv, VecLowLevelAUV2DObstacleEnv
from utils.visualizer import TrainingVisualizer


class ObstacleFinetuneTrainer:
    """
    避障微调训练器
    
    核心策略：
    1. 加载预训练模型的前几层权重
    2. 扩展输入层以接受障碍物信息
    3. 课程学习：逐步增加障碍物数量
    """
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.experiment_name = args.experiment_name or f"obs_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 可视化器
        self.visualizer = TrainingVisualizer(save_dir=os.path.join(self.log_dir, "plots"))
        
        # 课程学习状态
        self.current_num_obstacles = args.initial_obstacles
        self.curriculum_success_window = deque(maxlen=args.curriculum_window)
        
        print(f"Device: {self.device}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Obstacle Curriculum: {args.initial_obstacles} -> {args.max_obstacles}")
        
        self._init_envs()
        self._init_network()
        
        self.success_count = 0
        self.episode_count = 0
        self.collision_count = 0
        
    def _init_envs(self):
        """初始化环境"""
        tc = self.config['training']
        self.num_envs = tc.get('n_envs', 4)
        
        env_kwargs = {
            'target_distance': self.args.target_distance,
            'num_obstacles': self.args.max_obstacles,
            'current_num_obstacles': self.current_num_obstacles,
            'goal_reward': self.args.goal_reward,
            'collision_penalty': self.args.collision_penalty,
            'obstacle_avoidance_coef': self.args.obstacle_avoidance_coef,
            'distance_coef': self.args.distance_coef,
            'heading_coef': self.args.heading_coef,
            'success_threshold': self.args.success_threshold,
            'max_steps': self.args.max_steps
        }
        
        self.train_env = VecLowLevelAUV2DObstacleEnv(num_envs=self.num_envs, **env_kwargs)
        self.eval_env = LowLevelAUV2DObstacleEnv(**env_kwargs)
        
        self.obs_dim = self.train_env.observation_space.shape[0]  # 10维
        self.num_actions = self.train_env.action_space.n  # 6动作
        
        print(f"Env: obs={self.obs_dim}, actions={self.num_actions}")
        print(f"Current obstacles: {self.current_num_obstacles}")
        
    def _init_network(self):
        """初始化网络并加载预训练权重"""
        # 网络结构：必须与预训练模型架构匹配 [128, 64]
        hidden_sizes = [128, 64]
        
        # 创建新的Actor网络（10维输入）
        actor_layers = []
        prev = self.obs_dim  # 10维
        for h in hidden_sizes:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        actor_layers.append(nn.Linear(prev, self.num_actions))
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        
        # 创建新的Critic网络
        critic_layers = []
        prev = self.obs_dim
        for h in hidden_sizes:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        
        # 加载预训练权重（部分迁移）
        if self.args.pretrain_checkpoint:
            self._load_pretrain_weights(self.args.pretrain_checkpoint)
        
        # 优化器
        tc = self.config['training']
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=tc.get('learning_rate', 3e-4)
        )
        
        # 超参数
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
    
    def _load_pretrain_weights(self, checkpoint_path: str):
        """
        加载预训练权重（智能迁移）
        
        策略：
        - 预训练模型是8维输入，新模型是10维输入
        - 第一层权重扩展：复制前8维的权重，新增2维初始化为0
        - 其他层直接复制
        """
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Pretrain checkpoint not found: {checkpoint_path}")
            return
        
        print(f"\n>>> Loading pretrain weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        if 'actor' not in ckpt:
            print("  ✗ No actor weights in checkpoint")
            return
        
        pretrain_actor = ckpt['actor']
        pretrain_critic = ckpt.get('critic', None)
        
        # 获取预训练模型的输入维度
        first_key = list(pretrain_actor.keys())[0]
        pretrain_input_dim = pretrain_actor[first_key].shape[1]
        
        print(f"  Pretrain input dim: {pretrain_input_dim}")
        print(f"  Current input dim: {self.obs_dim}")
        
        # 加载Actor权重
        current_state = self.actor.state_dict()
        for key in pretrain_actor.keys():
            if key in current_state:
                pretrain_weight = pretrain_actor[key]
                current_weight = current_state[key]
                
                if pretrain_weight.shape == current_weight.shape:
                    # 形状相同，直接复制
                    current_state[key] = pretrain_weight
                    print(f"  ✓ Loaded {key}: {pretrain_weight.shape}")
                elif len(pretrain_weight.shape) == 2 and pretrain_weight.shape[1] == pretrain_input_dim:
                    # 第一层权重，需要扩展
                    # 新权重初始化为小随机值
                    new_weight = torch.zeros_like(current_weight)
                    # 复制前8维的权重
                    new_weight[:, :pretrain_input_dim] = pretrain_weight
                    # 新增的2维用小随机值初始化
                    new_weight[:, pretrain_input_dim:] = torch.randn(
                        new_weight.shape[0], self.obs_dim - pretrain_input_dim
                    ) * 0.01
                    current_state[key] = new_weight
                    print(f"  ✓ Extended {key}: {pretrain_weight.shape} -> {current_weight.shape}")
                else:
                    print(f"  ✗ Skipped {key}: shape mismatch {pretrain_weight.shape} vs {current_weight.shape}")
        
        self.actor.load_state_dict(current_state)
        
        # 加载Critic权重（同样处理）
        if pretrain_critic is not None:
            current_state = self.critic.state_dict()
            for key in pretrain_critic.keys():
                if key in current_state:
                    pretrain_weight = pretrain_critic[key]
                    current_weight = current_state[key]
                    
                    if pretrain_weight.shape == current_weight.shape:
                        current_state[key] = pretrain_weight
                    elif len(pretrain_weight.shape) == 2 and pretrain_weight.shape[1] == pretrain_input_dim:
                        new_weight = torch.zeros_like(current_weight)
                        new_weight[:, :pretrain_input_dim] = pretrain_weight
                        new_weight[:, pretrain_input_dim:] = torch.randn(
                            new_weight.shape[0], self.obs_dim - pretrain_input_dim
                        ) * 0.01
                        current_state[key] = new_weight
            
            self.critic.load_state_dict(current_state)
            print("  ✓ Loaded critic weights")
        
        print("  → Fine-tuning from pretrain model\n")
    
    def _update_curriculum(self, success: bool):
        """更新课程学习状态"""
        self.curriculum_success_window.append(1 if success else 0)
        
        if len(self.curriculum_success_window) >= self.args.curriculum_window:
            success_rate = sum(self.curriculum_success_window) / len(self.curriculum_success_window)
            
            # 成功率达标，增加障碍物数量
            if success_rate >= self.args.curriculum_threshold and self.current_num_obstacles < self.args.max_obstacles:
                self.current_num_obstacles += 1
                self.curriculum_success_window.clear()
                
                # 更新环境
                self.train_env.set_current_num_obstacles(self.current_num_obstacles)
                self.eval_env.current_num_obstacles = self.current_num_obstacles
                
                print(f"\n>>> Curriculum Update: Now using {self.current_num_obstacles} obstacles (success rate: {success_rate:.1%})")
    
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
                elif info.get('termination_reason') == 'collision':
                    self.collision_count += 1
                    self._update_curriculum(False)
                if dones[i]:
                    self.episode_count += 1
                    if info.get('termination_reason') not in ['success', 'collision']:
                        self._update_curriculum(False)
            
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
                elif info.get('termination_reason') == 'collision':
                    collisions += 1
            rewards.append(ep_r)
        return {
            'mean_reward': np.mean(rewards),
            'success_rate': successes / n_episodes,
            'collision_rate': collisions / n_episodes
        }
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_num_obstacles': self.current_num_obstacles
        }, path)
    
    def train(self, total_timesteps=None):
        if total_timesteps is None:
            total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*70}")
        print(f"Obstacle Finetune Training")
        print(f"Total: {total_timesteps:,}, Updates: {num_updates}")
        print(f"Pretrain: {self.args.pretrain_checkpoint}")
        print(f"Obstacle Curriculum: {self.args.initial_obstacles} -> {self.args.max_obstacles}")
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
            self.visualizer.add_scalar('curriculum/num_obstacles', self.current_num_obstacles, timesteps)
            
            sr = self.success_count / max(1, self.episode_count)
            cr = self.collision_count / max(1, self.episode_count)
            self.visualizer.add_scalar('train/success_rate', sr, timesteps)
            self.visualizer.add_scalar('train/collision_rate', cr, timesteps)
            
            # 评估和打印
            if timesteps % 5000 < steps_per_update:
                eval_info = self.evaluate(10)
                self.visualizer.add_scalar('eval/success_rate', eval_info['success_rate'], timesteps)
                self.visualizer.add_scalar('eval/collision_rate', eval_info['collision_rate'], timesteps)
                self.visualizer.add_scalar('eval/mean_reward', eval_info['mean_reward'], timesteps)
                
                fps = timesteps / (time.time() - start_time)
                print(f"\nUpdate {update}/{num_updates} | Steps: {timesteps:,} | FPS: {fps:.0f}", flush=True)
                print(f"  Obstacles: {self.current_num_obstacles} | Train SR: {sr:.1%} | Train CR: {cr:.1%}", flush=True)
                print(f"  Eval SR: {eval_info['success_rate']:.1%} | Eval CR: {eval_info['collision_rate']:.1%}", flush=True)
                print(f"  Actor Loss: {train_info['actor_loss']:.4f} | Entropy: {train_info['entropy']:.4f}", flush=True)
                
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {best_sr:.1%}", flush=True)
            
            # 定期保存曲线
            if timesteps - last_plot_step >= 5000:
                self.visualizer.plot_training_curves(
                    metrics=['train/actor_loss', 'train/critic_loss', 'train/entropy',
                            'train/success_rate', 'train/collision_rate', 'curriculum/num_obstacles'],
                    save_name='training_curves.png'
                )
                last_plot_step = timesteps
            
            # 保存检查点
            if timesteps % 20000 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        # 最终保存
        self.visualizer.plot_training_curves(
            metrics=['train/actor_loss', 'train/critic_loss', 'train/entropy',
                    'train/success_rate', 'train/collision_rate', 'curriculum/num_obstacles'],
            save_name='training_curves_final.png'
        )
        self.visualizer.save_data('training_data.json')
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        
        print(f"\n{'='*70}")
        print(f"Training done! Best SR: {best_sr:.1%}")
        print(f"Final obstacles: {self.current_num_obstacles}")
        print(f"Plots saved to: {self.visualizer.save_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Obstacle Avoidance Finetune Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    # 预训练模型
    parser.add_argument("--pretrain-checkpoint", type=str, 
                       default="checkpoints/shaping_10m/best_model.pth",
                       help="Path to pretrain model")
    
    # 环境参数
    parser.add_argument("--target-distance", type=float, default=5.0)
    parser.add_argument("--max-obstacles", type=int, default=5)
    parser.add_argument("--initial-obstacles", type=int, default=0)
    parser.add_argument("--success-threshold", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=2000)
    
    # 奖励参数
    parser.add_argument("--goal-reward", type=float, default=100.0)
    parser.add_argument("--collision-penalty", type=float, default=-50.0)
    parser.add_argument("--obstacle-avoidance-coef", type=float, default=0.5)
    parser.add_argument("--distance-coef", type=float, default=0.5)
    parser.add_argument("--heading-coef", type=float, default=0.3)
    
    # 课程学习参数
    parser.add_argument("--curriculum-threshold", type=float, default=0.7,
                       help="Success rate threshold to increase obstacles")
    parser.add_argument("--curriculum-window", type=int, default=100,
                       help="Window size for computing success rate")
    
    # 训练参数
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if args.total_timesteps:
        config['training']['total_timesteps'] = args.total_timesteps
    
    trainer = ObstacleFinetuneTrainer(config, args)
    trainer.train()


if __name__ == "__main__":
    main()
