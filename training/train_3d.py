"""
3D 环境训练脚本 - 8 动作（含深度控制）
======================================
方案A: 从2D扩展到3D

使用方法：
    python training/train_3d.py --target-distance 5.0 --experiment-name 3d_dist_5m
"""
import os, sys, argparse, yaml, time
import numpy as np
import torch
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_3d import LowLevelAUV3DEnv, VecLowLevelAUV3DEnv
from utils.visualizer import TrainingVisualizer


class Trainer3D:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.experiment_name = args.experiment_name or f"3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 初始化可视化器
        self.visualizer = TrainingVisualizer(save_dir=os.path.join(self.log_dir, "plots"))
        
        self._init_envs()
        self._init_network()
        self.success_count, self.episode_count = 0, 0

    def _init_envs(self):
        tc = self.config['training']
        self.num_envs = self.args.n_envs if self.args.n_envs else tc.get('n_envs', 4)
        heading_coef = getattr(self.args, 'heading_coef', 0.5)
        direction_coef = getattr(self.args, 'direction_coef', 0.6)
        strict_direction_heading = getattr(self.args, 'strict_direction_heading', False)
        heading_penalty_wrong = getattr(self.args, 'heading_penalty_wrong', 0.0)
        env_kwargs = dict(
            target_distance=self.args.target_distance,
            heading_coef=heading_coef,
            direction_coef=direction_coef,
            strict_direction_heading=strict_direction_heading,
            heading_penalty_wrong=heading_penalty_wrong,
        )
        self.train_env = VecLowLevelAUV3DEnv(num_envs=self.num_envs, **env_kwargs)
        self.eval_env = LowLevelAUV3DEnv(**env_kwargs)
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        print(f"3D Env: obs={self.obs_dim}, actions={self.num_actions}, target={self.args.target_distance}m, n_envs={self.num_envs}")
        print(f"  heading_coef={heading_coef}, direction_coef={direction_coef}, strict_heading={strict_direction_heading}, wrong_penalty={heading_penalty_wrong}")

    def _init_network(self):
        # 支持自定义网络结构
        if self.args.hidden_sizes:
            hs = self.args.hidden_sizes
        else:
            hs = [128, 64]
        
        # Actor网络
        layers_actor = []
        prev = self.obs_dim
        for h in hs:
            layers_actor.extend([torch.nn.Linear(prev, h), torch.nn.ReLU()])
            prev = h
        layers_actor.append(torch.nn.Linear(prev, self.num_actions))
        self.actor = torch.nn.Sequential(*layers_actor).to(self.device)
        
        # Critic网络
        layers_critic = []
        prev = self.obs_dim
        for h in hs:
            layers_critic.extend([torch.nn.Linear(prev, h), torch.nn.ReLU()])
            prev = h
        layers_critic.append(torch.nn.Linear(prev, 1))
        self.critic = torch.nn.Sequential(*layers_critic).to(self.device)
        
        tc = self.config['training']
        
        # 支持命令行参数覆盖配置
        lr = self.args.learning_rate if self.args.learning_rate else tc.get('learning_rate', 3e-4)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        
        self.gamma, self.gae_lambda = tc.get('gamma', 0.99), tc.get('gae_lambda', 0.95)
        self.clip_range = self.args.clip_range if self.args.clip_range else tc.get('clip_range', 0.2)
        self.ent_coef = self.args.ent_coef if self.args.ent_coef else tc.get('ent_coef', 0.05)
        self.vf_coef, self.max_grad_norm = tc.get('vf_coef', 0.5), tc.get('max_grad_norm', 0.5)
        
        self.n_steps = self.args.n_steps if self.args.n_steps else tc.get('n_steps', 1024)
        self.batch_size = self.args.batch_size if self.args.batch_size else tc.get('batch_size', 256)
        self.n_epochs = tc.get('n_epochs', 4)
        
        total_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        print(f"Network: hidden_sizes={hs}, params={total_params:,}")
        print(f"Training: lr={lr}, clip={self.clip_range}, ent={self.ent_coef}, n_steps={self.n_steps}, batch={self.batch_size}")

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits = self.actor(torch.tensor(obs, dtype=torch.float32, device=self.device))
            probs = torch.softmax(logits, dim=-1)
            return (torch.argmax(probs, dim=-1) if deterministic else torch.distributions.Categorical(probs).sample()).cpu().numpy()

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
            
            obs_buf.append(obs); act_buf.append(actions.cpu().numpy()); rew_buf.append(rewards)
            done_buf.append(dones); val_buf.append(values.cpu().numpy()); logp_buf.append(logprobs.cpu().numpy())
            
            for i, info in enumerate(infos):
                if info.get('termination_reason') == 'success': self.success_count += 1
                if dones[i]: self.episode_count += 1
            obs = next_obs
        
        with torch.no_grad():
            last_val = self.critic(torch.tensor(obs, dtype=torch.float32, device=self.device)).squeeze(-1).cpu().numpy()
        
        adv, ret = self._compute_gae(np.array(rew_buf), np.array(val_buf), np.array(done_buf), last_val)
        return {'obs': np.array(obs_buf), 'actions': np.array(act_buf), 'logprobs': np.array(logp_buf),
                'advantages': adv, 'returns': ret, 'mean_reward': np.mean(rew_buf)}

    def _compute_gae(self, rewards, values, dones, last_val):
        adv = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            nv = last_val if t == len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma * nv * (1-dones[t]) - values[t]
            adv[t] = last_gae = delta + self.gamma * self.gae_lambda * (1-dones[t]) * last_gae
        return adv, adv + values

    def train_step(self, data):
        obs = torch.tensor(data['obs'].reshape(-1, self.obs_dim), dtype=torch.float32, device=self.device)
        actions = torch.tensor(data['actions'].reshape(-1), dtype=torch.long, device=self.device)
        old_logp = torch.tensor(data['logprobs'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = torch.tensor(data['advantages'].reshape(-1), dtype=torch.float32, device=self.device)
        ret = torch.tensor(data['returns'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0
        
        for _ in range(self.n_epochs):
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), self.batch_size):
                bi = idx[start:start+self.batch_size]
                logits = self.actor(obs[bi])
                dist = torch.distributions.Categorical(torch.softmax(logits, -1))
                new_logp = dist.log_prob(actions[bi])
                entropy = dist.entropy().mean()
                values = self.critic(obs[bi]).squeeze(-1)
                
                ratio = torch.exp(new_logp - old_logp[bi])
                actor_loss = -torch.min(ratio * adv[bi], torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * adv[bi]).mean()
                value_loss = torch.nn.functional.mse_loss(values, ret[bi])
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1
        
        return {
            'loss': loss.item(),
            'actor_loss': total_actor_loss / num_batches if num_batches > 0 else 0,
            'critic_loss': total_value_loss / num_batches if num_batches > 0 else 0,
            'entropy': total_entropy / num_batches if num_batches > 0 else 0
        }

    def evaluate(self, n_episodes=10):
        rewards, successes = [], 0
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            ep_r, done = 0, False
            while not done:
                action = self.get_action(obs[np.newaxis, :], deterministic=True)[0]
                obs, r, term, trunc, info = self.eval_env.step(action)
                ep_r += r; done = term or trunc
                if info.get('termination_reason') == 'success': successes += 1
            rewards.append(ep_r)
        return {'mean_reward': np.mean(rewards), 'success_rate': successes / n_episodes}

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def train(self, total_timesteps=None):
        if total_timesteps is None: total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*60}\nStarting 3D training\nTotal: {total_timesteps:,}, Updates: {num_updates}\n{'='*60}\n")
        start_time, timesteps, best_sr = time.time(), 0, 0.0
        
        # 记录初始状态
        plot_interval = 5000  # 每5000步绘制一次曲线
        last_plot_step = 0
        
        for update in range(1, num_updates + 1):
            data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(data)
            
            # 记录训练指标
            self.visualizer.add_scalar('train/actor_loss', train_info['actor_loss'], timesteps)
            self.visualizer.add_scalar('train/critic_loss', train_info['critic_loss'], timesteps)
            self.visualizer.add_scalar('train/total_loss', train_info['loss'], timesteps)
            self.visualizer.add_scalar('train/entropy', train_info['entropy'], timesteps)
            self.visualizer.add_scalar('rollout/mean_reward', data['mean_reward'], timesteps)
            
            # 计算训练成功率
            sr = self.success_count / max(1, self.episode_count)
            self.visualizer.add_scalar('train/success_rate', sr, timesteps)
            
            # 评估和打印
            if timesteps % 5000 < steps_per_update:
                eval_info = self.evaluate(10)
                self.visualizer.add_scalar('eval/success_rate', eval_info['success_rate'], timesteps)
                self.visualizer.add_scalar('eval/mean_reward', eval_info['mean_reward'], timesteps)
                
                print(f"\nUpdate {update}/{num_updates} | Steps: {timesteps:,} | FPS: {timesteps/(time.time()-start_time):.0f}", flush=True)
                print(f"  Eval SR: {eval_info['success_rate']:.1%} | Train SR: {sr:.1%} | Reward: {eval_info['mean_reward']:.2f}", flush=True)
                print(f"  Actor Loss: {train_info['actor_loss']:.4f} | Critic Loss: {train_info['critic_loss']:.4f} | Entropy: {train_info['entropy']:.4f}", flush=True)
                
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {best_sr:.1%}", flush=True)
            
            # 定期保存训练曲线
            if timesteps - last_plot_step >= plot_interval:
                self.visualizer.plot_training_curves(
                    metrics=['train/actor_loss', 'train/critic_loss', 'train/entropy', 
                            'rollout/mean_reward', 'train/success_rate', 'eval/success_rate'],
                    save_name='training_curves.png'
                )
                self.visualizer.plot_reward_curve(
                    reward_key='rollout/mean_reward',
                    save_name='reward_curve.png'
                )
                self.visualizer.plot_loss_curves(
                    save_name='loss_curves.png'
                )
                last_plot_step = timesteps
                print(f"  [Plots updated] Training curves saved", flush=True)
            
            # 保存检查点
            if timesteps % 20000 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        # 训练结束，保存最终曲线
        self.visualizer.plot_training_curves(
            metrics=['train/actor_loss', 'train/critic_loss', 'train/entropy', 
                    'rollout/mean_reward', 'train/success_rate', 'eval/success_rate'],
            save_name='training_curves_final.png'
        )
        self.visualizer.plot_reward_curve(
            reward_key='rollout/mean_reward',
            save_name='reward_curve_final.png'
        )
        self.visualizer.plot_loss_curves(
            save_name='loss_curves_final.png'
        )
        self.visualizer.save_data('training_data.json')
        
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\nTraining done! Best SR: {best_sr:.1%}")
        print(f"Training curves saved to: {self.visualizer.save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--target-distance", type=float, default=3.0)
    parser.add_argument("--heading-coef", type=float, default=0.5,
                        help="朝向目标奖励系数（默认 0.5）")
    parser.add_argument("--direction-coef", type=float, default=0.6,
                        help="对准且前进时的速度奖励系数（默认 0.6）")
    parser.add_argument("--strict-direction-heading", action="store_true",
                        help="仅当朝向误差<30°才给前进奖励，促先对准再走")
    parser.add_argument("--heading-penalty-wrong", type=float, default=0.0,
                        help="朝向>90°却前进时的小惩罚（默认 0）")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # 网络结构参数
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None, 
                       help="Hidden layer sizes, e.g., --hidden-sizes 256 128 64")
    # 训练超参数
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--clip-range", type=float, default=None, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient")
    parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=None, help="Number of steps per rollout")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    args = parser.parse_args()
    
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f: config = yaml.safe_load(f)
    if args.total_timesteps: config['training']['total_timesteps'] = args.total_timesteps
    
    Trainer3D(config, args).train()

if __name__ == "__main__":
    main()
