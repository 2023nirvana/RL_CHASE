"""
连续动作 PPO 训练脚本 - 真正的 RL 底层控制
==========================================

使用方法：
    python training/train_continuous.py --target-distance 2.0 --experiment-name continuous_2m
"""
import os, sys, argparse, yaml, time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_continuous import LowLevelContinuousEnv, VecLowLevelContinuousEnv


class GaussianActorCritic(nn.Module):
    """高斯策略 Actor-Critic（连续动作）"""
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 128]):
        super().__init__()
        
        # 共享特征提取器
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.feature = nn.Sequential(*layers)
        
        # Actor: 输出均值
        self.actor_mean = nn.Linear(hidden_sizes[-1], action_dim)
        # 可学习的对数标准差
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic
        self.critic = nn.Linear(hidden_sizes[-1], 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Actor 输出层使用小权重
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, obs):
        features = self.feature(obs)
        mean = self.actor_mean(features)
        # 限制 log_std 范围
        log_std = torch.clamp(self.actor_log_std, -2, 0.5)
        std = torch.exp(log_std)
        value = self.critic(features).squeeze(-1)
        return mean, std, value
    
    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        # Tanh 压缩到 [-1, 1]
        action = torch.tanh(action)
        return action, value
    
    def evaluate_actions(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        # 逆 tanh（用于计算 log_prob）
        actions_unscaled = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_prob = dist.log_prob(actions_unscaled).sum(dim=-1)
        # tanh 的雅可比修正
        log_prob -= torch.log(1 - actions**2 + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return value, log_prob, entropy


class ContinuousTrainer:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        self.experiment_name = args.experiment_name or f"continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self._init_envs()
        self._init_network()
        self.success_count, self.episode_count = 0, 0

    def _init_envs(self):
        tc = self.config['training']
        self.num_envs = tc.get('n_envs', 4)
        env_kw = {'target_distance': self.args.target_distance, 'enable_3d': self.args.enable_3d}
        self.train_env = VecLowLevelContinuousEnv(num_envs=self.num_envs, **env_kw)
        self.eval_env = LowLevelContinuousEnv(**env_kw)
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.action_dim = self.train_env.action_space.shape[0]
        print(f"Continuous Env: obs={self.obs_dim}, action={self.action_dim}, target={self.args.target_distance}m")

    def _init_network(self):
        self.network = GaussianActorCritic(self.obs_dim, self.action_dim, [256, 128]).to(self.device)
        tc = self.config['training']
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=tc.get('learning_rate', 3e-4))
        
        self.gamma = tc.get('gamma', 0.99)
        self.gae_lambda = tc.get('gae_lambda', 0.95)
        self.clip_range = tc.get('clip_range', 0.2)
        self.ent_coef = 0.01  # 连续动作用较小的熵系数
        self.vf_coef = tc.get('vf_coef', 0.5)
        self.max_grad_norm = tc.get('max_grad_norm', 0.5)
        self.n_steps = tc.get('n_steps', 2048)
        self.batch_size = tc.get('batch_size', 64)
        self.n_epochs = tc.get('n_epochs', 10)
        
        print(f"Network params: {sum(p.numel() for p in self.network.parameters()):,}")

    def collect_rollouts(self):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
        obs, _ = self.train_env.reset()
        
        for _ in range(self.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions, values = self.network.get_action(obs_t, deterministic=False)
                mean, std, _ = self.network(obs_t)
                dist = torch.distributions.Normal(mean, std)
                actions_unscaled = torch.atanh(torch.clamp(actions, -0.999, 0.999))
                logprobs = dist.log_prob(actions_unscaled).sum(dim=-1)
                logprobs -= torch.log(1 - actions**2 + 1e-6).sum(dim=-1)
            
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terms, truncs, infos = self.train_env.step(actions_np)
            dones = np.logical_or(terms, truncs)
            
            obs_buf.append(obs)
            act_buf.append(actions_np)
            rew_buf.append(rewards)
            done_buf.append(dones)
            val_buf.append(values.cpu().numpy())
            logp_buf.append(logprobs.cpu().numpy())
            
            for i, info in enumerate(infos):
                if info.get('termination_reason') == 'success': self.success_count += 1
                if dones[i]: self.episode_count += 1
            obs = next_obs
        
        with torch.no_grad():
            _, last_val = self.network.get_action(torch.tensor(obs, dtype=torch.float32, device=self.device))
        
        adv, ret = self._compute_gae(np.array(rew_buf), np.array(val_buf), np.array(done_buf), last_val.cpu().numpy())
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
        actions = torch.tensor(data['actions'].reshape(-1, self.action_dim), dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(data['logprobs'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = torch.tensor(data['advantages'].reshape(-1), dtype=torch.float32, device=self.device)
        ret = torch.tensor(data['returns'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), self.batch_size):
                bi = idx[start:start+self.batch_size]
                values, new_logp, entropy = self.network.evaluate_actions(obs[bi], actions[bi])
                
                ratio = torch.exp(new_logp - old_logp[bi])
                actor_loss = -torch.min(ratio * adv[bi], 
                                       torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * adv[bi]).mean()
                value_loss = nn.functional.mse_loss(values, ret[bi])
                entropy_loss = -entropy.mean()
                
                loss = actor_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return {'loss': loss.item(), 'entropy': -entropy_loss.item()}

    def evaluate(self, n_episodes=10):
        rewards, successes = [], 0
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            ep_r, done = 0, False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.network.get_action(obs_t, deterministic=True)
                obs, r, term, trunc, info = self.eval_env.step(action.cpu().numpy()[0])
                ep_r += r
                done = term or trunc
                if info.get('termination_reason') == 'success': successes += 1
            rewards.append(ep_r)
        return {'mean_reward': np.mean(rewards), 'success_rate': successes / n_episodes}

    def save(self, path):
        torch.save({'network': self.network.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def train(self, total_timesteps=None):
        if total_timesteps is None:
            total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*60}")
        print(f"Continuous Action Training (Direct Force Control)")
        print(f"Total: {total_timesteps:,}, Updates: {num_updates}")
        print(f"{'='*60}\n")
        
        start_time, timesteps, best_sr = time.time(), 0, 0.0
        
        for update in range(1, num_updates + 1):
            data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(data)
            
            if timesteps % 8192 < steps_per_update:
                eval_info = self.evaluate(10)
                sr = self.success_count / max(1, self.episode_count)
                fps = timesteps / (time.time() - start_time)
                print(f"\nUpdate {update}/{num_updates} | Steps: {timesteps:,} | FPS: {fps:.0f}", flush=True)
                print(f"  Eval SR: {eval_info['success_rate']:.1%} | Train SR: {sr:.1%}", flush=True)
                print(f"  Eval Reward: {eval_info['mean_reward']:.1f} | Entropy: {train_info['entropy']:.4f}", flush=True)
                
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {best_sr:.1%}", flush=True)
            
            if timesteps % 40960 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\nTraining done! Best SR: {best_sr:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--target-distance", type=float, default=2.0)
    parser.add_argument("--enable-3d", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if args.total_timesteps:
        config['training']['total_timesteps'] = args.total_timesteps
    
    ContinuousTrainer(config, args).train()


if __name__ == "__main__":
    main()
