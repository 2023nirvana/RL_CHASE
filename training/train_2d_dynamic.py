"""
2D 动态目标训练脚本 - MLP 和 LSTM 版本
========================================
方案B: 动态目标追踪

使用方法：
    # MLP版本
    python training/train_2d_dynamic.py --target-speed 0.3 --target-motion constant --experiment-name dynamic_mlp
    
    # LSTM版本
    python training/train_2d_dynamic.py --use-lstm --target-speed 0.3 --experiment-name dynamic_lstm
"""
import os, sys, argparse, yaml, time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_2d_dynamic import LowLevelAUV2DDynamicEnv, VecLowLevelAUV2DDynamicEnv


class LSTMActorCritic(nn.Module):
    """LSTM 版本的 Actor-Critic 网络"""
    def __init__(self, obs_dim, num_actions, hidden_size=128, lstm_hidden=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_hidden = lstm_hidden
        
        # 特征提取
        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU())
        
        # LSTM 层
        self.lstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True)
        
        # Actor 和 Critic 头
        self.actor_head = nn.Linear(lstm_hidden, num_actions)
        self.critic_head = nn.Linear(lstm_hidden, 1)
        
    def forward(self, obs, hidden=None):
        # obs: (batch, obs_dim) 或 (batch, seq, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # 添加序列维度
        
        batch_size, seq_len, _ = obs.shape
        
        # 特征提取
        features = self.feature(obs)  # (batch, seq, hidden)
        
        # LSTM
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.lstm_hidden, device=obs.device)
            c0 = torch.zeros(1, batch_size, self.lstm_hidden, device=obs.device)
            hidden = (h0, c0)
        
        lstm_out, new_hidden = self.lstm(features, hidden)
        
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        
        # Actor 和 Critic
        logits = self.actor_head(last_out)
        value = self.critic_head(last_out)
        
        return logits, value.squeeze(-1), new_hidden


class MLPActorCritic(nn.Module):
    """MLP 版本的 Actor-Critic 网络"""
    def __init__(self, obs_dim, num_actions, hidden_sizes=[128, 64]):
        super().__init__()
        
        # Actor
        actor_layers = []
        prev = obs_dim
        for h in hidden_sizes:
            actor_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        actor_layers.append(nn.Linear(prev, num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic
        critic_layers = []
        prev = obs_dim
        for h in hidden_sizes:
            critic_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)
    
    def forward(self, obs, hidden=None):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value, None


class DynamicTrainer:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lstm = args.use_lstm
        print(f"Device: {self.device}, Use LSTM: {self.use_lstm}")
        
        name_suffix = "lstm" if self.use_lstm else "mlp"
        self.experiment_name = args.experiment_name or f"dynamic_{name_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self._init_envs()
        self._init_network()
        self.success_count, self.episode_count = 0, 0

    def _init_envs(self):
        tc = self.config['training']
        self.num_envs = tc.get('n_envs', 4)
        env_kwargs = {
            'target_distance': self.args.target_distance,
            'target_speed': self.args.target_speed,
            'target_motion': self.args.target_motion
        }
        self.train_env = VecLowLevelAUV2DDynamicEnv(num_envs=self.num_envs, **env_kwargs)
        self.eval_env = LowLevelAUV2DDynamicEnv(**env_kwargs)
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        print(f"Dynamic Env: obs={self.obs_dim}, actions={self.num_actions}")
        print(f"  target_speed={self.args.target_speed}, motion={self.args.target_motion}")

    def _init_network(self):
        if self.use_lstm:
            self.network = LSTMActorCritic(self.obs_dim, self.num_actions).to(self.device)
        else:
            self.network = MLPActorCritic(self.obs_dim, self.num_actions).to(self.device)
        
        tc = self.config['training']
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=tc.get('learning_rate', 3e-4))
        self.gamma, self.gae_lambda = tc.get('gamma', 0.99), tc.get('gae_lambda', 0.95)
        self.clip_range, self.ent_coef = tc.get('clip_range', 0.2), tc.get('ent_coef', 0.05)
        self.vf_coef, self.max_grad_norm = tc.get('vf_coef', 0.5), tc.get('max_grad_norm', 0.5)
        self.n_steps, self.batch_size, self.n_epochs = tc.get('n_steps', 1024), tc.get('batch_size', 256), tc.get('n_epochs', 4)
        
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"Network params: {total_params:,}")

    def get_action(self, obs, hidden=None, deterministic=False):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            logits, value, new_hidden = self.network(obs_t, hidden)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.distributions.Categorical(probs).sample()
        return action.cpu().numpy(), value.cpu().numpy(), new_hidden

    def collect_rollouts(self):
        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []
        obs, _ = self.train_env.reset()
        hidden = None
        
        for _ in range(self.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits, values, hidden = self.network(obs_t, hidden)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)
            
            next_obs, rewards, terms, truncs, infos = self.train_env.step(actions.cpu().numpy())
            dones = np.logical_or(terms, truncs)
            
            obs_buf.append(obs); act_buf.append(actions.cpu().numpy()); rew_buf.append(rewards)
            done_buf.append(dones); val_buf.append(values.cpu().numpy()); logp_buf.append(logprobs.cpu().numpy())
            
            for i, info in enumerate(infos):
                if info.get('termination_reason') == 'success': self.success_count += 1
                if dones[i]: self.episode_count += 1
            
            # LSTM: 重置 hidden state 当 episode 结束
            if self.use_lstm and any(dones):
                h, c = hidden
                for i, d in enumerate(dones):
                    if d:
                        h[:, i, :] = 0
                        c[:, i, :] = 0
                hidden = (h, c)
            
            obs = next_obs
        
        with torch.no_grad():
            _, last_val, _ = self.network(torch.tensor(obs, dtype=torch.float32, device=self.device), hidden)
        
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
        actions = torch.tensor(data['actions'].reshape(-1), dtype=torch.long, device=self.device)
        old_logp = torch.tensor(data['logprobs'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = torch.tensor(data['advantages'].reshape(-1), dtype=torch.float32, device=self.device)
        ret = torch.tensor(data['returns'].reshape(-1), dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), self.batch_size):
                bi = idx[start:start+self.batch_size]
                logits, values, _ = self.network(obs[bi])
                dist = torch.distributions.Categorical(torch.softmax(logits, -1))
                new_logp = dist.log_prob(actions[bi])
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_logp - old_logp[bi])
                actor_loss = -torch.min(ratio * adv[bi], torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * adv[bi]).mean()
                value_loss = nn.functional.mse_loss(values, ret[bi])
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        return {'loss': loss.item(), 'entropy': entropy.item()}

    def evaluate(self, n_episodes=10):
        rewards, successes = [], 0
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            ep_r, done, hidden = 0, False, None
            while not done:
                action, _, hidden = self.get_action(obs[np.newaxis, :], hidden, deterministic=True)
                obs, r, term, trunc, info = self.eval_env.step(action[0])
                ep_r += r; done = term or trunc
                if info.get('termination_reason') == 'success': successes += 1
            rewards.append(ep_r)
        return {'mean_reward': np.mean(rewards), 'success_rate': successes / n_episodes}

    def save(self, path):
        torch.save({'network': self.network.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def train(self, total_timesteps=None):
        if total_timesteps is None: total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        net_type = "LSTM" if self.use_lstm else "MLP"
        print(f"\n{'='*60}\nDynamic Target Training ({net_type})\nMotion: {self.args.target_motion}, Speed: {self.args.target_speed}\n{'='*60}\n")
        start_time, timesteps, best_sr = time.time(), 0, 0.0
        
        for update in range(1, num_updates + 1):
            data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(data)
            
            if timesteps % 5000 < steps_per_update:
                eval_info = self.evaluate(10)
                sr = self.success_count / max(1, self.episode_count)
                print(f"\nUpdate {update}/{num_updates} | Steps: {timesteps:,} | FPS: {timesteps/(time.time()-start_time):.0f}", flush=True)
                print(f"  Eval SR: {eval_info['success_rate']:.1%} | Train SR: {sr:.1%} | Entropy: {train_info['entropy']:.4f}", flush=True)
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {best_sr:.1%}", flush=True)
            
            if timesteps % 20000 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\nTraining done! Best SR: {best_sr:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--target-distance", type=float, default=3.0)
    parser.add_argument("--target-speed", type=float, default=0.3)
    parser.add_argument("--target-motion", type=str, default='constant', choices=['constant', 'random_walk', 'circular', 'evasive'])
    parser.add_argument("--use-lstm", action="store_true", help="Use LSTM network")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f: config = yaml.safe_load(f)
    if args.total_timesteps: config['training']['total_timesteps'] = args.total_timesteps
    
    DynamicTrainer(config, args).train()

if __name__ == "__main__":
    main()
