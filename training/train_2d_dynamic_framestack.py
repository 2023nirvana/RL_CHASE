"""
2D 动态目标训练脚本 - 帧堆叠版本
================================
用帧堆叠代替 LSTM，更简单且通常更有效

使用方法：
    python training/train_2d_dynamic_framestack.py --n-stack 4 --experiment-name dynamic_stack4
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

from core.environment.low_level_env_2d_dynamic import LowLevelAUV2DDynamicEnv


class FrameStackEnv:
    """帧堆叠环境包装器"""
    def __init__(self, env, n_stack=4):
        self.env = env
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        self.obs_dim = env.obs_dim * n_stack
        
    @property
    def observation_space(self):
        return type('Space', (), {'shape': (self.obs_dim,)})()
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_stacked_obs(), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, term, trunc, info
    
    def _get_stacked_obs(self):
        return np.concatenate(list(self.frames))


class VecFrameStackEnv:
    """向量化帧堆叠环境"""
    def __init__(self, num_envs=4, n_stack=4, **kwargs):
        self.envs = [FrameStackEnv(LowLevelAUV2DDynamicEnv(**kwargs), n_stack) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed=None):
        results = [env.reset(seed=seed+i if seed else None) for i, env in enumerate(self.envs)]
        return np.array([r[0] for r in results]), [r[1] for r in results]
    
    def step(self, actions):
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
            results.append((obs, reward, term, trunc, info))
        return (np.array([r[0] for r in results]), np.array([r[1] for r in results]),
                np.array([r[2] for r in results]), np.array([r[3] for r in results]), [r[4] for r in results])


class FrameStackTrainer:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_stack = args.n_stack
        print(f"Device: {self.device}, Frame Stack: {self.n_stack}")
        
        self.experiment_name = args.experiment_name or f"dynamic_stack{self.n_stack}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self._init_envs()
        self._init_network()
        self.success_count, self.episode_count = 0, 0

    def _init_envs(self):
        tc = self.config['training']
        self.num_envs = tc.get('n_envs', 4)
        env_kwargs = {'target_distance': self.args.target_distance, 'target_speed': self.args.target_speed, 'target_motion': self.args.target_motion}
        
        self.train_env = VecFrameStackEnv(num_envs=self.num_envs, n_stack=self.n_stack, **env_kwargs)
        self.eval_env = FrameStackEnv(LowLevelAUV2DDynamicEnv(**env_kwargs), self.n_stack)
        
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        print(f"Env: obs={self.obs_dim} ({self.obs_dim//self.n_stack}x{self.n_stack}), actions={self.num_actions}")
        print(f"  target_speed={self.args.target_speed}, motion={self.args.target_motion}")

    def _init_network(self):
        # 帧堆叠版本使用更大的第一层来处理堆叠的输入
        hidden1 = 256  # 更大的第一层
        hidden2 = 64
        
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, self.num_actions)
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1)
        ).to(self.device)
        
        tc = self.config['training']
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=tc.get('learning_rate', 3e-4))
        self.gamma, self.gae_lambda = tc.get('gamma', 0.99), tc.get('gae_lambda', 0.95)
        self.clip_range, self.ent_coef = tc.get('clip_range', 0.2), tc.get('ent_coef', 0.05)
        self.vf_coef, self.max_grad_norm = tc.get('vf_coef', 0.5), tc.get('max_grad_norm', 0.5)
        self.n_steps, self.batch_size, self.n_epochs = tc.get('n_steps', 1024), tc.get('batch_size', 256), tc.get('n_epochs', 4)
        
        total_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        print(f"Network params: {total_params:,}")

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
                value_loss = nn.functional.mse_loss(values, ret[bi])
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()
        return {'loss': loss.item(), 'entropy': entropy.item()}

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
        
        print(f"\n{'='*60}\nFrame Stack Training (n_stack={self.n_stack})\nMotion: {self.args.target_motion}, Speed: {self.args.target_speed}\n{'='*60}\n")
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
    parser.add_argument("--n-stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f: config = yaml.safe_load(f)
    if args.total_timesteps: config['training']['total_timesteps'] = args.total_timesteps
    
    FrameStackTrainer(config, args).train()

if __name__ == "__main__":
    main()
