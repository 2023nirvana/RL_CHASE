"""
简化奖励训练脚本 - 对比不同奖励结构
====================================

目的：测试哪种奖励结构能够训练成功

奖励类型：
1. sparse_only      - 纯稀疏：只有到达给奖励
2. distance_linear  - 线性距离：-k * distance
3. distance_exp     - 指数距离：exp(-distance)
4. minimal          - 最小化：距离 + 朝向 + 到达
5. shaping          - 势能塑形：基于距离变化

使用方法：
    # 测试稀疏奖励
    python training/train_2d_simple_reward.py --reward-type sparse_only --experiment-name sparse_test
    
    # 测试线性距离
    python training/train_2d_simple_reward.py --reward-type distance_linear --experiment-name linear_test
    
    # 测试势能塑形
    python training/train_2d_simple_reward.py --reward-type shaping --experiment-name shaping_test
    
    # 批量测试所有类型
    python training/train_2d_simple_reward.py --test-all
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
    VecLowLevelAUV2DSimpleRewardEnv
)


class SimpleRewardTrainer:
    """简化奖励训练器"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.reward_type = args.reward_type
        self.experiment_name = args.experiment_name or f"simple_{self.reward_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Reward Type: {self.reward_type}")
        print(f"Experiment: {self.experiment_name}")
        
        self._init_envs()
        self._init_network()
        self.success_count = 0
        self.episode_count = 0

    def _init_envs(self):
        tc = self.config['training']
        self.num_envs = tc.get('n_envs', 4)
        
        env_kwargs = {
            'reward_type': self.reward_type,
            'target_distance': self.args.target_distance,
            'goal_reward': self.args.goal_reward,
            'distance_coef': self.args.distance_coef,
            'heading_coef': self.args.heading_coef,
            'time_penalty': self.args.time_penalty,
            'success_threshold': self.args.success_threshold,
            'max_steps': self.args.max_steps
        }
        
        self.train_env = VecLowLevelAUV2DSimpleRewardEnv(num_envs=self.num_envs, **env_kwargs)
        self.eval_env = LowLevelAUV2DSimpleRewardEnv(**env_kwargs)
        
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        print(f"Env: obs={self.obs_dim}, actions={self.num_actions}")
        print(f"  goal_reward={self.args.goal_reward}, distance_coef={self.args.distance_coef}")

    def _init_network(self):
        hidden_sizes = [128, 64]
        
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
            'optimizer': self.optimizer.state_dict()
        }, path)

    def train(self, total_timesteps=None):
        if total_timesteps is None:
            total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*60}")
        print(f"Simple Reward Training: {self.reward_type}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        timesteps = 0
        best_sr = 0.0
        
        for update in range(1, num_updates + 1):
            data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(data)
            
            if timesteps % 5000 < steps_per_update:
                eval_info = self.evaluate(10)
                train_sr = self.success_count / max(1, self.episode_count)
                fps = timesteps / (time.time() - start_time)
                
                print(f"\nUpdate {update}/{num_updates} | Steps: {timesteps:,} | FPS: {fps:.0f}", flush=True)
                print(f"  Eval SR: {eval_info['success_rate']:.1%} | Train SR: {train_sr:.1%} | Entropy: {train_info['entropy']:.4f}", flush=True)
                print(f"  Reward Type: {self.reward_type}", flush=True)
                
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {best_sr:.1%}", flush=True)
            
            if timesteps % 20000 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\nTraining done! Best SR: {best_sr:.1%}")
        
        return best_sr


def run_all_tests(config, args, timesteps=500000):
    """运行所有奖励类型的对比测试"""
    reward_types = ['sparse_only', 'distance_linear', 'distance_exp', 'minimal', 'shaping']
    results = {}
    
    print("\n" + "="*70)
    print("REWARD ABLATION STUDY")
    print("="*70)
    
    for rtype in reward_types:
        print(f"\n>>> Testing reward type: {rtype}")
        args.reward_type = rtype
        args.experiment_name = f"ablation_{rtype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trainer = SimpleRewardTrainer(config, args)
        best_sr = trainer.train(total_timesteps=timesteps)
        results[rtype] = best_sr
        
        # 清理
        del trainer
        torch.cuda.empty_cache()
    
    # 汇总结果
    print("\n" + "="*70)
    print("ABLATION RESULTS")
    print("="*70)
    print(f"{'Reward Type':<20} {'Best Success Rate':>20}")
    print("-"*40)
    for rtype, sr in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{rtype:<20} {sr:>20.1%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Simple Reward Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--reward-type", type=str, default='minimal',
                        choices=['sparse_only', 'distance_linear', 'distance_exp', 
                                 'heading_only', 'minimal', 'shaping'])
    parser.add_argument("--target-distance", type=float, default=3.0)
    parser.add_argument("--goal-reward", type=float, default=100.0, help="到达目标的奖励")
    parser.add_argument("--distance-coef", type=float, default=0.5, help="距离奖励系数")
    parser.add_argument("--heading-coef", type=float, default=0.3, help="朝向奖励系数")
    parser.add_argument("--time-penalty", type=float, default=0.01, help="每步时间惩罚")
    parser.add_argument("--success-threshold", type=float, default=0.3, help="成功阈值(米)")
    parser.add_argument("--max-steps", type=int, default=2000, help="最大步数")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-all", action="store_true", help="测试所有奖励类型")
    parser.add_argument("--ablation-steps", type=int, default=500000, help="消融实验的步数")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if args.total_timesteps:
        config['training']['total_timesteps'] = args.total_timesteps
    
    if args.test_all:
        run_all_tests(config, args, timesteps=args.ablation_steps)
    else:
        trainer = SimpleRewardTrainer(config, args)
        trainer.train()


if __name__ == "__main__":
    main()
