"""
3D模型并行训练脚本 - 多个变体同时训练
========================================

尝试不同的网络结构、超参数组合，找到最佳配置

使用方法：
    # 训练所有变体（并行）
    python training/train_3d_variants_parallel.py --all
    
    # 训练特定变体
    python training/train_3d_variants_parallel.py --variants variant1 variant2
    
    # 指定目标距离
    python training/train_3d_variants_parallel.py --all --target-distance 5.0
"""
import os, sys, argparse, yaml, time, subprocess
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ==================== 变体配置 ====================
VARIANTS = {
    '3d_baseline': {
        'description': 'Baseline (128, 64)',
        'network': {'hidden_sizes': [128, 64]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 4,
            'n_steps': 1024,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_large_net': {
        'description': 'Large Network (256, 128, 64)',
        'network': {'hidden_sizes': [256, 128, 64]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 4,
            'n_steps': 1024,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_deep_net': {
        'description': 'Deep Network (128, 128, 64, 32)',
        'network': {'hidden_sizes': [128, 128, 64, 32]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 4,
            'n_steps': 1024,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_high_lr': {
        'description': 'High Learning Rate (1e-3)',
        'network': {'hidden_sizes': [128, 64]},
        'training': {
            'learning_rate': 1e-3,
            'n_envs': 4,
            'n_steps': 1024,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_low_clip': {
        'description': 'Low Clip Range (0.1)',
        'network': {'hidden_sizes': [128, 64]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 4,
            'n_steps': 1024,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.1,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_high_ent': {
        'description': 'High Entropy Coef (0.1)',
        'network': {'hidden_sizes': [128, 64]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 4,
            'n_steps': 1024,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.1,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_more_envs': {
        'description': 'More Environments (8)',
        'network': {'hidden_sizes': [128, 64]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 8,
            'n_steps': 1024,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_long_steps': {
        'description': 'Longer Steps (2048)',
        'network': {'hidden_sizes': [128, 64]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 4,
            'n_steps': 2048,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_large_batch': {
        'description': 'Large Batch (512)',
        'network': {'hidden_sizes': [128, 64]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 4,
            'n_steps': 1024,
            'batch_size': 512,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
    '3d_combined': {
        'description': 'Combined (Large Net + More Envs + Long Steps)',
        'network': {'hidden_sizes': [256, 128, 64]},
        'training': {
            'learning_rate': 3e-4,
            'n_envs': 8,
            'n_steps': 2048,
            'batch_size': 256,
            'n_epochs': 4,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    },
}


def create_variant_trainer(variant_name: str, variant_config: Dict, target_distance: float, 
                           total_timesteps: int, seed: int):
    """为每个变体创建训练脚本"""
    
    # 创建修改后的训练脚本
    trainer_code = f'''
"""
3D训练脚本 - {variant_name}
自动生成用于并行训练
"""
import os, sys, argparse, yaml, time
import numpy as np
import torch
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.environment.low_level_env_3d import LowLevelAUV3DEnv, VecLowLevelAUV3DEnv


class Trainer3D:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{variant_name}] Using device: {{self.device}}")
        
        self.experiment_name = args.experiment_name or f"{variant_name}_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}"
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
        self.train_env = VecLowLevelAUV3DEnv(num_envs=self.num_envs, target_distance=self.args.target_distance)
        self.eval_env = LowLevelAUV3DEnv(target_distance=self.args.target_distance)
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        print(f"[{variant_name}] 3D Env: obs={{self.obs_dim}}, actions={{self.num_actions}}, target={{self.args.target_distance}}m")

    def _init_network(self):
        hs = {variant_config['network']['hidden_sizes']}
        layers_actor = []
        prev = self.obs_dim
        for h in hs:
            layers_actor.extend([torch.nn.Linear(prev, h), torch.nn.ReLU()])
            prev = h
        layers_actor.append(torch.nn.Linear(prev, self.num_actions))
        self.actor = torch.nn.Sequential(*layers_actor).to(self.device)
        
        layers_critic = []
        prev = self.obs_dim
        for h in hs:
            layers_critic.extend([torch.nn.Linear(prev, h), torch.nn.ReLU()])
            prev = h
        layers_critic.append(torch.nn.Linear(prev, 1))
        self.critic = torch.nn.Sequential(*layers_critic).to(self.device)
        
        tc = self.config['training']
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), 
                                         lr=tc.get('learning_rate', 3e-4))
        self.gamma, self.gae_lambda = tc.get('gamma', 0.99), tc.get('gae_lambda', 0.95)
        self.clip_range, self.ent_coef = tc.get('clip_range', 0.2), tc.get('ent_coef', 0.05)
        self.vf_coef, self.max_grad_norm = tc.get('vf_coef', 0.5), tc.get('max_grad_norm', 0.5)
        self.n_steps, self.batch_size, self.n_epochs = tc.get('n_steps', 1024), tc.get('batch_size', 256), tc.get('n_epochs', 4)
        
        total_params = sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())
        print(f"[{variant_name}] Network params: {{total_params:,}}")

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
        return {{'obs': np.array(obs_buf), 'actions': np.array(act_buf), 'logprobs': np.array(logp_buf),
                'advantages': adv, 'returns': ret, 'mean_reward': np.mean(rew_buf)}}

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
                value_loss = torch.nn.functional.mse_loss(values, ret[bi])
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()
        return {{'loss': loss.item(), 'entropy': entropy.item()}}

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
        return {{'mean_reward': np.mean(rewards), 'success_rate': successes / n_episodes}}

    def save(self, path):
        torch.save({{'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 
                   'optimizer': self.optimizer.state_dict()}}, path)

    def train(self, total_timesteps=None):
        if total_timesteps is None: total_timesteps = self.config['training'].get('total_timesteps', 2000000)
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\\n[{{variant_name}}] {{'='*60}}\\nStarting 3D training\\nTotal: {{total_timesteps:,}}, Updates: {{num_updates}}\\n{{'='*60}}\\n")
        start_time, timesteps, best_sr = time.time(), 0, 0.0
        
        for update in range(1, num_updates + 1):
            data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(data)
            
            if timesteps % 5000 < steps_per_update:
                eval_info = self.evaluate(10)
                sr = self.success_count / max(1, self.episode_count)
                print(f"\\n[{variant_name}] Update {{update}}/{{num_updates}} | Steps: {{timesteps:,}} | FPS: {{timesteps/(time.time()-start_time):.0f}}", flush=True)
                print(f"  Eval SR: {{eval_info['success_rate']:.1%}} | Train SR: {{sr:.1%}} | Entropy: {{train_info['entropy']:.4f}}", flush=True)
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {{best_sr:.1%}}", flush=True)
            
            if timesteps % 20000 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{{timesteps}}.pth"))
        
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        print(f"\\n[{variant_name}] Training done! Best SR: {{best_sr:.1%}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--target-distance", type=float, default={target_distance})
    parser.add_argument("--total-timesteps", type=int, default={total_timesteps})
    parser.add_argument("--seed", type=int, default={seed})
    args = parser.parse_args()
    
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    
    config_path = args.config or os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
    with open(config_path) as f: config = yaml.safe_load(f)
    
    # 应用变体配置
    config['training'].update({variant_config['training']})
    
    Trainer3D(config, args).train()

if __name__ == "__main__":
    main()
'''
    
    return trainer_code


def train_variant(variant_name: str, variant_config: Dict, target_distance: float, 
                 total_timesteps: int, seed: int, gpu_id: int = None):
    """训练单个变体"""
    print(f"\n{'='*70}")
    print(f"Starting training: {variant_name}")
    print(f"Description: {variant_config['description']}")
    print(f"Target Distance: {target_distance}m")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"{'='*70}\n")
    
    # 创建临时训练脚本
    temp_script = os.path.join(PROJECT_ROOT, f"training/temp_train_{variant_name}.py")
    trainer_code = create_variant_trainer(variant_name, variant_config, target_distance, total_timesteps, seed)
    
    with open(temp_script, 'w') as f:
        f.write(trainer_code)
    
    # 构建命令
    cmd = [sys.executable, temp_script, 
           '--experiment-name', variant_name,
           '--target-distance', str(target_distance),
           '--total-timesteps', str(total_timesteps),
           '--seed', str(seed)]
    
    if gpu_id is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 运行训练
    try:
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, 
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1)
        
        # 实时输出
        for line in process.stdout:
            print(f"[{variant_name}] {line}", end='', flush=True)
        
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"[{variant_name}] Error: {e}")
        return False
    finally:
        # 清理临时脚本
        if os.path.exists(temp_script):
            os.remove(temp_script)


def train_parallel(variant_names: List[str], target_distance: float, total_timesteps: int, 
                  seed: int, max_parallel: int = 4):
    """并行训练多个变体"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    variants_to_train = {name: VARIANTS[name] for name in variant_names if name in VARIANTS}
    
    if not variants_to_train:
        print("No valid variants to train!")
        return
    
    print(f"\n{'='*70}")
    print(f"Parallel Training: {len(variants_to_train)} variants")
    print(f"Max Parallel: {max_parallel}")
    print(f"Target Distance: {target_distance}m")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"{'='*70}\n")
    
    # 使用进程池并行训练
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for i, (name, config) in enumerate(variants_to_train.items()):
            gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
            future = executor.submit(train_variant, name, config, target_distance, 
                                    total_timesteps, seed + i, gpu_id)
            futures[future] = name
        
        # 等待完成
        for future in as_completed(futures):
            variant_name = futures[future]
            try:
                success = future.result()
                if success:
                    print(f"\n[{variant_name}] ✓ Training completed successfully")
                else:
                    print(f"\n[{variant_name}] ✗ Training failed")
            except Exception as e:
                print(f"\n[{variant_name}] ✗ Error: {e}")
    
    print(f"\n{'='*70}")
    print("All training jobs completed!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Train 3D model variants in parallel")
    parser.add_argument("--all", action="store_true", help="Train all variants")
    parser.add_argument("--variants", nargs="+", help="Specific variants to train")
    parser.add_argument("--target-distance", type=float, default=5.0, help="Target distance")
    parser.add_argument("--total-timesteps", type=int, default=2000000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-parallel", type=int, default=4, help="Max parallel training jobs")
    args = parser.parse_args()
    
    import torch
    
    # 确定要训练的变体
    if args.all:
        variant_names = list(VARIANTS.keys())
    elif args.variants:
        variant_names = args.variants
    else:
        print("Available variants:")
        for name, config in VARIANTS.items():
            print(f"  {name:20s} - {config['description']}")
        print("\nUse --all to train all variants or --variants <name1> <name2> ...")
        return
    
    # 并行训练
    train_parallel(variant_names, args.target_distance, args.total_timesteps, 
                  args.seed, args.max_parallel)


if __name__ == "__main__":
    main()
