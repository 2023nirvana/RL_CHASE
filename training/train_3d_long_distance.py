"""
3D 远距离训练脚本 - 解决"倒着开"问题 + 自动可视化
=======================================================

改进点：
1. 目标距离增大到 8-12m
2. 禁止后退接近目标（强惩罚）
3. 强制先转向后前进的奖励机制
4. 训练结束后自动生成轨迹图和GIF

使用方法：
    python training/train_3d_long_distance.py
    python training/train_3d_long_distance.py --target-distance 10.0
"""
import os, sys, argparse, yaml, time
import numpy as np
import torch
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.visualizer import TrainingVisualizer


class LongDistanceAUV3DEnv:
    """
    远距离3D环境 - 防止倒着开
    
    关键改进：
    1. 后退动作接近目标时强惩罚
    2. 必须先对准（heading < 45°）才给前进奖励
    3. 增加转向奖励
    """
    
    ACTION_NAMES = ["forward", "backward", "left", "right", "yaw_left", "yaw_right", "heave_up", "heave_down"]
    
    def __init__(self, target_distance=10.0, z_range=(2.0, 5.0)):
        from core.environment.low_level_env import LowLevelAUVEnv
        
        config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config_2d.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(PROJECT_ROOT, "configs/low_level_config.yaml")
        
        self.inner = LowLevelAUVEnv(config_path=config_path)
        self.target_distance = target_distance
        self.z_range = z_range
        
        # 动作映射
        self.action_map = {
            0: 1,   # forward  -> surge_forward
            1: 2,   # backward -> surge_backward
            2: 3,   # left     -> sway_left
            3: 4,   # right    -> sway_right
            4: 7,   # yaw_left  -> yaw_left
            5: 8,   # yaw_right -> yaw_right
            6: 5,   # heave_up
            7: 6,   # heave_down
        }
        
        import gymnasium as gym
        self.action_space = gym.spaces.Discrete(8)
        self.obs_dim = 10
        obs_high = np.ones(self.obs_dim) * 10.0
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        
        self.prev_pos_error = None
        self.prev_heading_error = None
        self.step_count = 0
        self.last_action = None
        
    def _sample_3d_target(self):
        """采样远距离目标"""
        angle = np.random.uniform(0, 2 * np.pi)
        # 远距离：0.7~1.0 倍的 target_distance
        distance = np.random.uniform(self.target_distance * 0.7, self.target_distance)
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        target_z = np.random.uniform(self.z_range[0], self.z_range[1])
        return np.array([target_x, target_y, target_z])
    
    def reset(self, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        
        # AUV 初始位置在原点，随机 yaw
        self.inner.dynamics.eta[:3] = np.array([0.0, 0.0, np.mean(self.z_range)])
        self.inner.dynamics.eta[3:] = np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])
        self.inner.dynamics.nu[:] = 0.0
        
        # 远距离目标
        target_pos = self._sample_3d_target()
        self.inner.target_state = np.zeros(6)
        self.inner.target_state[:3] = target_pos
        
        self._enforce_constraints()
        
        self.prev_pos_error = np.linalg.norm(target_pos)
        self.prev_heading_error = np.pi  # 初始假设最大朝向误差
        self.step_count = 0
        self.last_action = None
        
        obs = self._get_obs()
        info = {'target_pos': target_pos.tolist(), 'init_pos': self.inner.dynamics.eta[:3].tolist()}
        return obs, info
    
    def _enforce_constraints(self):
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        eta[3] = 0.0  # roll
        eta[4] = 0.0  # pitch
        nu[3] = 0.0
        nu[4] = 0.0
        eta[2] = np.clip(eta[2], self.z_range[0], self.z_range[1])
    
    def step(self, action):
        mapped_action = self.action_map[action]
        _, _, _, _, info = self.inner.step(mapped_action)
        self._enforce_constraints()
        self.step_count += 1
        self.last_action = action
        
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated, truncated, term_info = self._check_termination()
        
        info.update(reward_info)
        info.update(term_info)
        info['action_name'] = self.ACTION_NAMES[action]
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        pos_error_world = target[:3] - eta[:3]
        distance = np.linalg.norm(pos_error_world)
        
        target_angle_xy = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        heading_error = np.arctan2(np.sin(target_angle_xy - yaw), np.cos(target_angle_xy - yaw))
        
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        pos_error_body_x = cos_yaw * pos_error_world[0] + sin_yaw * pos_error_world[1]
        pos_error_body_y = -sin_yaw * pos_error_world[0] + cos_yaw * pos_error_world[1]
        pos_error_body_z = pos_error_world[2]
        
        obs = np.array([
            pos_error_body_x / 10.0,  # 适应远距离
            pos_error_body_y / 10.0,
            pos_error_body_z / 5.0,
            heading_error / np.pi,
            target_angle_xy / np.pi,
            nu[0] / 2.0, nu[1] / 2.0, nu[2] / 2.0,
            nu[5] / 2.0,
            distance / 10.0
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)
    
    def _compute_reward(self, action):
        eta = self.inner.dynamics.eta
        nu = self.inner.dynamics.nu
        target = self.inner.target_state
        
        pos_error = np.linalg.norm(eta[:3] - target[:3])
        pos_error_world = target[:2] - eta[:2]
        target_angle = np.arctan2(pos_error_world[1], pos_error_world[0])
        yaw = eta[5]
        heading_error = abs(np.arctan2(np.sin(target_angle - yaw), np.cos(target_angle - yaw)))
        
        reward_info = {}
        
        # 1. 位置误差惩罚
        position_reward = -0.2 * pos_error
        reward_info['position'] = position_reward
        
        # 2. 位置改进奖励
        improvement_reward = 0.0
        if self.prev_pos_error is not None:
            improvement = self.prev_pos_error - pos_error
            improvement_reward = 8.0 * improvement  # 靠近奖励
        self.prev_pos_error = pos_error
        reward_info['improvement'] = improvement_reward
        
        # 3. 朝向奖励（强调对准目标）
        heading_reward = 0.8 * (1.0 - heading_error / np.pi)
        if heading_error < np.pi / 6:  # <30°
            heading_reward += 0.3
        elif heading_error < np.pi / 4:  # <45°
            heading_reward += 0.15
        reward_info['heading'] = heading_reward
        
        # 4. 转向奖励：朝向改进
        turn_reward = 0.0
        if self.prev_heading_error is not None:
            heading_improve = self.prev_heading_error - heading_error
            if heading_improve > 0:
                turn_reward = 3.0 * heading_improve  # 强转向奖励
        self.prev_heading_error = heading_error
        reward_info['turn'] = turn_reward
        
        # 5. 正确前进奖励（必须对准才奖励前进）
        direction_reward = 0.0
        forward_vel = nu[0]
        if action == 0 and forward_vel > 0:  # forward action
            if heading_error < np.pi / 4:  # 对准 <45°
                direction_reward = 1.0 * forward_vel  # 大奖励
            elif heading_error < np.pi / 2:  # 45-90°
                direction_reward = 0.3 * forward_vel  # 小奖励
            else:  # >90° 还前进
                direction_reward = -0.5  # 惩罚
        reward_info['direction'] = direction_reward
        
        # 6. ★ 关键：禁止后退接近目标 ★
        backward_penalty = 0.0
        if action == 1:  # backward action
            backward_vel = -nu[0]  # backward 时 nu[0] < 0
            # 如果后退让距离变近（说明在"倒着开"）
            if self.prev_pos_error is not None and improvement_reward > 0:
                # 严厉惩罚"倒着接近目标"
                backward_penalty = -2.0 - 3.0 * abs(backward_vel)
            else:
                # 普通后退小惩罚
                backward_penalty = -0.3
        reward_info['backward_penalty'] = backward_penalty
        
        # 7. Z 轴方向奖励
        z_error = target[2] - eta[2]
        z_vel = nu[2]
        z_reward = 0.0
        if abs(z_error) > 0.1:
            if z_error > 0 and z_vel > 0:
                z_reward = 0.5 * z_vel
            elif z_error < 0 and z_vel < 0:
                z_reward = 0.5 * abs(z_vel)
        reward_info['z_direction'] = z_reward
        
        # 8. 接近目标奖励
        proximity_reward = 0.0
        if pos_error < 1.0:
            proximity_reward = 5.0 * (1.0 - pos_error)
        elif pos_error < 2.0:
            proximity_reward = 2.0 * (2.0 - pos_error)
        reward_info['proximity'] = proximity_reward
        
        # 9. 到达目标
        goal_bonus = 0.0
        if pos_error < 0.5:
            goal_bonus = 150.0
            reward_info['success'] = True
        else:
            reward_info['success'] = False
        reward_info['goal'] = goal_bonus
        
        # 10. 时间惩罚
        time_penalty = -0.02
        reward_info['time'] = time_penalty
        
        total = (position_reward + improvement_reward + heading_reward + turn_reward +
                 direction_reward + backward_penalty + z_reward + proximity_reward + 
                 goal_bonus + time_penalty)
        reward_info['total'] = total
        reward_info['pos_error'] = pos_error
        reward_info['heading_error'] = heading_error
        
        return total, reward_info
    
    def _check_termination(self):
        info = {}
        eta = self.inner.dynamics.eta
        target = self.inner.target_state
        pos_error = np.linalg.norm(eta[:3] - target[:3])
        
        if pos_error < 0.5:
            info['termination_reason'] = 'success'
            return True, False, info
        
        if abs(eta[0]) > 25 or abs(eta[1]) > 25 or eta[2] < 0 or eta[2] > 10:
            info['termination_reason'] = 'out_of_bounds'
            return True, False, info
        
        if self.step_count >= 4000:  # 更长时间限制
            info['termination_reason'] = 'timeout'
            return False, True, info
        
        info['termination_reason'] = None
        info['pos_error'] = pos_error
        return False, False, info


class VecLongDistanceEnv:
    """向量化环境"""
    def __init__(self, num_envs=4, **kwargs):
        self.num_envs = num_envs
        self.envs = [LongDistanceAUV3DEnv(**kwargs) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed=None):
        obs_list, info_list = [], []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list
    
    def step(self, actions):
        obs_list, reward_list, term_list, trunc_list, info_list = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return np.array(obs_list), np.array(reward_list), np.array(term_list), np.array(trunc_list), info_list


class Trainer3DLongDistance:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        self.experiment_name = args.experiment_name or f"3d_long_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.visualizer = TrainingVisualizer(save_dir=os.path.join(self.log_dir, "plots"))
        
        self._init_envs()
        self._init_network()
        self.success_count, self.episode_count = 0, 0
    
    def _init_envs(self):
        self.num_envs = self.args.n_envs
        env_kwargs = dict(target_distance=self.args.target_distance)
        self.train_env = VecLongDistanceEnv(num_envs=self.num_envs, **env_kwargs)
        self.eval_env = LongDistanceAUV3DEnv(**env_kwargs)
        self.obs_dim = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.n
        print(f"Env: obs={self.obs_dim}, actions={self.num_actions}, target_dist={self.args.target_distance}m")
    
    def _init_network(self):
        hs = self.args.hidden_sizes
        
        # Actor
        layers_actor = []
        prev = self.obs_dim
        for h in hs:
            layers_actor.extend([torch.nn.Linear(prev, h), torch.nn.ReLU()])
            prev = h
        layers_actor.append(torch.nn.Linear(prev, self.num_actions))
        self.actor = torch.nn.Sequential(*layers_actor).to(self.device)
        
        # Critic
        layers_critic = []
        prev = self.obs_dim
        for h in hs:
            layers_critic.extend([torch.nn.Linear(prev, h), torch.nn.ReLU()])
            prev = h
        layers_critic.append(torch.nn.Linear(prev, 1))
        self.critic = torch.nn.Sequential(*layers_critic).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=self.args.learning_rate
        )
        
        self.gamma, self.gae_lambda = 0.99, 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.05
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.n_steps = self.args.n_steps
        self.batch_size = self.args.batch_size
        self.n_epochs = 4
        
        print(f"Network: {hs}, lr={self.args.learning_rate}")
    
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
        return {'obs': np.array(obs_buf), 'actions': np.array(act_buf), 'logprobs': np.array(logp_buf),
                'advantages': adv, 'returns': ret, 'mean_reward': np.mean(rew_buf)}
    
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
                actor_loss = -torch.min(ratio * adv[bi], 
                                       torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv[bi]).mean()
                value_loss = torch.nn.functional.mse_loss(values, ret[bi])
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 
                                               self.max_grad_norm)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1
        
        return {'actor_loss': total_actor_loss / num_batches, 
                'critic_loss': total_value_loss / num_batches,
                'entropy': total_entropy / num_batches}
    
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
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)
    
    def run_visualization(self):
        """训练结束后自动生成轨迹可视化和GIF"""
        print("\n" + "=" * 60)
        print("Generating trajectory visualizations and GIFs...")
        print("=" * 60)
        
        import subprocess
        viz_script = os.path.join(PROJECT_ROOT, "evaluation", "eval_3d_trajectory_visualize.py")
        
        # 将当前模型配置添加到可视化脚本的模型列表（临时方案：直接运行）
        cmd = [
            sys.executable, viz_script,
            "--models", self.experiment_name,
            "--episodes", "5",
            "--save-plots",
            "--save-gif",
            "--max-plots", "5"
        ]
        
        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            print(f"Visualizations saved to: {self.log_dir}/plots/")
        except Exception as e:
            print(f"Visualization failed: {e}")
            # 备用方案：直接在这里生成
            self._generate_visualizations_inline()
    
    def _generate_visualizations_inline(self):
        """内联生成可视化"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        print("Generating visualizations inline...")
        
        plot_dir = os.path.join(self.log_dir, "plots")
        gif_dir = os.path.join(plot_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)
        
        # 运行几个评估 episode 并收集轨迹
        for ep_idx in range(5):
            obs, info = self.eval_env.reset()
            trajectory = {
                'positions': [self.eval_env.inner.dynamics.eta[:2].tolist()],
                'z': [float(self.eval_env.inner.dynamics.eta[2])],
                'yaw_angles': [float(self.eval_env.inner.dynamics.eta[5])],
                'target': info['target_pos']
            }
            
            done = False
            while not done:
                action = self.get_action(obs[np.newaxis, :], deterministic=True)[0]
                obs, _, term, trunc, info = self.eval_env.step(action)
                done = term or trunc
                
                eta = self.eval_env.inner.dynamics.eta
                trajectory['positions'].append(eta[:2].tolist())
                trajectory['z'].append(float(eta[2]))
                trajectory['yaw_angles'].append(float(eta[5]))
            
            success = info.get('termination_reason') == 'success'
            
            # 绘制 3D 轨迹图
            self._plot_trajectory_3d(trajectory, success, ep_idx, plot_dir)
            
            # 生成 GIF
            self._save_trajectory_gif(trajectory, success, ep_idx, gif_dir)
        
        print(f"Visualizations saved to: {plot_dir}")
    
    def _plot_trajectory_3d(self, traj, success, ep_idx, save_dir):
        import matplotlib.pyplot as plt
        
        pos = np.array(traj['positions'])
        z = np.array(traj['z'])
        yaw = np.array(traj['yaw_angles'])
        target = np.array(traj['target'])
        
        x, y = pos[:, 0], pos[:, 1]
        n_pts = len(x)
        
        fig = plt.figure(figsize=(14, 5))
        
        # 3D 轨迹
        ax1 = fig.add_subplot(131, projection='3d')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_pts - 1))
        for i in range(n_pts - 1):
            ax1.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], lw=2)
        ax1.scatter([x[0]], [y[0]], [z[0]], c='green', s=100, marker='o', label='Start')
        ax1.scatter([x[-1]], [y[-1]], [z[-1]], c='blue', s=80, marker='s', label='End')
        ax1.scatter([target[0]], [target[1]], [target[2]], c='red', s=150, marker='*', label='Target')
        
        # 绘制航向箭头
        arrow_interval = max(1, n_pts // 12)
        arrow_scale = 0.5
        for i in range(0, n_pts, arrow_interval):
            u = arrow_scale * np.cos(yaw[i])
            v = arrow_scale * np.sin(yaw[i])
            ax1.quiver(x[i], y[i], z[i], u, v, 0, color='orange', arrow_length_ratio=0.2, linewidth=1.2)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D Trajectory - {"Success" if success else "Fail"}')
        ax1.legend()
        
        # XY 平面
        ax2 = fig.add_subplot(132)
        ax2.plot(x, y, 'b-', lw=2, alpha=0.7)
        ax2.plot(x[0], y[0], 'go', markersize=12, label='Start')
        ax2.plot(x[-1], y[-1], 'bs', markersize=10, label='End')
        ax2.plot(target[0], target[1], 'r*', markersize=14, label='Target')
        for i in range(0, n_pts, arrow_interval):
            dx, dy = arrow_scale * np.cos(yaw[i]), arrow_scale * np.sin(yaw[i])
            ax2.arrow(x[i], y[i], dx, dy, head_width=0.15, head_length=0.08, fc='orange', ec='orange', alpha=0.7)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Trajectory (heading arrows)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 深度曲线
        ax3 = fig.add_subplot(133)
        ax3.plot(range(len(z)), z, 'b-', lw=2, label='AUV depth')
        ax3.axhline(y=target[2], color='r', linestyle='--', label='Target depth')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Depth over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'[{self.experiment_name}] Episode {ep_idx+1} - {"Success" if success else "Fail"}')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'trajectory_3d_ep{ep_idx+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")
    
    def _save_trajectory_gif(self, traj, success, ep_idx, save_dir):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter
        except ImportError:
            print("  matplotlib.animation not available, skip GIF")
            return
        
        pos = np.array(traj['positions'])
        z = np.array(traj['z'])
        yaw = np.array(traj['yaw_angles'])
        target = np.array(traj['target'])
        
        x, y = pos[:, 0], pos[:, 1]
        n_pts = len(x)
        
        every_n = 3
        indices = list(range(0, n_pts, every_n))
        if indices[-1] != n_pts - 1:
            indices.append(n_pts - 1)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        margin = 1.0
        ax.set_xlim(x.min() - margin, x.max() + margin)
        ax.set_ylim(y.min() - margin, y.max() + margin)
        ax.set_zlim(z.min() - margin, z.max() + margin)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.scatter([target[0]], [target[1]], [target[2]], c='red', s=150, marker='*', label='Target')
        
        line, = ax.plot([], [], [], 'b-', lw=2, alpha=0.8)
        point, = ax.plot([], [], [], 'go', markersize=12)
        title_text = ax.set_title('')
        qv_holder = [None]
        arrow_scale = 0.5
        
        def animate(i):
            idx = indices[min(i, len(indices) - 1)]
            line.set_data(x[:idx+1], y[:idx+1])
            line.set_3d_properties(z[:idx+1])
            point.set_data([x[idx]], [y[idx]])
            point.set_3d_properties([z[idx]])
            if qv_holder[0] is not None:
                try:
                    qv_holder[0].remove()
                except:
                    pass
            u = arrow_scale * np.cos(yaw[idx])
            v = arrow_scale * np.sin(yaw[idx])
            qv_holder[0] = ax.quiver(x[idx], y[idx], z[idx], u, v, 0, color='orange', 
                                     arrow_length_ratio=0.25, linewidth=2)
            title_text.set_text(f'Step {idx}/{n_pts-1} | {"Success" if success else "Fail"}')
            return line, point, title_text
        
        anim = animation.FuncAnimation(fig, animate, frames=len(indices), interval=80, blit=False)
        
        save_path = os.path.join(save_dir, f'{self.experiment_name}_trajectory_3d_ep{ep_idx+1}.gif')
        writer = PillowWriter(fps=12)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        print(f"  GIF saved: {save_path}")
    
    def train(self, total_timesteps):
        steps_per_update = self.n_steps * self.num_envs
        num_updates = total_timesteps // steps_per_update
        
        print(f"\n{'='*60}")
        print(f"Training: {self.experiment_name}")
        print(f"Total: {total_timesteps:,}, Updates: {num_updates}")
        print(f"Target distance: {self.args.target_distance}m")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        timesteps = 0
        best_sr = 0.0
        
        for update in range(1, num_updates + 1):
            data = self.collect_rollouts()
            timesteps += steps_per_update
            train_info = self.train_step(data)
            
            self.visualizer.add_scalar('train/actor_loss', train_info['actor_loss'], timesteps)
            self.visualizer.add_scalar('train/critic_loss', train_info['critic_loss'], timesteps)
            self.visualizer.add_scalar('rollout/mean_reward', data['mean_reward'], timesteps)
            
            sr = self.success_count / max(1, self.episode_count)
            self.visualizer.add_scalar('train/success_rate', sr, timesteps)
            
            if timesteps % 10000 < steps_per_update:
                eval_info = self.evaluate(10)
                self.visualizer.add_scalar('eval/success_rate', eval_info['success_rate'], timesteps)
                self.visualizer.add_scalar('eval/mean_reward', eval_info['mean_reward'], timesteps)
                
                fps = timesteps / (time.time() - start_time)
                print(f"\nUpdate {update}/{num_updates} | Steps: {timesteps:,} | FPS: {fps:.0f}")
                print(f"  Eval SR: {eval_info['success_rate']:.1%} | Train SR: {sr:.1%}")
                print(f"  Actor Loss: {train_info['actor_loss']:.4f} | Entropy: {train_info['entropy']:.4f}")
                
                if eval_info['success_rate'] > best_sr:
                    best_sr = eval_info['success_rate']
                    self.save(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"  -> New best! SR: {best_sr:.1%}")
                
                # 绘制训练曲线
                self.visualizer.plot_training_curves(
                    metrics=['train/actor_loss', 'train/critic_loss', 'train/entropy',
                            'rollout/mean_reward', 'train/success_rate', 'eval/success_rate'],
                    save_name='training_curves.png'
                )
            
            if timesteps % 50000 < steps_per_update:
                self.save(os.path.join(self.checkpoint_dir, f"checkpoint_{timesteps}.pth"))
        
        self.save(os.path.join(self.checkpoint_dir, "final_model.pth"))
        
        # 保存最终训练曲线
        self.visualizer.plot_training_curves(
            metrics=['train/actor_loss', 'train/critic_loss', 'train/entropy',
                    'rollout/mean_reward', 'train/success_rate', 'eval/success_rate'],
            save_name='training_curves_final.png'
        )
        self.visualizer.save_data('training_data.json')
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best SR: {best_sr:.1%}")
        print(f"{'='*60}")
        
        # 自动生成可视化
        self._generate_visualizations_inline()
        
        return best_sr


def main():
    parser = argparse.ArgumentParser(description="3D Long Distance Training - Anti-Backward")
    parser.add_argument("--experiment-name", type=str, default="3d_long_distance")
    parser.add_argument("--target-distance", type=float, default=10.0, help="Target distance (default: 10m)")
    parser.add_argument("--total-timesteps", type=int, default=3000000, help="Total training steps")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    trainer = Trainer3DLongDistance(args)
    trainer.train(args.total_timesteps)


if __name__ == "__main__":
    main()
