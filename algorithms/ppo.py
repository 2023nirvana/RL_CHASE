"""  
PPO算法实现  
===========  
  
Proximal Policy Optimization算法的完整实现，  
支持LSTM和多种训练技巧。  
  
参考：  
- Schulman et al., "Proximal Policy Optimization Algorithms"  
- Implementation details from CleanRL  
"""  
  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.tensorboard import SummaryWriter  
from typing import Dict, Tuple, Optional, List, Callable  
from collections import defaultdict
import time  
import os  
import sys  
  
# 添加项目路径  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, PROJECT_ROOT)  
  
from networks.ppo_lstm_network import PPOLSTMNetwork, PPONetwork  
from algorithms.replay_buffer import RolloutBuffer, RolloutBufferWithLSTM  
  
  
class PPO:  
    """  
    PPO算法  
      
    支持：  
    - 离散动作空间  
    - LSTM策略网络  
    - GAE优势估计  
    - 价值函数裁剪  
    - 熵正则化  
    - 学习率衰减  
    - 梯度裁剪  
    """  
      
    def __init__(self,  
                 env,  
                 network: nn.Module,  
                 learning_rate: float = 3e-4,  
                 n_steps: int = 2048,  
                 batch_size: int = 64,  
                 n_epochs: int = 10,  
                 gamma: float = 0.99,  
                 gae_lambda: float = 0.95,  
                 clip_range: float = 0.2,  
                 clip_range_vf: Optional[float] = None,  
                 ent_coef: float = 0.01,  
                 vf_coef: float = 0.5,  
                 max_grad_norm: float = 0.5,  
                 target_kl: Optional[float] = None,  
                 use_lstm: bool = True,  
                 lstm_seq_length: int = 16,  
                 device: torch.device = None,  
                 tensorboard_log: Optional[str] = None,  
                 verbose: int = 1,
                 logger: Optional[object] = None):
        """  
        初始化PPO算法  
          
        Args:  
            env: 训练环境（支持向量化环境）  
            network: 策略网络  
            learning_rate: 学习率  
            n_steps: 每次更新收集的步数  
            batch_size: 批次大小  
            n_epochs: 每次更新的epoch数  
            gamma: 折扣因子  
            gae_lambda: GAE参数  
            clip_range: PPO裁剪范围  
            clip_range_vf: 价值函数裁剪范围（None表示不裁剪）  
            ent_coef: 熵系数  
            vf_coef: 价值函数系数  
            max_grad_norm: 最大梯度范数  
            target_kl: 目标KL散度（用于早停）  
            use_lstm: 是否使用LSTM  
            lstm_seq_length: LSTM序列长度  
            device: 计算设备  
            tensorboard_log: TensorBoard日志目录  
            verbose: 日志详细程度  
        """  
        self.env = env  
        self.network = network  
        self.learning_rate = learning_rate  
        self.n_steps = n_steps  
        self.batch_size = batch_size  
        self.n_epochs = n_epochs  
        self.gamma = gamma  
        self.gae_lambda = gae_lambda  
        self.clip_range = clip_range  
        self.clip_range_vf = clip_range_vf  
        self.ent_coef = ent_coef  
        self.vf_coef = vf_coef  
        self.max_grad_norm = max_grad_norm  
        self.target_kl = target_kl  
        self.use_lstm = use_lstm  
        self.lstm_seq_length = lstm_seq_length  
        self.verbose = verbose  
          
        # 设备  
        if device is None:  
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        else:  
            self.device = device  
              
        self.network = self.network.to(self.device)  
          
        # 优化器  
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)  
          
        # 获取环境信息  
        self.num_envs = getattr(env, 'num_envs', 1)  
        self.observation_dim = env.observation_space.shape[0]  
        self.num_actions = env.action_space.n  
          
        # 初始化缓冲区  
        if use_lstm:  
            self.buffer = RolloutBufferWithLSTM(  
                buffer_size=n_steps,  
                observation_dim=self.observation_dim,  
                num_envs=self.num_envs,  
                lstm_hidden_size=network.lstm_hidden_size,  
                lstm_num_layers=network.lstm_num_layers,  
                gamma=gamma,  
                gae_lambda=gae_lambda,  
                device=self.device  
            )  
        else:  
            self.buffer = RolloutBuffer(  
                buffer_size=n_steps,  
                observation_dim=self.observation_dim,  
                num_envs=self.num_envs,  
                gamma=gamma,  
                gae_lambda=gae_lambda,  
                device=self.device  
            )  
              
        # TensorBoard  
        self.writer = None  
        if tensorboard_log is not None:  
            self.writer = SummaryWriter(tensorboard_log)  
        # 可选的外部 Logger（用于写入 CSV 等）
        self.logger = logger
              
        # 训练统计  
        self.num_timesteps = 0  
        self.num_episodes = 0  
        self.episode_rewards = []  
        self.episode_lengths = []  
          
        # LSTM隐藏状态  
        if use_lstm:  
            self.lstm_states = self._get_initial_lstm_states()  
        else:  
            self.lstm_states = None  
              
        # 上一次的dones  
        self.last_dones = np.zeros(self.num_envs, dtype=np.float32)  
          
    def _get_initial_lstm_states(self) -> Tuple[torch.Tensor, torch.Tensor]:  
        """获取初始LSTM状态"""  
        return self.network.lstm.get_initial_state(self.num_envs, self.device)  
      
    def collect_rollouts(self) -> Dict[str, float]:  
        """  
        收集rollout数据  
        
        Returns:  
            rollout_info: rollout统计信息  
        """  
        self.network.eval()  
        self.buffer.reset()  
        
        rollout_rewards = []  
        rollout_lengths = []  
        # 用于收集 rollout 期间 env 返回的 reward components
        reward_comp_acc = defaultdict(list)
        
        # 获取当前观测  
        if hasattr(self, 'last_obs'):  
            obs = self.last_obs  
        else:  
            obs, _ = self.env.reset()  
            self.last_obs = obs  
            
        episode_rewards = np.zeros(self.num_envs)  
        episode_lengths = np.zeros(self.num_envs)  
        
        with torch.no_grad():  
            for step in range(self.n_steps):  
                # 转换观测 - 确保形状为 [num_envs, obs_dim]  
                if isinstance(obs, np.ndarray):  
                    if obs.ndim == 1:  
                        obs = obs.reshape(1, -1)  # [obs_dim] -> [1, obs_dim]  
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)  
                
                # 转换done - 确保形状为 [num_envs]  
                done_tensor = torch.tensor(self.last_dones, dtype=torch.float32, device=self.device)  
                
                # 获取动作  
                if self.use_lstm:  
                    action, log_prob, _, value, new_lstm_states = self.network.get_action_and_value(  
                        obs_tensor, self.lstm_states, done_tensor  
                    )  
                    
                    # 保存隐藏状态（转换为numpy）  
                    hidden_state_np = (  
                        self.lstm_states[0].cpu().numpy(),  
                        self.lstm_states[1].cpu().numpy()  
                    )  
                else:  
                    action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)  
                    hidden_state_np = None  
                    
                action_np = action.cpu().numpy()  
                log_prob_np = log_prob.cpu().numpy()  
                value_np = value.cpu().numpy()  
                
                # 确保action是正确的形状  
                if action_np.ndim == 0:  
                    action_np = np.array([action_np])  
                
                # 执行动作  
                next_obs, rewards, terminateds, truncateds, infos = self.env.step(  
                    action_np[0] if self.num_envs == 1 and action_np.shape[0] == 1 else action_np  
                )  
                
                # 处理单环境返回值  
                if self.num_envs == 1:  
                    if not isinstance(rewards, np.ndarray):  
                        rewards = np.array([rewards])  
                    if not isinstance(terminateds, np.ndarray):  
                        terminateds = np.array([terminateds])  
                    if not isinstance(truncateds, np.ndarray):  
                        truncateds = np.array([truncateds])  
                    if next_obs.ndim == 1:  
                        next_obs = next_obs.reshape(1, -1)  
                        
                dones = np.logical_or(terminateds, truncateds).astype(np.float32)  
                
                # 更新episode统计  
                episode_rewards += rewards  
                episode_lengths += 1  
                
                # 检查完成的episode  
                for env_idx in range(self.num_envs):  
                    if dones[env_idx]:  
                        rollout_rewards.append(episode_rewards[env_idx])  
                        rollout_lengths.append(episode_lengths[env_idx])  
                        episode_rewards[env_idx] = 0  
                        episode_lengths[env_idx] = 0  
                        self.num_episodes += 1  
                        
                # 确保数据形状正确  
                if value_np.ndim == 0:  
                    value_np = np.array([value_np])  
                if log_prob_np.ndim == 0:  
                    log_prob_np = np.array([log_prob_np])  
                        
                # 存储数据  
                if self.use_lstm:  
                    episode_start = np.zeros(self.num_envs)  
                    for env_idx in range(self.num_envs):  
                        if dones[env_idx] or step == 0:  
                            episode_start[env_idx] = 1.0  
                            
                    self.buffer.add(  
                        observation=obs if obs.ndim == 2 else obs.reshape(self.num_envs, -1),  
                        action=action_np,  
                        reward=rewards,  
                        done=dones,  
                        value=value_np,  
                        log_prob=log_prob_np,  
                        hidden_state=hidden_state_np,  
                        episode_start=episode_start  
                    )  
                    
                    # 更新LSTM状态  
                    self.lstm_states = new_lstm_states  
                    
                    # 如果episode结束，重置对应环境的隐藏状态  
                    for env_idx in range(self.num_envs):  
                        if dones[env_idx]:  
                            self.lstm_states[0][:, env_idx] = 0  
                            self.lstm_states[1][:, env_idx] = 0  
                else:  
                    self.buffer.add(  
                        observation=obs if obs.ndim == 2 else obs.reshape(self.num_envs, -1),  
                        action=action_np,  
                        reward=rewards,  
                        done=dones,  
                        value=value_np,  
                        log_prob=log_prob_np  
                    )  
                    
                # 更新状态  
                obs = next_obs  
                self.last_obs = obs  
                self.last_dones = dones  
                self.num_timesteps += self.num_envs  
                # 聚合 infos 中的 reward components（支持单 env 和多 env）
                try:
                    info_list = infos if isinstance(infos, (list, tuple)) else [infos]
                    for env_idx, info in enumerate(info_list):
                        if not isinstance(info, dict):
                            continue
                        # 将可数值字段累积（忽略 episode/final_info 容器）
                        comps = {}
                        for k, v in info.items():
                            if k in ('episode', 'final_info'):
                                continue
                            try:
                                if np.isscalar(v):
                                    val = float(v)
                                    reward_comp_acc[k].append(val)
                                    comps[k] = val
                            except Exception:
                                continue
                        # 如果提供了外部 logger，则写入每步 CSV 记录（包含 env_id、step、global_timestep）
                        if self.logger is not None and comps:
                            try:
                                # 使用当前 self.num_timesteps 作为全局时间步参考
                                self.logger.log_reward_components_step(env_idx, step, int(self.num_timesteps), comps)
                            except Exception:
                                pass
                except Exception:
                    pass
                
        # 计算最后状态的价值（用于GAE）  
        with torch.no_grad():  
            if obs.ndim == 1:  
                obs = obs.reshape(1, -1)  
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)  
            
            if self.use_lstm:  
                done_tensor = torch.tensor(self.last_dones, dtype=torch.float32, device=self.device)  
                last_value = self.network.get_value(obs_tensor, self.lstm_states, done_tensor)  
            else:  
                last_value = self.network.get_value(obs_tensor)  
                
            last_value_np = last_value.cpu().numpy()  
            if last_value_np.ndim == 0:  
                last_value_np = np.array([last_value_np])  
                
        # 计算GAE  
        self.buffer.compute_returns_and_advantages(last_value_np, self.last_dones)  
        
        # 更新全局统计
        self.episode_rewards.extend(rollout_rewards)
        self.episode_lengths.extend(rollout_lengths)

        # 汇总 reward components 的均值（如果有收集到）
        reward_components_mean = {k: float(np.mean(vs)) for k, vs in reward_comp_acc.items() if len(vs) > 0}

        rollout_info = {
            'mean_reward': np.mean(rollout_rewards) if rollout_rewards else 0,
            'mean_length': np.mean(rollout_lengths) if rollout_lengths else 0,
            'num_episodes': len(rollout_rewards),
            'reward_components_mean': reward_components_mean
        }

        return rollout_info

      
    def train(self) -> Dict[str, float]:  
        """  
        执行一次PPO更新  
          
        Returns:  
            train_info: 训练统计信息  
        """  
        self.network.train()  
          
        # 统计  
        all_actor_losses = []  
        all_critic_losses = []  
        all_entropy_losses = []  
        all_kl_divs = []  
        all_clip_fractions = []  
          
        # 多个epoch  
        for epoch in range(self.n_epochs):  
            # 获取数据批次  
            if self.use_lstm:  
                data_generator = self.buffer.get_sequential(  
                    seq_length=self.lstm_seq_length,  
                    batch_size=self.batch_size  
                )  
            else:  
                data_generator = self.buffer.get(batch_size=self.batch_size)  
                  
            for batch in data_generator:  
                # 解包数据  
                obs = batch['observations']  
                actions = batch['actions']  
                old_values = batch['old_values']  
                old_log_probs = batch['old_log_probs']  
                advantages = batch['advantages']  
                returns = batch['returns']  
                  
                if self.use_lstm:  
                    dones = batch['dones']  
                    initial_hidden = batch['initial_hidden']  
                      
                    # 评估动作  
                    log_probs, values, entropies = self.network.evaluate_actions(  
                        obs, actions, initial_hidden, dones  
                    )  
                      
                    # 展平  
                    log_probs = log_probs.flatten()  
                    values = values.flatten()  
                    entropies = entropies.flatten()  
                    old_log_probs = old_log_probs.flatten()  
                    old_values = old_values.flatten()  
                    advantages = advantages.flatten()  
                    returns = returns.flatten()  
                else:  
                    # 评估动作  
                    action_dist, values = self.network(obs)  
                    log_probs = action_dist.log_prob(actions)  
                    entropies = action_dist.entropy()  
                    values = values.squeeze(-1)  
                      
                # 计算比率  
                log_ratio = log_probs - old_log_probs  
                ratio = torch.exp(log_ratio)  
                  
                # 计算KL散度（近似）  
                with torch.no_grad():  
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()  
                    all_kl_divs.append(approx_kl)  
                      
                    # 裁剪比例  
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()  
                    all_clip_fractions.append(clip_fraction)  
                      
                # KL早停  
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:  
                    if self.verbose >= 1:  
                        print(f"Early stopping at epoch {epoch} due to KL divergence: {approx_kl:.4f}")  
                    break  
                      
                # Actor损失（PPO-Clip）  
                actor_loss1 = advantages * ratio  
                actor_loss2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)  
                actor_loss = -torch.min(actor_loss1, actor_loss2).mean()  
                  
                # Critic损失  
                if self.clip_range_vf is not None:  
                    # 价值函数裁剪  
                    values_clipped = old_values + torch.clamp(  
                        values - old_values,   
                        -self.clip_range_vf,   
                        self.clip_range_vf  
                    )  
                    critic_loss1 = (values - returns) ** 2  
                    critic_loss2 = (values_clipped - returns) ** 2  
                    critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()  
                else:  
                    critic_loss = 0.5 * ((values - returns) ** 2).mean()  
                      
                # 熵损失  
                entropy_loss = -entropies.mean()  
                  
                # 总损失  
                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss  
                  
                # 优化  
                self.optimizer.zero_grad()  
                loss.backward()  
                  
                # 梯度裁剪  
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)  
                  
                self.optimizer.step()  
                  
                # 记录  
                all_actor_losses.append(actor_loss.item())  
                all_critic_losses.append(critic_loss.item())  
                all_entropy_losses.append(-entropy_loss.item())  
                  
            # 检查KL早停  
            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:  
                break  
                  
        return {  
            'actor_loss': np.mean(all_actor_losses),  
            'critic_loss': np.mean(all_critic_losses),  
            'entropy': np.mean(all_entropy_losses),  
            'approx_kl': np.mean(all_kl_divs),  
            'clip_fraction': np.mean(all_clip_fractions)  
        }  
      
    def learn(self,  
              total_timesteps: int,  
              callback: Optional[Callable] = None,  
              log_interval: int = 1,  
              eval_env = None,  
              eval_freq: int = 10000,  
              n_eval_episodes: int = 10,  
              save_freq: int = 50000,  
              save_path: str = "checkpoints") -> 'PPO':  
        """  
        训练智能体  
          
        Args:  
            total_timesteps: 总训练步数  
            callback: 回调函数  
            log_interval: 日志间隔（更新次数）  
            eval_env: 评估环境  
            eval_freq: 评估频率（步数）  
            n_eval_episodes: 评估episode数  
            save_freq: 保存频率（步数）  
            save_path: 保存路径  
              
        Returns:  
            self: 训练后的智能体  
        """  
        # 创建保存目录  
        os.makedirs(save_path, exist_ok=True)  
          
        num_updates = total_timesteps // (self.n_steps * self.num_envs)  
          
        start_time = time.time()  
        last_eval_step = 0  
        last_save_step = 0  
          
        if self.verbose >= 1:  
            print(f"Starting training for {total_timesteps} timesteps")  
            print(f"Number of updates: {num_updates}")  
            print(f"Device: {self.device}")  
              
        for update in range(1, num_updates + 1):  
            # 学习率衰减（线性）  
            frac = 1.0 - (update - 1.0) / num_updates  
            lr_now = frac * self.learning_rate  
            for param_group in self.optimizer.param_groups:  
                param_group['lr'] = lr_now  
                  
            # 收集数据  
            rollout_info = self.collect_rollouts()  
              
            # 训练  
            train_info = self.train()  
              
            # 日志  
            if self.writer is not None:  
                self.writer.add_scalar("rollout/mean_reward", rollout_info['mean_reward'], self.num_timesteps)  
                self.writer.add_scalar("rollout/mean_length", rollout_info['mean_length'], self.num_timesteps)  
                self.writer.add_scalar("train/actor_loss", train_info['actor_loss'], self.num_timesteps)  
                self.writer.add_scalar("train/critic_loss", train_info['critic_loss'], self.num_timesteps)  
                self.writer.add_scalar("train/entropy", train_info['entropy'], self.num_timesteps)  
                self.writer.add_scalar("train/approx_kl", train_info['approx_kl'], self.num_timesteps)  
                self.writer.add_scalar("train/clip_fraction", train_info['clip_fraction'], self.num_timesteps)  
                self.writer.add_scalar("train/learning_rate", lr_now, self.num_timesteps)  
                  
            # 控制台输出  
            if self.verbose >= 1 and update % log_interval == 0:  
                elapsed = time.time() - start_time  
                fps = self.num_timesteps / elapsed  
                  
                print(f"\n{'='*60}")  
                print(f"Update {update}/{num_updates}")  
                print(f"Timesteps: {self.num_timesteps}")  
                print(f"FPS: {fps:.1f}")  
                print(f"Time elapsed: {elapsed:.1f}s")  
                print(f"Mean reward: {rollout_info['mean_reward']:.2f}")  
                print(f"Mean length: {rollout_info['mean_length']:.1f}")  
                print(f"Actor loss: {train_info['actor_loss']:.4f}")  
                print(f"Critic loss: {train_info['critic_loss']:.4f}")  
                print(f"Entropy: {train_info['entropy']:.4f}")  
                print(f"KL div: {train_info['approx_kl']:.4f}")  
                print(f"Clip fraction: {train_info['clip_fraction']:.2%}")  
                print(f"Learning rate: {lr_now:.2e}")  
                  
            # 评估  
            if eval_env is not None and self.num_timesteps - last_eval_step >= eval_freq:  
                eval_info = self.evaluate(eval_env, n_eval_episodes)  
                last_eval_step = self.num_timesteps  
                  
                if self.writer is not None:  
                    self.writer.add_scalar("eval/mean_reward", eval_info['mean_reward'], self.num_timesteps)  
                    self.writer.add_scalar("eval/success_rate", eval_info['success_rate'], self.num_timesteps)  
                      
                if self.verbose >= 1:  
                    print(f"\n[Evaluation] Mean reward: {eval_info['mean_reward']:.2f}, "  
                          f"Success rate: {eval_info['success_rate']:.2%}")  
                      
            # 保存  
            if self.num_timesteps - last_save_step >= save_freq:  
                save_file = os.path.join(save_path, f"ppo_{self.num_timesteps}.pth")  
                self.save(save_file)  
                last_save_step = self.num_timesteps  
                  
                if self.verbose >= 1:  
                    print(f"Model saved to {save_file}")  
                      
            # 回调  
            if callback is not None:  
                callback(locals())  
                  
        # 最终保存  
        save_file = os.path.join(save_path, "ppo_final.pth")  
        self.save(save_file)  
          
        if self.verbose >= 1:  
            print(f"\nTraining completed!")  
            print(f"Total timesteps: {self.num_timesteps}")  
            print(f"Total episodes: {self.num_episodes}")  
            print(f"Final model saved to {save_file}")  
              
        return self  
      
    def evaluate(self,   
                env,  
                n_episodes: int = 10,  
                deterministic: bool = True) -> Dict[str, float]:  
        """  
        评估策略  
        
        Args:  
            env: 评估环境  
            n_episodes: 评估episode数  
            deterministic: 是否使用确定性策略  
            
        Returns:  
            eval_info: 评估结果  
        """  
        self.network.eval()  
        
        episode_rewards = []  
        episode_lengths = []  
        successes = []  
        # 用于收集评估期间每个episode的 reward_components_mean
        reward_comp_records = defaultdict(list)
        
        # 获取环境数量  
        num_envs = getattr(env, 'num_envs', 1)  
        is_single_env = num_envs == 1 and not hasattr(env, 'envs')  # 检查是否是真正的单环境  
        
        # 重置  
        obs, _ = env.reset()  
        
        # 处理单环境观测  
        if is_single_env and isinstance(obs, np.ndarray) and obs.ndim == 1:  
            obs = obs.reshape(1, -1)  
        
        if self.use_lstm:  
            lstm_states = self.network.lstm.get_initial_state(num_envs, self.device)  
            dones = np.zeros(num_envs, dtype=np.float32)  
            
        current_rewards = np.zeros(num_envs)  
        current_lengths = np.zeros(num_envs)  
        
        while len(episode_rewards) < n_episodes:  
            with torch.no_grad():  
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)  
                
                # 确保观测是2D  
                if obs_tensor.dim() == 1:  
                    obs_tensor = obs_tensor.unsqueeze(0)  
                
                if self.use_lstm:  
                    done_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)  
                    action_dist, _, lstm_states = self.network(obs_tensor, lstm_states, done_tensor)  
                else:  
                    action_dist, _ = self.network(obs_tensor)  
                    
                if deterministic:  
                    action = action_dist.probs.argmax(dim=-1)  
                else:  
                    action = action_dist.sample()  
                    
                action_np = action.cpu().numpy()  
                
            # 处理单环境的动作格式  
            if is_single_env:  
                # 单环境期望标量动作  
                action_to_env = action_np.item() if action_np.ndim > 0 else int(action_np)  
            else:  
                action_to_env = action_np  
                
            # 执行动作  
            step_result = env.step(action_to_env)  
            
            # 处理返回值  
            if len(step_result) == 5:  
                next_obs, rewards, terminateds, truncateds, infos = step_result  
            else:  
                # 旧版本gym  
                next_obs, rewards, dones_step, infos = step_result  
                terminateds = dones_step  
                truncateds = np.zeros_like(terminateds) if isinstance(terminateds, np.ndarray) else False  
            
            # 处理单环境返回值，转换为数组格式  
            if is_single_env:  
                if not isinstance(rewards, np.ndarray):  
                    rewards = np.array([rewards])  
                if not isinstance(terminateds, np.ndarray):  
                    terminateds = np.array([terminateds])  
                if not isinstance(truncateds, np.ndarray):  
                    truncateds = np.array([truncateds])  
                if isinstance(next_obs, np.ndarray) and next_obs.ndim == 1:  
                    next_obs = next_obs.reshape(1, -1)  
                # 将info包装成列表  
                if not isinstance(infos, list):  
                    infos = [infos]  
                    
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)  
            
            current_rewards += rewards  
            current_lengths += 1  
            
            # 检查完成的episode  
            for env_idx in range(num_envs):  
                if dones[env_idx]:  
                    episode_rewards.append(current_rewards[env_idx])  
                    episode_lengths.append(current_lengths[env_idx])  
                    
                    # 检查是否成功  
                    if isinstance(infos, list) and len(infos) > env_idx:  
                        info = infos[env_idx]  
                        if isinstance(info, dict):  
                            if 'final_info' in info:  
                                success = info['final_info'].get('episode', {}).get('success', False)  
                            elif 'episode' in info:  
                                success = info['episode'].get('success', False)  
                            else:  
                                success = info.get('success', False)  
                        else:  
                            success = False  
                        # 如果包含 reward_components_mean，则收集用于总体统计
                        try:
                            ep_info = None
                            if isinstance(info, dict):
                                if 'final_info' in info:
                                    ep_info = info['final_info'].get('episode', None)
                                elif 'episode' in info:
                                    ep_info = info.get('episode', None)
                                else:
                                    ep_info = info.get('episode', None)
                            if isinstance(ep_info, dict):
                                rcm = ep_info.get('reward_components_mean', None)
                                if isinstance(rcm, dict):
                                    for k, v in rcm.items():
                                        # 仅记录数值类型
                                        try:
                                            if np.isscalar(v) or isinstance(v, (int, float)):
                                                reward_comp_records[k].append(float(v))
                                        except Exception:
                                            continue
                        except Exception:
                            pass
                    else:  
                        success = False  
                    successes.append(success)  
                    
                    current_rewards[env_idx] = 0  
                    current_lengths[env_idx] = 0  
                    
                    if self.use_lstm:  
                        lstm_states[0][:, env_idx] = 0  
                        lstm_states[1][:, env_idx] = 0  
                        
                    # 如果达到目标episode数，停止  
                    if len(episode_rewards) >= n_episodes:  
                        break  
                        
            obs = next_obs  
            
        # 汇总 reward component 均值
        reward_components_mean = {k: float(np.mean(vs)) for k, vs in reward_comp_records.items() if len(vs) > 0}

        return {  
            'mean_reward': np.mean(episode_rewards[:n_episodes]),  
            'std_reward': np.std(episode_rewards[:n_episodes]),  
            'mean_length': np.mean(episode_lengths[:n_episodes]),  
            'success_rate': np.mean(successes[:n_episodes]) if successes else 0,
            'reward_components_mean': reward_components_mean
        }  

      
    def predict(self,  
                observation: np.ndarray,  
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
                done: Optional[np.ndarray] = None,  
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[torch.Tensor, torch.Tensor]]]:  
        """  
        预测动作  
          
        Args:  
            observation: 观测  
            hidden_state: LSTM隐藏状态  
            done: 终止标志  
            deterministic: 是否确定性  
              
        Returns:  
            action: 动作  
            hidden_state: 新的隐藏状态  
        """  
        self.network.eval()  
          
        with torch.no_grad():  
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)  
              
            if obs_tensor.dim() == 1:  
                obs_tensor = obs_tensor.unsqueeze(0)  
                  
            if self.use_lstm:  
                if hidden_state is None:  
                    hidden_state = self.network.lstm.get_initial_state(obs_tensor.shape[0], self.device)  
                      
                if done is not None:  
                    done_tensor = torch.tensor(done, dtype=torch.float32, device=self.device)  
                else:  
                    done_tensor = None  
                      
                action_dist, _, new_hidden = self.network(obs_tensor, hidden_state, done_tensor)  
            else:  
                action_dist, _ = self.network(obs_tensor)  
                new_hidden = None  
                  
            if deterministic:  
                action = action_dist.probs.argmax(dim=-1)  
            else:  
                action = action_dist.sample()  
                  
        return action.cpu().numpy(), new_hidden  
      
    def save(self, path: str):  
        """保存模型"""  
        torch.save({  
            'network_state_dict': self.network.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict(),  
            'num_timesteps': self.num_timesteps,  
            'num_episodes': self.num_episodes,  
            'config': {  
                'learning_rate': self.learning_rate,  
                'n_steps': self.n_steps,  
                'batch_size': self.batch_size,  
                'n_epochs': self.n_epochs,  
                'gamma': self.gamma,  
                'gae_lambda': self.gae_lambda,  
                'clip_range': self.clip_range,  
                'ent_coef': self.ent_coef,  
                'vf_coef': self.vf_coef,  
                'use_lstm': self.use_lstm  
            }  
        }, path)  
          
    def load(self, path: str):  
        """加载模型"""  
        checkpoint = torch.load(path, map_location=self.device)  
        self.network.load_state_dict(checkpoint['network_state_dict'])  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        self.num_timesteps = checkpoint['num_timesteps']  
        self.num_episodes = checkpoint['num_episodes']  
          
        if self.verbose >= 1:  
            print(f"Model loaded from {path}")  
            print(f"Timesteps: {self.num_timesteps}, Episodes: {self.num_episodes}")  
  
  
class PPOTrainer:  
    """  
    PPO训练器  
      
    提供更高级的训练接口，支持课程学习、早停等  
    """  
      
    def __init__(self,  
                 env_fn: Callable,  
                 config: Dict,  
                 device: torch.device = None):  
        """  
        初始化训练器  
          
        Args:  
            env_fn: 环境创建函数  
            config: 配置字典  
            device: 计算设备  
        """  
        self.env_fn = env_fn  
        self.config = config  
          
        if device is None:  
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        else:  
            self.device = device  
              
        # 创建环境  
        self.env = env_fn()  
          
        # 获取观测和动作维度  
        obs_dim = self.env.observation_space.shape[0]  
        num_actions = self.env.action_space.n  
          
        # 创建网络  
        network_config = config.get('network', {})  
        self.network = PPOLSTMNetwork(  
            obs_dim=obs_dim,  
            num_actions=num_actions,  
            feature_hidden_sizes=network_config.get('feature_hidden_sizes', [256, 256]),  
            lstm_hidden_size=network_config.get('lstm_hidden_size', 256),  
            lstm_num_layers=network_config.get('lstm_num_layers', 2),  
            actor_hidden_sizes=network_config.get('actor_hidden_sizes', [128]),  
            critic_hidden_sizes=network_config.get('critic_hidden_sizes', [128])  
        )  
          
        # 创建PPO算法  
        training_config = config.get('training', {})  
        self.ppo = PPO(  
            env=self.env,  
            network=self.network,  
            learning_rate=training_config.get('learning_rate', 3e-4),  
            n_steps=training_config.get('n_steps', 2048),  
            batch_size=training_config.get('batch_size', 64),  
            n_epochs=training_config.get('n_epochs', 10),  
            gamma=training_config.get('gamma', 0.99),  
            gae_lambda=training_config.get('gae_lambda', 0.95),  
            clip_range=training_config.get('clip_range', 0.2),  
            ent_coef=training_config.get('ent_coef', 0.01),  
            vf_coef=training_config.get('vf_coef', 0.5),  
            max_grad_norm=training_config.get('max_grad_norm', 0.5),  
            device=self.device,  
            tensorboard_log=config.get('tensorboard_log'),  
            verbose=config.get('verbose', 1)  
        )  
          
        # 课程学习  
        self.curriculum_config = config.get('curriculum', {})  
        self.current_stage = 0  
          
        # 最佳性能追踪  
        self.best_reward = float('-inf')  
        self.patience = 0  
        self.max_patience = config.get('early_stop_patience', 50)  
          
    def train(self,  
              total_timesteps: int,  
              eval_env_fn: Optional[Callable] = None,  
              save_path: str = "checkpoints"):  
        """  
        执行训练  
          
        Args:  
            total_timesteps: 总训练步数  
            eval_env_fn: 评估环境创建函数  
            save_path: 保存路径  
        """  
        # 创建评估环境  
        eval_env = eval_env_fn() if eval_env_fn is not None else None  
          
        # 回调函数：处理课程学习和早停  
        def callback(locals_dict):  
            nonlocal self  
              
            # 检查是否需要更新课程  
            if self.curriculum_config.get('enabled', False):  
                self._update_curriculum(locals_dict)  
                  
            # 早停检查  
            rollout_info = locals_dict.get('rollout_info', {})  
            mean_reward = rollout_info.get('mean_reward', 0)  
              
            if mean_reward > self.best_reward:  
                self.best_reward = mean_reward  
                self.patience = 0  
                  
                # 保存最佳模型  
                best_path = os.path.join(save_path, "ppo_best.pth")  
                self.ppo.save(best_path)  
            else:  
                self.patience += 1  
                  
            # if self.patience >= self.max_patience:  
            #     print(f"Early stopping: no improvement for {self.patience} updates")  
            #     return False  
                  
            return True  
              
        # 开始训练  
        self.ppo.learn(  
            total_timesteps=total_timesteps,  
            callback=callback,  
            eval_env=eval_env,  
            eval_freq=self.config.get('eval_freq', 10000),  
            n_eval_episodes=self.config.get('eval_episodes', 10),  
            save_freq=self.config.get('save_freq', 50000),  
            save_path=save_path  
        )  
          
        # 关闭环境  
        self.env.close()  
        if eval_env is not None:  
            eval_env.close()  
              
    def _update_curriculum(self, locals_dict):  
        """更新课程学习阶段"""  
        stages = self.curriculum_config.get('stages', [])  
        if self.current_stage >= len(stages):  
            return  
              
        current_stage_config = stages[self.current_stage]  
        success_threshold = current_stage_config.get('success_threshold', 0.8)  
          
        # 获取成功率  
        rollout_info = locals_dict.get('rollout_info', {})  
        # 这里需要实际的成功率统计，简化处理  
        mean_reward = rollout_info.get('mean_reward', 0)  
          
        # 简单的进阶条件：平均奖励达到阈值  
        reward_threshold = current_stage_config.get('reward_threshold', 50)  
          
        if mean_reward >= reward_threshold:  
            self.current_stage += 1  
              
            if self.current_stage < len(stages):  
                print(f"\n{'='*60}")  
                print(f"Curriculum: Advancing to stage {self.current_stage}")  
                print(f"New stage: {stages[self.current_stage].get('name', 'unknown')}")  
                print(f"{'='*60}\n")  
                  
                # 更新环境的课程阶段  
                if hasattr(self.env, 'set_curriculum_stage'):  
                    self.env.set_curriculum_stage(self.current_stage)  
