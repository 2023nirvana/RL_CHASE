"""  
底层控制器训练脚本  
==================  
  
训练AUV底层运动控制器，使其能够执行基础运动技能。  
  
使用方法:  
    python training/train_low_level.py --config configs/low_level_config.yaml  
  
    # 指定课程阶段  
    python training/train_low_level.py --curriculum-stage 0  
  
    # 从检查点恢复  
    python training/train_low_level.py --resume checkpoints/ppo_100000.pth  
  
    # 快速测试  
    python training/train_low_level.py --test  
"""  
  
import os  
import sys  
import argparse  
import yaml  
import time  
import json  
from datetime import datetime  
from typing import Dict, Optional, Callable  
import numpy as np  
import torch  
  
# 添加项目根目录  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, PROJECT_ROOT)  
  
from core.environment.low_level_env import LowLevelAUVEnv, VecLowLevelAUVEnv  
from core.environment.low_level_env_2d import LowLevelAUV2DEnv
from networks.ppo_lstm_network import PPOLSTMNetwork  
from algorithms.ppo import PPO  
from utils.logger import Logger, MetricsTracker  
from utils.visualizer import TrainingVisualizer  
from utils.video_recorder import VideoRecorder  
  
  
class LowLevelTrainer:  
    """  
    底层控制器训练器  
      
    整合所有训练组件  
    """  
      
    def __init__(self, config: Dict, args: argparse.Namespace):  
        """  
        初始化训练器  
        
        Args:  
            config: 配置字典  
            args: 命令行参数  
        """  
        self.config = config  
        self.args = args  
        
        # 设置设备  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        print(f"Using device: {self.device}")  
        
        # 创建实验目录  
        self.experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")  
        self.log_dir = os.path.join(PROJECT_ROOT, "visualization/logs", self.experiment_name)  
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", self.experiment_name)  
        self.video_dir = os.path.join(PROJECT_ROOT, "visualization/videos", self.experiment_name)  
        
        os.makedirs(self.log_dir, exist_ok=True)  
        os.makedirs(self.checkpoint_dir, exist_ok=True)  
        os.makedirs(self.video_dir, exist_ok=True)  
        
        # 保存配置  
        self._save_config()  
        
        # 课程学习状态（必须在_init_environments之前初始化）  
        self.current_curriculum_stage = args.curriculum_stage  
        self.curriculum_history = []  
        
        # 训练统计  
        self.best_reward = float('-inf')  
        self.no_improvement_count = 0  
        
        # 初始化组件（顺序很重要）  
        self._init_environments()  
        self._init_network()  
        self._init_algorithm()
        self._init_logging()
        # 将 Logger 注入到 PPO（使其可以写 CSV 等外部记录）
        if hasattr(self, 'ppo') and hasattr(self, 'logger'):
            try:
                self.ppo.logger = self.logger
            except Exception:
                pass

          
    def _save_config(self):  
        """保存配置到实验目录"""  
        config_path = os.path.join(self.log_dir, "config.yaml")  
        with open(config_path, 'w') as f:  
            yaml.dump(self.config, f, default_flow_style=False)  
              
        # 保存命令行参数  
        args_path = os.path.join(self.log_dir, "args.json")  
        with open(args_path, 'w') as f:  
            json.dump(vars(self.args), f, indent=2)  
              
        print(f"Config saved to {self.log_dir}")  
          
    def _init_environments(self):  
        """初始化训练和评估环境"""  
        training_config = self.config['training']  
          
        # 训练环境（向量化）  
        self.num_envs = training_config.get('n_envs', 8)  
        # 支持可选的2D简化模式（CLI/配置中开启）
        use_2d = self.args.use_2d or training_config.get('use_2d', False)

        if use_2d:
            # 对于2D简化模式，默认增加每次更新收集的步数，使得每次rollout包含更多交互数据
            # 这是为了在保留控制器完整性的同时，给策略更多时间与环境交互以学习位移效果
            orig_n_steps = training_config.get('n_steps', 2048)
            training_config['n_steps'] = int(max(orig_n_steps * 2, orig_n_steps))
            print(f"2D mode enabled: increasing n_steps {orig_n_steps} -> {training_config['n_steps']}")
            # 手动构造向量化环境（轻量，不依赖外部Vec封装）
            envs = []
            for i in range(self.num_envs):
                seed_i = self.args.seed + i if self.args.seed else None
                env = LowLevelAUV2DEnv(
                    config_path=self.args.config,
                    curriculum_stage=self.current_curriculum_stage,
                    enable_domain_randomization=self.config.get('domain_randomization', {}).get('enabled', True),
                    seed=seed_i,
                    render_mode=None
                )
                envs.append(env)

            class SimpleVecEnv:
                def __init__(self, env_list):
                    self.envs = env_list
                    self.num_envs = len(env_list)
                    self.observation_space = env_list[0].observation_space
                    self.action_space = env_list[0].action_space

                def reset(self, seed: Optional[int] = None):
                    obs = []
                    infos = []
                    for i, e in enumerate(self.envs):
                        s = seed + i if seed is not None else None
                        o, info = e.reset(seed=s)
                        obs.append(o)
                        infos.append(info)
                    return np.array(obs), infos

                def step(self, actions):
                    observations, rewards, terminateds, truncateds, infos = [], [], [], [], []
                    if isinstance(actions, np.ndarray):
                        actions = actions.flatten()
                    for e, a in zip(self.envs, actions):
                        o, r, term, trunc, info = e.step(int(a))
                        observations.append(o)
                        rewards.append(r)
                        terminateds.append(term)
                        truncateds.append(trunc)
                        infos.append(info)
                    return np.array(observations), np.array(rewards), np.array(terminateds), np.array(truncateds), infos

                def set_curriculum_stage(self, stage: int):
                    for e in self.envs:
                        e.set_curriculum_stage(stage)

                def close(self):
                    for e in self.envs:
                        e.close()

            self.train_env = SimpleVecEnv(envs)
        else:
            self.train_env = VecLowLevelAUVEnv(
                num_envs=self.num_envs,  
                curriculum_stage=self.current_curriculum_stage,  
                enable_domain_randomization=self.config.get('domain_randomization', {}).get('enabled', True),  
                seeds=[self.args.seed + i if self.args.seed else None for i in range(self.num_envs)]  
            )
          
        # 评估环境（单个，确定性）  
        # 评估环境：与训练模式一致（2D时也使用2D环境）
        if use_2d:
            self.eval_env = LowLevelAUV2DEnv(
                curriculum_stage=self.current_curriculum_stage,
                enable_domain_randomization=False,
                seed=self.args.seed + 1000 if self.args.seed else None,
                render_mode=None
            )
        else:
            self.eval_env = LowLevelAUVEnv(  
                curriculum_stage=self.current_curriculum_stage,  
                enable_domain_randomization=False,  
                seed=self.args.seed + 1000 if self.args.seed else None  
            )
          
        # 获取空间信息  
        self.obs_dim = self.train_env.observation_space.shape[0]  
        self.num_actions = self.train_env.action_space.n  
          
        print(f"Training environments: {self.num_envs}")  
        print(f"Observation dim: {self.obs_dim}")  
        print(f"Action dim: {self.num_actions}")  
          
    def _init_network(self):  
        """初始化神经网络"""  
        network_config = self.config.get('network', {})  
          
        self.network = PPOLSTMNetwork(  
            obs_dim=self.obs_dim,  
            num_actions=self.num_actions,  
            feature_hidden_sizes=network_config.get('feature_extractor', {}).get('hidden_sizes', [256, 256]),  
            lstm_hidden_size=network_config.get('lstm', {}).get('hidden_size', 256),  
            lstm_num_layers=network_config.get('lstm', {}).get('num_layers', 2),  
            actor_hidden_sizes=network_config.get('actor_head', {}).get('hidden_sizes', [128]),  
            critic_hidden_sizes=network_config.get('critic_head', {}).get('hidden_sizes', [128]),  
            encoder_hidden_sizes=network_config.get('encoder', {}).get('hidden_sizes'),  
            decoder_hidden_sizes=network_config.get('decoder', {}).get('hidden_sizes'),  
            use_layer_norm=network_config.get('feature_extractor', {}).get('use_layer_norm', True)  
        )  
          
        # 打印网络信息  
        total_params = sum(p.numel() for p in self.network.parameters())  
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)  
        print(f"Network parameters: {total_params:,} (trainable: {trainable_params:,})")  
          
        # 如果需要恢复  
        if self.args.resume:  
            self._load_checkpoint(self.args.resume)  
              
    def _init_algorithm(self):  
        """初始化PPO算法"""  
        training_config = self.config['training']  
          
        self.ppo = PPO(  
            env=self.train_env,  
            network=self.network,  
            learning_rate=training_config.get('learning_rate', 3e-4),  
            n_steps=training_config.get('n_steps', 2048),  
            batch_size=training_config.get('batch_size', 64),  
            n_epochs=training_config.get('n_epochs', 10),  
            gamma=training_config.get('gamma', 0.99),  
            gae_lambda=training_config.get('gae_lambda', 0.95),  
            clip_range=training_config.get('clip_range', 0.2),  
            clip_range_vf=training_config.get('clip_range_vf'),  
            ent_coef=training_config.get('ent_coef', 0.01),  
            vf_coef=training_config.get('vf_coef', 0.5),  
            max_grad_norm=training_config.get('max_grad_norm', 0.5),  
            target_kl=training_config.get('target_kl'),  
            use_lstm=True,  
            lstm_seq_length=16,  
            device=self.device,  
            tensorboard_log=self.log_dir,  
            verbose=1  
        )  
          
    def _init_logging(self):  
        """初始化日志记录"""  
        self.logger = Logger(  
            log_dir=self.log_dir,  
            experiment_name="",  # 已经在log_dir中  
            use_tensorboard=True,  
            use_csv=True,  
            config=self.config  
        )  
          
        self.metrics_tracker = MetricsTracker(window_size=100)  
        self.visualizer = TrainingVisualizer(save_dir=os.path.join(self.log_dir, "plots"))  
        self.video_recorder = VideoRecorder(save_dir=self.video_dir)  
          
    def _load_checkpoint(self, checkpoint_path: str):  
        """加载检查点"""  
        print(f"Loading checkpoint from {checkpoint_path}")  
        self.ppo.load(checkpoint_path)  
          
    def train(self, total_timesteps: Optional[int] = None):  
        """  
        执行训练  
          
        Args:  
            total_timesteps: 总训练步数（如果为None则使用配置）  
        """  
        if total_timesteps is None:  
            total_timesteps = self.config['training'].get('total_timesteps', 1000000)  
              
        training_config = self.config['training']  
        curriculum_config = self.config.get('curriculum', {})  
          
        # 计算更新次数  
        steps_per_update = training_config.get('n_steps', 2048) * self.num_envs  
        num_updates = total_timesteps // steps_per_update  
          
        print(f"\n{'='*60}")  
        print(f"Starting training")  
        print(f"Total timesteps: {total_timesteps:,}")  
        print(f"Steps per update: {steps_per_update:,}")  
        print(f"Number of updates: {num_updates:,}")  
        print(f"{'='*60}\n")  
          
        start_time = time.time()  
          
        # 评估和保存频率  
        eval_freq = training_config.get('eval_freq', 10000)  
        save_freq = training_config.get('save_freq', 50000)  
        video_freq = self.config.get('logging', {}).get('video', {}).get('record_freq', 100000)  
          
        last_eval_step = 0  
        last_save_step = 0  
        last_video_step = 0  
          
        for update in range(1, num_updates + 1):  
            update_start = time.time()  
              
            # 学习率衰减  
            frac = 1.0 - (update - 1.0) / num_updates  
            lr_now = frac * training_config.get('learning_rate', 3e-4)  
              
            if training_config.get('lr_schedule', 'linear') == 'linear':  
                lr_end = training_config.get('lr_end', 1e-5)  
                lr_now = max(lr_now, lr_end)  
                  
            for param_group in self.ppo.optimizer.param_groups:  
                param_group['lr'] = lr_now  
                  
            # 收集数据  
            rollout_info = self.ppo.collect_rollouts()  
              
            # 训练  
            train_info = self.ppo.train()  
              
            # 更新指标  
            self.metrics_tracker.add('episode_reward', rollout_info['mean_reward'])  
            self.metrics_tracker.add('episode_length', rollout_info['mean_length'])  
            self.metrics_tracker.add('actor_loss', train_info['actor_loss'])  
            self.metrics_tracker.add('critic_loss', train_info['critic_loss'])  
            self.metrics_tracker.add('entropy', train_info['entropy'])  
              
            # 记录到可视化器  
            self.visualizer.add_scalar('episode/reward', rollout_info['mean_reward'], self.ppo.num_timesteps)  
            self.visualizer.add_scalar('episode/length', rollout_info['mean_length'], self.ppo.num_timesteps)  
            self.visualizer.add_scalar('train/actor_loss', train_info['actor_loss'], self.ppo.num_timesteps)  
            self.visualizer.add_scalar('train/critic_loss', train_info['critic_loss'], self.ppo.num_timesteps)  
            self.visualizer.add_scalar('train/entropy', train_info['entropy'], self.ppo.num_timesteps)  
            self.visualizer.add_scalar('train/learning_rate', lr_now, self.ppo.num_timesteps)  
            # 如果 collect_rollouts 返回了 reward_components_mean，则记录到 visualizer 和 logger
            rcm = rollout_info.get('reward_components_mean', {})
            if isinstance(rcm, dict) and rcm:
                try:
                    # 可视化器（plots）
                    for k, v in rcm.items():
                        self.visualizer.add_scalar(f'train/reward_comp/{k}', float(v), self.ppo.num_timesteps)
                    # Logger 的 TensorBoard（写成一组 scalars）
                    self.logger.log_scalars('train/reward_comp', {k: float(v) for k, v in rcm.items()}, self.ppo.num_timesteps)
                except Exception:
                    pass
              
            # 检查课程学习进阶  
            if curriculum_config.get('enabled', False):  
                self._check_curriculum_advancement(rollout_info)  
                  
            # 评估  
            if self.ppo.num_timesteps - last_eval_step >= eval_freq:  
                eval_info = self._evaluate()  
                last_eval_step = self.ppo.num_timesteps  
                  
                # 检查是否是最佳模型  
                if eval_info['mean_reward'] > self.best_reward:  
                    self.best_reward = eval_info['mean_reward']  
                    self.no_improvement_count = 0  
                      
                    # 保存最佳模型  
                    best_path = os.path.join(self.checkpoint_dir, "best_model.pth")  
                    self.ppo.save(best_path)  
                    print(f"New best model saved! Reward: {self.best_reward:.2f}")  
                else:  
                    self.no_improvement_count += 1  
                      
            # 保存检查点  
            if self.ppo.num_timesteps - last_save_step >= save_freq:  
                checkpoint_path = os.path.join(  
                    self.checkpoint_dir,   
                    f"checkpoint_{self.ppo.num_timesteps}.pth"  
                )  
                self.ppo.save(checkpoint_path)  
                last_save_step = self.ppo.num_timesteps  
                  
            # 录制视频  
            if self.ppo.num_timesteps - last_video_step >= video_freq:  
                self._record_video()  
                last_video_step = self.ppo.num_timesteps  
                  
            # 打印进度  
            if update % 10 == 0:  
                update_time = time.time() - update_start  
                total_time = time.time() - start_time  
                fps = self.ppo.num_timesteps / total_time  
                  
                print(f"\nUpdate {update}/{num_updates}")  
                print(f"  Timesteps: {self.ppo.num_timesteps:,}")  
                print(f"  FPS: {fps:.1f}")  
                print(f"  Mean Reward: {rollout_info['mean_reward']:.2f}")  
                print(f"  Mean Length: {rollout_info['mean_length']:.1f}")  
                print(f"  Actor Loss: {train_info['actor_loss']:.4f}")  
                print(f"  Critic Loss: {train_info['critic_loss']:.4f}")  
                print(f"  Entropy: {train_info['entropy']:.4f}")  
                print(f"  KL Div: {train_info['approx_kl']:.4f}")  
                print(f"  LR: {lr_now:.2e}")  
                print(f"  Curriculum Stage: {self.current_curriculum_stage}")  
                  
            # 早停检查  
            # early_stop_patience = training_config.get('early_stop_patience', 50)  
            # if self.no_improvement_count >= early_stop_patience:  
            #     print(f"\nEarly stopping: no improvement for {self.no_improvement_count} evaluations")  
            #     break  
                  
        # 训练结束  
        total_time = time.time() - start_time  
          
        print(f"\n{'='*60}")  
        print(f"Training completed!")  
        print(f"Total time: {total_time/3600:.2f} hours")  
        print(f"Total timesteps: {self.ppo.num_timesteps:,}")  
        print(f"Total episodes: {self.ppo.num_episodes:,}")  
        print(f"Best reward: {self.best_reward:.2f}")  
        print(f"{'='*60}")  
          
        # 保存最终模型  
        final_path = os.path.join(self.checkpoint_dir, "final_model.pth")  
        self.ppo.save(final_path)  
          
        # 生成训练报告  
        self._generate_report()  
          
        # 关闭  
        self.close()  
          
    def _evaluate(self) -> Dict:  
        """  
        评估当前策略  
          
        Returns:  
            eval_info: 评估结果  
        """  
        print("\n[Evaluation]")  
          
        eval_episodes = self.config['training'].get('eval_episodes', 20)  
        eval_info = self.ppo.evaluate(self.eval_env, n_episodes=eval_episodes, deterministic=True)  
          
        # 记录  
        # logger.log_evaluation expects scalar values; 如果 eval_info 中包含嵌套字典（如 reward_components_mean），
        # 请先过滤出标量项，再单独记录复合项。  
        eval_info_scalars = {k: v for k, v in eval_info.items() if np.isscalar(v)}
        self.logger.log_evaluation(eval_info_scalars, self.ppo.num_timesteps)  
        self.visualizer.add_scalar('eval/mean_reward', eval_info['mean_reward'], self.ppo.num_timesteps)  
        self.visualizer.add_scalar('eval/success_rate', eval_info['success_rate'], self.ppo.num_timesteps)  
          
        print(f"  Mean Reward: {eval_info['mean_reward']:.2f} ± {eval_info['std_reward']:.2f}")  
        print(f"  Mean Length: {eval_info['mean_length']:.1f}")  
        print(f"  Success Rate: {eval_info['success_rate']:.2%}")  
          
        # 如果包含 reward_components_mean，则记录到可视化器（便于在 TensorBoard/plots 中查看）
        rcm = eval_info.get('reward_components_mean', {})
        if isinstance(rcm, dict) and rcm:
            for k, v in rcm.items():
                try:
                    self.visualizer.add_scalar(f'eval/reward_comp/{k}', float(v), self.ppo.num_timesteps)
                except Exception:
                    continue
            # 同步写入 Logger 的 TensorBoard（以便在 TB 中查看这些分量）
            try:
                # logger.log_scalars 接受 main_tag 和字典
                self.logger.log_scalars('eval/reward_comp', {k: float(v) for k, v in rcm.items()}, self.ppo.num_timesteps)
            except Exception:
                pass

        return eval_info  
          
    def _check_curriculum_advancement(self, rollout_info: Dict):  
        """  
        检查是否需要进阶课程  
          
        Args:  
            rollout_info: rollout统计信息  
        """  
        curriculum_config = self.config.get('curriculum', {})  
        stages = curriculum_config.get('stages', [])  
          
        if self.current_curriculum_stage >= len(stages):  
            return  
              
        current_stage = stages[self.current_curriculum_stage]  
        success_threshold = current_stage.get('success_threshold', 0.8)  
          
        # 获取最近的成功率  
        recent_rewards = self.metrics_tracker.metrics.get('episode_reward', [])  
        if len(recent_rewards) < 50:  
            return  
              
        # 简化的进阶条件：基于奖励  
        # 实际应该基于成功率  
        mean_recent_reward = np.mean(recent_rewards[-50:])  
        reward_threshold = current_stage.get('reward_threshold', 0)  
          
        # 如果没有设置reward_threshold，根据阶段估计  
        if reward_threshold == 0:  
            reward_threshold = 20 * (self.current_curriculum_stage + 1)  
              
        if mean_recent_reward >= reward_threshold:  
            self.current_curriculum_stage += 1  
              
            if self.current_curriculum_stage < len(stages):  
                new_stage = stages[self.current_curriculum_stage]  
                  
                print(f"\n{'='*60}")  
                print(f"CURRICULUM ADVANCEMENT!")  
                print(f"Stage {self.current_curriculum_stage - 1} -> {self.current_curriculum_stage}")  
                print(f"New stage: {new_stage.get('name', 'unknown')}")  
                print(f"{'='*60}\n")  
                  
                # 更新环境  
                self.train_env.set_curriculum_stage(self.current_curriculum_stage)  
                self.eval_env.set_curriculum_stage(self.current_curriculum_stage)  
                  
                # 记录  
                self.curriculum_history.append({  
                    'timestep': self.ppo.num_timesteps,  
                    'stage': self.current_curriculum_stage,  
                    'reward': mean_recent_reward  
                })  
                  
    def _record_video(self):  
        """录制评估视频"""  
        print("\n[Recording video]")  
          
        self.video_recorder.start_recording()  
          
        obs, _ = self.eval_env.reset()  
        hidden_state = None  
        done = False  
          
        max_steps = 500  
        step = 0  
          
        while not done and step < max_steps:  
            # 获取动作  
            action, hidden_state = self.ppo.predict(obs, hidden_state, deterministic=True)  
              
            # 执行  
            obs, reward, terminated, truncated, info = self.eval_env.step(action[0])  
            done = terminated or truncated  
              
            # 记录帧  
            state = self.eval_env.dynamics.get_state()  
            self.video_recorder.add_frame({  
                'position': state['position'],  
                'orientation': state['orientation'],  
                'velocity': state['linear_velocity'],  
                'target': self.eval_env.target_state[:3],  
                'action': action[0],  
                'reward': reward  
            })  
              
            step += 1  
              
        self.video_recorder.stop_recording()  
          
        # 保存视频  
        video_name = f"eval_{self.ppo.num_timesteps}.mp4"  
        self.video_recorder.save_video_2d(video_name)  
          
        print(f"  Video saved: {video_name}")  
          
    def _generate_report(self):  
        """生成训练报告"""  
        report_path = os.path.join(self.log_dir, "training_report.txt")  
          
        with open(report_path, 'w') as f:  
            f.write("=" * 60 + "\n")  
            f.write("LOW-LEVEL CONTROLLER TRAINING REPORT\n")  
            f.write("=" * 60 + "\n\n")  
              
            f.write(f"Experiment: {self.experiment_name}\n")  
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")  
              
            f.write("Configuration:\n")  
            f.write(f"  Total timesteps: {self.ppo.num_timesteps:,}\n")  
            f.write(f"  Total episodes: {self.ppo.num_episodes:,}\n")  
            f.write(f"  Number of environments: {self.num_envs}\n")  
            f.write(f"  Device: {self.device}\n\n")  
              
            f.write("Results:\n")  
            f.write(f"  Best reward: {self.best_reward:.2f}\n")  
            f.write(f"  Final curriculum stage: {self.current_curriculum_stage}\n\n")  
              
            if self.curriculum_history:  
                f.write("Curriculum progression:\n")  
                for entry in self.curriculum_history:  
                    f.write(f"  Stage {entry['stage']} at step {entry['timestep']:,} (reward: {entry['reward']:.2f})\n")  
                f.write("\n")  
                  
            f.write("Files:\n")  
            f.write(f"  Best model: {self.checkpoint_dir}/best_model.pth\n")  
            f.write(f"  Final model: {self.checkpoint_dir}/final_model.pth\n")  
            f.write(f"  Logs: {self.log_dir}\n")  
            f.write(f"  Videos: {self.video_dir}\n")  
              
        print(f"Training report saved to {report_path}")  
          
        # 生成训练曲线图  
        self.visualizer.plot_reward_curve(  
            reward_key='episode/reward',  
            save_name='reward_curve.png'  
        )  
        self.visualizer.plot_loss_curves(  
            actor_loss_key='train/actor_loss',  
            critic_loss_key='train/critic_loss',  
            save_name='loss_curves.png'  
        )  
        self.visualizer.save_data('training_data.json')  
          
    def close(self):  
        """关闭所有资源"""  
        self.train_env.close()  
        self.eval_env.close()  
        self.logger.close()  
  
  
def parse_args():  
    """解析命令行参数"""  
    parser = argparse.ArgumentParser(description="Train low-level AUV controller")  
      
    # 配置  
    parser.add_argument('--config', type=str,   
                       default='configs/low_level_config.yaml',  
                       help='Path to config file')  
      
    # 训练参数  
    parser.add_argument('--total-timesteps', type=int, default=None,  
                       help='Total training timesteps (overrides config)')  
    parser.add_argument('--seed', type=int, default=None,  
                       help='Random seed')  
    parser.add_argument('--curriculum-stage', type=int, default=0,  
                       help='Starting curriculum stage')  
    parser.add_argument('--use-2d', action='store_true', help='Use simplified 2D 4-action environment')
      
    # 实验管理  
    parser.add_argument('--experiment-name', type=str, default=None,  
                       help='Experiment name (default: timestamp)')  
    parser.add_argument('--resume', type=str, default=None,  
                       help='Path to checkpoint to resume from')  
      
    # 调试  
    parser.add_argument('--test', action='store_true',  
                       help='Run a quick test (1000 steps)')  
    parser.add_argument('--no-cuda', action='store_true',  
                       help='Disable CUDA')  
      
    return parser.parse_args()  
  
  
def main():  
    """主函数"""  
    args = parse_args()  
      
    # 加载配置  
    config_path = os.path.join(PROJECT_ROOT, args.config)  
    with open(config_path, 'r') as f:  
        config = yaml.safe_load(f)  
          
    # 设置随机种子  
    if args.seed is not None:  
        np.random.seed(args.seed)  
        torch.manual_seed(args.seed)  
        if torch.cuda.is_available():  
            torch.cuda.manual_seed_all(args.seed)  
              
    # 禁用CUDA  
    if args.no_cuda:  
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  
          
    # 测试模式  
    if args.test:  
        args.total_timesteps = 1000  
        args.experiment_name = 'test_run'  
        config['training']['eval_freq'] = 500  
        config['training']['save_freq'] = 1000  
          
    # 创建训练器  
    trainer = LowLevelTrainer(config, args)  
      
    # 开始训练  
    try:  
        trainer.train(total_timesteps=args.total_timesteps)  
    except KeyboardInterrupt:  
        print("\nTraining interrupted by user")  
        trainer.close()  
    except Exception as e:  
        print(f"\nTraining failed with error: {e}")  
        import traceback  
        traceback.print_exc()  
        trainer.close()  
        raise  
  
  
if __name__ == "__main__":  
    main()  
