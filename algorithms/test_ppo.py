"""  
PPO算法测试脚本  
"""  
  
import numpy as np  
import torch  
import os  
import sys  
import time  
  
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, PROJECT_ROOT)  
  
from algorithms.replay_buffer import RolloutBuffer, RolloutBufferWithLSTM  
from algorithms.ppo import PPO, PPOTrainer  
from networks.ppo_lstm_network import PPOLSTMNetwork, PPONetwork  
from core.environment.low_level_env import LowLevelAUVEnv, VecLowLevelAUVEnv  
  
  
def test_rollout_buffer():  
    """测试Rollout缓冲区"""  
    print("=" * 60)  
    print("测试1: Rollout缓冲区")  
    print("=" * 60)  
      
    buffer_size = 128  
    obs_dim = 24  
    num_envs = 4  
      
    buffer = RolloutBuffer(  
        buffer_size=buffer_size,  
        observation_dim=obs_dim,  
        num_envs=num_envs,  
        gamma=0.99,  
        gae_lambda=0.95  
    )  
      
    # 填充数据  
    for step in range(buffer_size):  
        obs = np.random.randn(num_envs, obs_dim).astype(np.float32)  
        action = np.random.randint(0, 13, num_envs)  
        reward = np.random.randn(num_envs).astype(np.float32)  
        done = np.random.random(num_envs) < 0.01  
        value = np.random.randn(num_envs).astype(np.float32)  
        log_prob = np.random.randn(num_envs).astype(np.float32)  
          
        buffer.add(obs, action, reward, done.astype(np.float32), value, log_prob)  
          
    print(f"缓冲区大小: {buffer_size} x {num_envs}")  
    print(f"缓冲区已满: {buffer.full}")  
      
    # 计算GAE  
    last_value = np.random.randn(num_envs).astype(np.float32)  
    last_done = np.zeros(num_envs, dtype=np.float32)  
    buffer.compute_returns_and_advantages(last_value, last_done)  
      
    print(f"优势范围: [{buffer.advantages.min():.2f}, {buffer.advantages.max():.2f}]")  
    print(f"回报范围: [{buffer.returns.min():.2f}, {buffer.returns.max():.2f}]")  
      
    # 测试数据生成器  
    batch_count = 0  
    for batch in buffer.get(batch_size=64):  
        batch_count += 1  
          
    print(f"生成批次数: {batch_count}")  
    print("Rollout缓冲区测试通过!\n")  
  
  
def test_rollout_buffer_lstm():  
    """测试LSTM Rollout缓冲区"""  
    print("=" * 60)  
    print("测试2: LSTM Rollout缓冲区")  
    print("=" * 60)  
      
    buffer_size = 128  
    obs_dim = 24  
    num_envs = 4  
    lstm_hidden = 256  
    lstm_layers = 2  
      
    buffer = RolloutBufferWithLSTM(  
        buffer_size=buffer_size,  
        observation_dim=obs_dim,  
        num_envs=num_envs,  
        lstm_hidden_size=lstm_hidden,  
        lstm_num_layers=lstm_layers  
    )  
      
    # 填充数据  
    for step in range(buffer_size):  
        obs = np.random.randn(num_envs, obs_dim).astype(np.float32)  
        action = np.random.randint(0, 13, num_envs)  
        reward = np.random.randn(num_envs).astype(np.float32)  
        done = np.random.random(num_envs) < 0.05  
        value = np.random.randn(num_envs).astype(np.float32)  
        log_prob = np.random.randn(num_envs).astype(np.float32)  
        hidden = (  
            np.random.randn(lstm_layers, num_envs, lstm_hidden).astype(np.float32),  
            np.random.randn(lstm_layers, num_envs, lstm_hidden).astype(np.float32)  
        )  
        episode_start = (done | (step == 0)).astype(np.float32)  
          
        buffer.add(obs, action, reward, done.astype(np.float32), value, log_prob, hidden, episode_start)  
          
    # 计算GAE  
    last_value = np.random.randn(num_envs).astype(np.float32)  
    last_done = np.zeros(num_envs, dtype=np.float32)  
    buffer.compute_returns_and_advantages(last_value, last_done)  
      
    # 测试序列数据生成器  
    seq_length = 16  
    batch_size = 8  
    batch_count = 0  
      
    for batch in buffer.get_sequential(seq_length=seq_length, batch_size=batch_size):  
        batch_count += 1  
          
        # 验证形状  
        assert batch['observations'].shape[0] == seq_length  
        assert batch['initial_hidden'][0].shape[1] == batch['observations'].shape[1]  
          
    print(f"序列长度: {seq_length}")  
    print(f"生成序列批次数: {batch_count}")  
    print("LSTM Rollout缓冲区测试通过!\n")  
  
  
def test_ppo_single_env():  
    """测试单环境PPO"""  
    print("=" * 60)  
    print("测试3: 单环境PPO")  
    print("=" * 60)  
      
    # 创建环境  
    env = LowLevelAUVEnv(  
        curriculum_stage=0,  
        enable_domain_randomization=False,  
        seed=42  
    )  
      
    # 创建网络  
    obs_dim = env.observation_space.shape[0]  
    num_actions = env.action_space.n  
      
    network = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions,  
        feature_hidden_sizes=[128, 128],  
        lstm_hidden_size=128,  
        lstm_num_layers=1  
    )  
      
    print(f"观测维度: {obs_dim}")  
    print(f"动作数量: {num_actions}")  
    print(f"网络参数量: {sum(p.numel() for p in network.parameters()):,}")  
      
    # 创建PPO  
    ppo = PPO(  
        env=env,  
        network=network,  
        n_steps=256,  
        batch_size=64,  
        n_epochs=4,  
        learning_rate=3e-4,  
        verbose=0  
    )  
      
    # 收集一次rollout  
    print("\n收集rollout...")  
    rollout_info = ppo.collect_rollouts()  
    print(f"平均奖励: {rollout_info['mean_reward']:.2f}")  
    print(f"平均长度: {rollout_info['mean_length']:.1f}")  
      
    # 执行一次训练  
    print("\n执行训练...")  
    train_info = ppo.train()  
    print(f"Actor损失: {train_info['actor_loss']:.4f}")  
    print(f"Critic损失: {train_info['critic_loss']:.4f}")  
    print(f"熵: {train_info['entropy']:.4f}")  
    print(f"KL散度: {train_info['approx_kl']:.4f}")  
      
    env.close()  
    print("单环境PPO测试通过!\n")  
  
  
def test_ppo_vec_env():  
    """测试向量化环境PPO"""  
    print("=" * 60)  
    print("测试4: 向量化环境PPO")  
    print("=" * 60)  
      
    num_envs = 4  
      
    # 创建向量化环境  
    vec_env = VecLowLevelAUVEnv(  
        num_envs=num_envs,  
        curriculum_stage=0,  
        enable_domain_randomization=True  
    )  
      
    # 创建网络  
    obs_dim = vec_env.observation_space.shape[0]  
    num_actions = vec_env.action_space.n  
      
    network = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions,  
        feature_hidden_sizes=[128, 128],  
        lstm_hidden_size=128,  
        lstm_num_layers=1  
    )  
      
    # 创建PPO  
    ppo = PPO(  
        env=vec_env,  
        network=network,  
        n_steps=128,  
        batch_size=64,  
        n_epochs=4,  
        learning_rate=3e-4,  
        verbose=0  
    )  
      
    print(f"环境数量: {num_envs}")  
    print(f"每次收集步数: {ppo.n_steps * num_envs}")  
      
    # 多次收集和训练  
    for i in range(3):  
        rollout_info = ppo.collect_rollouts()  
        train_info = ppo.train()  
          
        print(f"迭代 {i+1}: 奖励={rollout_info['mean_reward']:.2f}, "  
              f"Actor损失={train_info['actor_loss']:.4f}, "  
              f"Critic损失={train_info['critic_loss']:.4f}")  
          
    vec_env.close()  
    print("向量化环境PPO测试通过!\n")  
  
  
def test_ppo_training():  
    """测试PPO完整训练流程"""  
    print("=" * 60)  
    print("测试5: PPO完整训练流程（短期）")  
    print("=" * 60)  
      
    # 创建环境  
    vec_env = VecLowLevelAUVEnv(  
        num_envs=4,  
        curriculum_stage=0,  
        enable_domain_randomization=False  
    )  
      
    # 创建网络  
    obs_dim = vec_env.observation_space.shape[0]  
    num_actions = vec_env.action_space.n  
      
    network = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions,  
        feature_hidden_sizes=[128, 128],  
        lstm_hidden_size=128,  
        lstm_num_layers=1  
    )  
      
    # 创建PPO  
    log_dir = os.path.join(PROJECT_ROOT, "visualization/logs/test_ppo")  
    os.makedirs(log_dir, exist_ok=True)  
      
    ppo = PPO(  
        env=vec_env,  
        network=network,  
        n_steps=128,  
        batch_size=64,  
        n_epochs=4,  
        learning_rate=3e-4,  
        tensorboard_log=log_dir,  
        verbose=1  
    )  
      
    # 短期训练  
    print("\n开始训练（5000步）...")  
    start_time = time.time()  
      
    ppo.learn(  
        total_timesteps=5000,  
        log_interval=1,  
        save_freq=10000,  
        save_path=os.path.join(PROJECT_ROOT, "checkpoints/test")  
    )  
      
    elapsed = time.time() - start_time  
    print(f"\n训练耗时: {elapsed:.1f}s")  
    print(f"FPS: {ppo.num_timesteps / elapsed:.1f}")  
      
    vec_env.close()  
    print("PPO完整训练流程测试通过!\n")  
  
  
def test_ppo_save_load():  
    """测试PPO保存和加载"""  
    print("=" * 60)  
    print("测试6: PPO保存和加载")  
    print("=" * 60)  
      
    # 创建环境  
    env = LowLevelAUVEnv(curriculum_stage=0, seed=42)  
      
    # 创建网络  
    obs_dim = env.observation_space.shape[0]  
    num_actions = env.action_space.n  
      
    network = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions  
    )  
      
    # 创建PPO并训练少量步数  
    ppo = PPO(  
        env=env,  
        network=network,  
        n_steps=64,  
        verbose=0  
    )  
      
    ppo.collect_rollouts()  
    ppo.train()  
      
    # 保存  
    save_path = os.path.join(PROJECT_ROOT, "checkpoints/test_save_load.pth")  
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    ppo.save(save_path)  
    print(f"模型已保存到: {save_path}")  
      
    # 获取预测结果  
    obs, _ = env.reset()  
    action1, _ = ppo.predict(obs, deterministic=True)  
      
    # 加载到新的PPO  
    network2 = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions  
    )  
    ppo2 = PPO(  
        env=env,  
        network=network2,  
        n_steps=64,  
        verbose=0  
    )  
    ppo2.load(save_path)  
      
    # 验证预测一致  
    action2, _ = ppo2.predict(obs, deterministic=True)  
      
    print(f"保存前动作: {action1}")  
    print(f"加载后动作: {action2}")  
    print(f"动作一致: {np.array_equal(action1, action2)}")  
      
    # 清理  
    os.remove(save_path)  
    env.close()  
      
    print("PPO保存和加载测试通过!\n")  
  
  
def test_ppo_evaluation():  
    """测试PPO评估"""  
    print("=" * 60)  
    print("测试7: PPO评估")  
    print("=" * 60)  
      
    # 创建环境  
    env = LowLevelAUVEnv(curriculum_stage=0, seed=42)  
    eval_env = LowLevelAUVEnv(curriculum_stage=0, seed=123)  
      
    # 创建网络  
    obs_dim = env.observation_space.shape[0]  
    num_actions = env.action_space.n  
      
    network = PPOLSTMNetwork(  
        obs_dim=obs_dim,  
        num_actions=num_actions  
    )  
      
    # 创建PPO  
    ppo = PPO(  
        env=env,  
        network=network,  
        n_steps=128,  
        verbose=0  
    )  
      
    # 评估（未训练）  
    print("评估未训练的策略...")  
    eval_info = ppo.evaluate(eval_env, n_episodes=5)  
    print(f"平均奖励: {eval_info['mean_reward']:.2f} ± {eval_info['std_reward']:.2f}")  
    print(f"平均长度: {eval_info['mean_length']:.1f}")  
    print(f"成功率: {eval_info['success_rate']:.2%}")  
      
    # 训练一点  
    print("\n训练中...")  
    for _ in range(5):  
        ppo.collect_rollouts()  
        ppo.train()  
          
    # 再次评估  
    print("\n评估训练后的策略...")  
    eval_info = ppo.evaluate(eval_env, n_episodes=5)  
    print(f"平均奖励: {eval_info['mean_reward']:.2f} ± {eval_info['std_reward']:.2f}")  
    print(f"平均长度: {eval_info['mean_length']:.1f}")  
    print(f"成功率: {eval_info['success_rate']:.2%}")  
      
    env.close()  
    eval_env.close()  
      
    print("PPO评估测试通过!\n")  
  
  
def run_all_tests():  
    """运行所有测试"""  
    print("\n" + "=" * 60)  
    print("PPO算法测试")  
    print("=" * 60 + "\n")  
      
    test_rollout_buffer()  
    test_rollout_buffer_lstm()  
    test_ppo_single_env()  
    test_ppo_vec_env()  
    test_ppo_training()  
    test_ppo_save_load()  
    test_ppo_evaluation()  
      
    print("=" * 60)  
    print("所有PPO测试完成!")  
    print("=" * 60)  
  
  
if __name__ == "__main__":  
    run_all_tests()  
