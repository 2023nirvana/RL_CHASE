6.5 模块说明
这个模块完成了什么？
replay_buffer.py - 经验回放缓冲区

RolloutBuffer: 标准on-policy缓冲区
RolloutBufferWithLSTM: 支持LSTM的序列化缓冲区
EpisodeBuffer: 按episode存储的缓冲区
支持GAE计算和批次生成
ppo.py - PPO算法核心

完整的PPO-Clip实现
支持LSTM策略网络
GAE优势估计
价值函数裁剪
熵正则化
学习率线性衰减
梯度裁剪
KL散度早停
hierarchical_controller.py - 分层控制

高层-底层策略结合
技能执行器
为后续高层策略做准备


调试技巧
1. 如果训练不收敛：

检查奖励范围是否合理
降低学习率
增加n_steps
检查观测是否归一化_
2. 如果KL散度太大：

降低学习率
减少n_epochs
启用target_kl早停
3. 如果熵下降太快：

增加ent_coef
检查奖励是否太稀疏_
4. 如果价值函数不准：

增加Critic网络容量
增加vf_coef
检查回报计算是否正确_