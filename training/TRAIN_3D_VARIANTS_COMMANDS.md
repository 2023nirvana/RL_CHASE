# 3D模型变体训练命令

以下是在不同终端窗口中分别运行的训练命令，每个命令对应一个变体配置。

## 通用参数
- **目标距离**: 5.0m
- **总训练步数**: 2,000,000
- **随机种子**: 42

---

## 变体1: Baseline (基准配置)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_baseline \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42
```

**配置**: 网络[128, 64], LR=3e-4, 环境数=4, 步数=1024

---

## 变体2: Large Network (大网络)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_large_net \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --hidden-sizes 256 128 64
```

**配置**: 网络[256, 128, 64], 更大的网络容量

---

## 变体3: Deep Network (深网络)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_deep_net \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --hidden-sizes 128 128 64 32
```

**配置**: 网络[128, 128, 64, 32], 更深的网络结构

---

## 变体4: High Learning Rate (高学习率)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_high_lr \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --learning-rate 0.001
```

**配置**: 学习率=1e-3 (默认3e-4的3.3倍)

---

## 变体5: Low Clip Range (低裁剪范围)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_low_clip \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --clip-range 0.1
```

**配置**: Clip范围=0.1 (默认0.2的一半，更保守的策略更新)

---

## 变体6: High Entropy (高熵系数)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_high_ent \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --ent-coef 0.1
```

**配置**: 熵系数=0.1 (默认0.05的2倍，更强的探索)

---

## 变体7: More Environments (更多环境)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_more_envs \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --n-envs 8
```

**配置**: 并行环境数=8 (默认4的2倍，更多样本收集)

---

## 变体8: Longer Steps (更长步数)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_long_steps \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --n-steps 2048
```

**配置**: Rollout步数=2048 (默认1024的2倍，更长的轨迹)

---

## 变体9: Large Batch (大批次)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_large_batch \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --batch-size 512
```

**配置**: 批次大小=512 (默认256的2倍，更稳定的梯度估计)

---

## 变体10: Combined (组合配置)

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_3d.py \
    --experiment-name 3d_combined \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    --hidden-sizes 256 128 64 \
    --n-envs 8 \
    --n-steps 2048
```

**配置**: 大网络[256, 128, 64] + 8环境 + 2048步，组合多种改进

---

## 使用建议

### 1. 终端分配
建议在10个不同的终端窗口中分别运行这些命令，每个终端对应一个变体，方便实时查看训练进度。

### 2. 监控训练
每个变体的训练日志会输出到终端，包括：
- 更新次数和总步数
- 评估成功率 (Eval SR)
- 训练成功率 (Train SR)
- 策略熵值
- 最佳模型保存提示

### 3. 检查点位置
训练过程中，模型会保存在：
```
checkpoints/<experiment_name>/
  - best_model.pth          # 最佳模型
  - checkpoint_<steps>.pth  # 定期保存的检查点
  - final_model.pth         # 最终模型
```

### 4. 训练曲线（自动生成）
训练过程中会自动生成训练曲线，保存在：
```
visualization/logs/<experiment_name>/plots/
  - training_curves.png      # 综合训练曲线（每5000步更新）
  - reward_curve.png         # 奖励曲线
  - loss_curves.png          # Loss曲线（Actor + Critic）
  - training_data.json       # 原始数据（JSON格式）
```

**记录的指标包括**：
- Actor Loss / Critic Loss / Total Loss
- Mean Reward（每个rollout的平均奖励）
- Train Success Rate / Eval Success Rate
- Entropy（策略熵值）

**查看曲线**：训练过程中，图表会每5000步自动更新，你可以实时查看训练进度。

### 4. 实时监控命令
在另一个终端中，可以使用以下命令监控训练进度：

```bash
# 查看所有变体的checkpoint目录
ls -lh checkpoints/3d_*/

# 查看特定变体的最新日志
tail -f visualization/logs/*/3d_baseline/*.log

# 查看训练曲线（实时更新）
ls -lt visualization/logs/*/plots/training_curves.png | head -1

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 监控训练曲线更新（每10秒检查一次）
watch -n 10 'ls -lh visualization/logs/*/plots/training_curves.png 2>/dev/null | tail -5'
```

### 5. 训练时间估算
- 每个变体大约需要训练数小时（取决于GPU性能）
- 可以根据实际情况调整 `--total-timesteps` 参数

### 6. 提前停止
如果某个变体表现明显不好，可以按 `Ctrl+C` 停止该变体的训练，节省资源给其他变体。

---

## 快速启动脚本

如果想一次性查看所有命令，可以运行：

```bash
bash training/train_3d_variants_commands.sh
```

这会打印出所有10个变体的训练命令，然后你可以复制到不同的终端中运行。
