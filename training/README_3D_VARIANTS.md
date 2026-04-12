# 3D模型并行训练指南

## 概述

为了找到最佳的3D八动作模型配置，我们创建了并行训练系统，可以同时训练多个不同网络结构和超参数的变体。

## 使用方法

### 方法1: 使用Shell脚本（推荐）

```bash
# 训练所有变体（并行）
bash training/train_3d_variants.sh --all --target-distance 5.0

# 训练特定变体
bash training/train_3d_variants.sh --variants 3d_large_net 3d_high_lr --target-distance 5.0

# 自定义参数
bash training/train_3d_variants.sh --all \
    --target-distance 5.0 \
    --total-timesteps 2000000 \
    --max-parallel 4
```

### 方法2: 直接使用Python脚本

```bash
# 单个变体训练
python training/train_3d.py \
    --experiment-name 3d_large_net \
    --target-distance 5.0 \
    --hidden-sizes 256 128 64 \
    --learning-rate 3e-4 \
    --n-envs 8 \
    --n-steps 2048
```

## 可用的变体配置

### 1. Baseline (3d_baseline)
- 网络: [128, 64]
- 学习率: 3e-4
- 环境数: 4
- 步数: 1024

### 2. Large Network (3d_large_net)
- 网络: [256, 128, 64]
- 更大的网络容量，可能学习更复杂的策略

### 3. Deep Network (3d_deep_net)
- 网络: [128, 128, 64, 32]
- 更深的网络，可能捕获更多层次特征

### 4. High Learning Rate (3d_high_lr)
- 学习率: 1e-3
- 更快的收敛速度（但可能不稳定）

### 5. Low Clip Range (3d_low_clip)
- Clip Range: 0.1
- 更保守的策略更新

### 6. High Entropy (3d_high_ent)
- Entropy Coef: 0.1
- 更强的探索

### 7. More Environments (3d_more_envs)
- 环境数: 8
- 更多的并行样本收集

### 8. Longer Steps (3d_long_steps)
- 步数: 2048
- 更长的rollout，可能提高样本效率

### 9. Large Batch (3d_large_batch)
- Batch Size: 512
- 更大的批次，可能更稳定

### 10. Combined (3d_combined)
- 网络: [256, 128, 64]
- 环境数: 8
- 步数: 2048
- 组合多种改进

## 训练参数说明

- `--target-distance`: 目标距离（默认5.0m）
- `--total-timesteps`: 总训练步数（默认2000000）
- `--max-parallel`: 最大并行训练数（默认4）
- `--seed`: 随机种子（默认42）

## 网络结构参数

- `--hidden-sizes`: 隐藏层大小，例如 `--hidden-sizes 256 128 64`
- `--learning-rate`: 学习率，例如 `--learning-rate 1e-3`
- `--clip-range`: PPO裁剪范围，例如 `--clip-range 0.1`
- `--ent-coef`: 熵系数，例如 `--ent-coef 0.1`
- `--n-envs`: 并行环境数，例如 `--n-envs 8`
- `--n-steps`: 每次rollout的步数，例如 `--n-steps 2048`
- `--batch-size`: 批次大小，例如 `--batch-size 512`

## 输出

训练结果保存在：
- Checkpoints: `checkpoints/<experiment_name>/`
- Logs: `visualization/logs/3d_variants_<timestamp>/`

每个变体的日志文件：`<variant_name>.log`

## 监控训练进度

```bash
# 查看所有训练日志
tail -f visualization/logs/3d_variants_*/3d_*.log

# 查看特定变体的日志
tail -f visualization/logs/3d_variants_*/3d_large_net.log
```

## 评估最佳模型

训练完成后，使用评估脚本比较不同变体的性能：

```bash
python evaluation/eval_3d.py --checkpoint checkpoints/3d_large_net/best_model.pth --target-distance 5.0
```

## 建议的训练策略

1. **先训练几个关键变体**：
   - Baseline（作为对比基准）
   - Large Network（测试网络容量）
   - Combined（测试组合效果）

2. **根据初步结果调整**：
   - 如果Large Network表现好，尝试更大的网络
   - 如果More Environments表现好，尝试更多环境
   - 如果Longer Steps表现好，尝试更长的rollout

3. **并行训练多个变体**：
   - 使用 `--max-parallel` 控制并行数量
   - 根据GPU内存调整并行数

## 注意事项

- 确保有足够的GPU内存来支持并行训练
- 每个变体会占用一定的GPU内存，根据实际情况调整 `--max-parallel`
- 训练时间取决于总步数和并行数量
- 建议定期检查日志，确保训练正常进行
