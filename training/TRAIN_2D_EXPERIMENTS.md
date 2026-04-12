# 2D模型实验训练命令

## 概述

本文件包含两类实验的多组训练配置：

1. **任务1：避障训练** - 基于shaping_10m预训练模型，学习避障能力
2. **任务2：稀疏+朝向+课程学习** - 纯稀疏奖励（+少量朝向10:1权重），从近到远逐步学习

---

## 任务1：避障训练变体

### 变体1A：基础避障（从0个障碍物逐步增加到5个）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_obstacle_finetune.py \
    --experiment-name obs_v1a_baseline \
    --pretrain-checkpoint checkpoints/shaping_10m/best_model.pth \
    --target-distance 5.0 \
    --initial-obstacles 0 \
    --max-obstacles 5 \
    --collision-penalty -50.0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 2000000
```

### 变体1B：更强碰撞惩罚（-100）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_obstacle_finetune.py \
    --experiment-name obs_v1b_strong_penalty \
    --pretrain-checkpoint checkpoints/shaping_10m/best_model.pth \
    --target-distance 5.0 \
    --initial-obstacles 0 \
    --max-obstacles 5 \
    --collision-penalty -100.0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 2000000
```

### 变体1C：更多障碍物（最多8个）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_obstacle_finetune.py \
    --experiment-name obs_v1c_more_obstacles \
    --pretrain-checkpoint checkpoints/shaping_10m/best_model.pth \
    --target-distance 5.0 \
    --initial-obstacles 0 \
    --max-obstacles 8 \
    --collision-penalty -50.0 \
    --curriculum-threshold 0.6 \
    --total-timesteps 3000000
```

### 变体1D：保守课程学习（更高成功率门槛0.8）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_obstacle_finetune.py \
    --experiment-name obs_v1d_conservative \
    --pretrain-checkpoint checkpoints/shaping_10m/best_model.pth \
    --target-distance 5.0 \
    --initial-obstacles 0 \
    --max-obstacles 5 \
    --collision-penalty -50.0 \
    --curriculum-threshold 0.8 \
    --curriculum-window 150 \
    --total-timesteps 2500000
```

### 变体1E：密集避障奖励 + 较强碰撞惩罚

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_obstacle_finetune.py \
    --experiment-name obs_v1e_dense_avoidance \
    --pretrain-checkpoint checkpoints/shaping_10m/best_model.pth \
    --target-distance 5.0 \
    --initial-obstacles 0 \
    --max-obstacles 5 \
    --collision-penalty -80.0 \
    --obstacle-avoidance-coef 1.0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 2000000
```

---

## 任务2：稀疏+朝向+课程学习变体

### 变体2A：基础版（10:1权重，1m->10m课程）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_sparse_curriculum.py \
    --experiment-name sparse_cur_v2a_baseline \
    --goal-reward 100.0 \
    --heading-bonus 0.1 \
    --initial-distance 1.0 \
    --max-distance 10.0 \
    --distance-increment 1.0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 3000000
```

### 变体2B：更小朝向奖励（20:1权重）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_sparse_curriculum.py \
    --experiment-name sparse_cur_v2b_less_heading \
    --goal-reward 100.0 \
    --heading-bonus 0.05 \
    --initial-distance 1.0 \
    --max-distance 10.0 \
    --distance-increment 1.0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 3000000
```

### 变体2C：更大朝向奖励（5:1权重）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_sparse_curriculum.py \
    --experiment-name sparse_cur_v2c_more_heading \
    --goal-reward 100.0 \
    --heading-bonus 0.2 \
    --initial-distance 1.0 \
    --max-distance 10.0 \
    --distance-increment 1.0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 3000000
```

### 变体2D：更细粒度课程（0.5m步长）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_sparse_curriculum.py \
    --experiment-name sparse_cur_v2d_fine_curriculum \
    --goal-reward 100.0 \
    --heading-bonus 0.1 \
    --initial-distance 0.5 \
    --max-distance 10.0 \
    --distance-increment 0.5 \
    --curriculum-threshold 0.65 \
    --curriculum-window 80 \
    --total-timesteps 4000000
```

### 变体2E：激进课程学习（0.6门槛，更大步长）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_sparse_curriculum.py \
    --experiment-name sparse_cur_v2e_aggressive \
    --goal-reward 100.0 \
    --heading-bonus 0.1 \
    --initial-distance 1.0 \
    --max-distance 10.0 \
    --distance-increment 2.0 \
    --curriculum-threshold 0.6 \
    --curriculum-window 50 \
    --total-timesteps 2500000
```

### 变体2F：纯稀疏（无朝向奖励，用于对比）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_sparse_curriculum.py \
    --experiment-name sparse_cur_v2f_pure_sparse \
    --goal-reward 100.0 \
    --heading-bonus 0.0 \
    --heading-penalty 0.0 \
    --initial-distance 1.0 \
    --max-distance 10.0 \
    --distance-increment 1.0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 3000000
```

### 变体2G：更大到达奖励（200分）

```bash
cd /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion

python training/train_sparse_curriculum.py \
    --experiment-name sparse_cur_v2g_big_goal \
    --goal-reward 200.0 \
    --heading-bonus 0.1 \
    --initial-distance 1.0 \
    --max-distance 10.0 \
    --distance-increment 1.0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 3000000
```

---

## 快速参考表

| 变体 | 描述 | 关键参数 |
|------|------|----------|
| **任务1：避障** | | |
| 1A | 基础避障 | 0→5障碍物, 碰撞-50 |
| 1B | 强惩罚 | 碰撞-100 |
| 1C | 更多障碍物 | 0→8障碍物 |
| 1D | 保守学习 | 门槛0.8 |
| 1E | 密集避障 | avoidance_coef=1.0 |
| **任务2：稀疏课程** | | |
| 2A | 基础版 | 10:1权重, 1m→10m |
| 2B | 少朝向 | 20:1权重 |
| 2C | 多朝向 | 5:1权重 |
| 2D | 细粒度 | 0.5m步长 |
| 2E | 激进 | 2m步长, 0.6门槛 |
| 2F | 纯稀疏 | 无朝向奖励 |
| 2G | 大奖励 | goal=200 |

---

## 推荐运行顺序

### 避障训练推荐
1. **首选：1A（基础版）** - 验证基本方法可行
2. 然后：1B（强惩罚）或 1E（密集避障）- 看哪个效果更好
3. 最后：1C（更多障碍物）- 用于挑战更难场景

### 稀疏课程学习推荐
1. **首选：2A（基础版）** - 验证10:1权重有效
2. 对比：2F（纯稀疏）- 验证朝向奖励的价值
3. 然后：2D（细粒度）- 如果基础版在某些阶段卡住
4. 可选：2C（多朝向）- 如果探索不足

---

## 监控训练

训练时可以用以下命令监控：

```bash
# 查看最新日志
tail -f visualization/logs/<experiment_name>/plots/training_curves.png

# 查看成功率
grep "Eval SR" visualization/logs/<experiment_name>/training.log

# 监控GPU使用
watch -n 1 nvidia-smi
```

## 并行运行

可以在不同终端同时运行多个实验。建议GPU显存足够时最多同时运行2-3个。
