# 训练曲线可视化指南

## 概述

训练脚本现在会自动记录和可视化训练过程中的各种指标，包括：
- **Loss曲线**: Actor Loss, Critic Loss, Total Loss
- **Reward曲线**: 每个rollout的平均奖励
- **成功率曲线**: 训练和评估的成功率
- **熵值曲线**: 策略熵（探索程度）

## 自动生成的图表

训练过程中，系统会在以下位置自动生成图表：

```
visualization/logs/<experiment_name>/plots/
  ├── training_curves.png      # 综合训练曲线（每5000步更新）
  ├── reward_curve.png         # 奖励曲线（每5000步更新）
  ├── loss_curves.png          # Loss曲线（每5000步更新）
  ├── training_curves_final.png  # 最终综合曲线
  ├── reward_curve_final.png     # 最终奖励曲线
  ├── loss_curves_final.png      # 最终Loss曲线
  └── training_data.json         # 原始数据（JSON格式）
```

## 记录的指标

### 训练指标
- `train/actor_loss`: Actor网络的损失
- `train/critic_loss`: Critic网络的损失
- `train/total_loss`: 总损失（Actor + Critic + Entropy）
- `train/entropy`: 策略熵值（探索程度）

### 性能指标
- `rollout/mean_reward`: 每个rollout的平均奖励
- `train/success_rate`: 训练过程中的成功率
- `eval/success_rate`: 评估时的成功率（每5000步评估一次）
- `eval/mean_reward`: 评估时的平均奖励

## 查看训练曲线

### 方法1: 实时查看（推荐）

在训练过程中，图表会每5000步自动更新。你可以：

1. **在文件浏览器中查看**:
   ```bash
   # 查看最新的训练曲线
   ls -lt visualization/logs/*/plots/training_curves.png
   ```

2. **使用watch命令实时监控**:
   ```bash
   # 每5秒刷新一次（需要安装feh或类似工具）
   watch -n 5 'ls -lt visualization/logs/*/plots/training_curves.png | head -1'
   ```

3. **在Jupyter Notebook中查看**:
   ```python
   from IPython.display import Image, display
   import os
   
   # 查看最新的训练曲线
   plot_path = "visualization/logs/<experiment_name>/plots/training_curves.png"
   display(Image(plot_path))
   ```

### 方法2: 训练完成后查看

训练结束后，所有最终图表会保存在 `plots/` 目录中，文件名带有 `_final` 后缀。

## 图表说明

### 1. training_curves.png
包含所有主要指标的6个子图：
- Actor Loss
- Critic Loss
- Entropy
- Mean Reward
- Train Success Rate
- Eval Success Rate

### 2. reward_curve.png
专门的奖励曲线图：
- 原始奖励值（散点，半透明）
- 移动平均曲线（平滑）
- 最佳奖励标注
- 最近100个episode的平均值线

### 3. loss_curves.png
Loss对比图：
- Actor Loss（左图）
- Critic Loss（右图）
- 如果值范围很大，会自动使用对数刻度

## 数据保存

所有训练数据也会以JSON格式保存到 `training_data.json`，方便后续分析：

```python
import json
import matplotlib.pyplot as plt

# 加载数据
with open('visualization/logs/<experiment_name>/plots/training_data.json', 'r') as f:
    data = json.load(f)

# 自定义分析
steps = data['train/actor_loss']['steps']
values = data['train/actor_loss']['values']
plt.plot(steps, values)
plt.show()
```

## 实时监控示例

### 使用tail监控训练日志
```bash
# 监控特定变体的训练输出
tail -f visualization/logs/3d_large_net/*.log
```

### 使用ls监控图表更新
```bash
# 每10秒检查一次图表是否更新
while true; do
    ls -lh visualization/logs/*/plots/training_curves.png
    sleep 10
done
```

## 多变体对比

如果你想对比多个变体的训练曲线，可以：

1. **手动对比**: 打开不同变体的 `training_curves.png` 文件
2. **使用Python脚本**:
   ```python
   import matplotlib.pyplot as plt
   import json
   
   variants = ['3d_baseline', '3d_large_net', '3d_combined']
   
   fig, axes = plt.subplots(2, 3, figsize=(15, 10))
   
   for variant in variants:
       with open(f'visualization/logs/{variant}/plots/training_data.json', 'r') as f:
           data = json.load(f)
       
       # 绘制对比图
       steps = data['eval/success_rate']['steps']
       values = data['eval/success_rate']['values']
       axes[0, 0].plot(steps, values, label=variant)
   
   axes[0, 0].legend()
   axes[0, 0].set_title('Success Rate Comparison')
   plt.tight_layout()
   plt.savefig('variant_comparison.png')
   ```

## 注意事项

1. **更新频率**: 图表每5000步更新一次，避免频繁IO操作
2. **内存使用**: 所有数据都保存在内存中，训练结束后保存到JSON
3. **文件大小**: PNG图片通常几百KB，JSON数据文件可能几MB
4. **并发训练**: 每个变体有独立的plots目录，不会冲突

## 故障排除

### 图表没有生成
- 检查是否有写入权限
- 确认matplotlib已正确安装
- 查看训练日志中的错误信息

### 图表显示异常
- 检查数据是否正常记录（查看training_data.json）
- 确认有足够的数据点（至少需要几个更新周期）

### 内存不足
- 如果训练时间很长，考虑增加plot_interval（减少更新频率）
- 或者定期清理旧的检查点文件
