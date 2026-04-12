# 3D V2 消融实验 V3 - 新增文件与用法说明

## 一、新增文件清单（不修改原有代码）

| 文件路径 | 说明 |
|----------|------|
| docs/3d_kinematics_and_simplifications.md | 3D 运动学架构文档 |
| docs/ABLATION_V3_README.md | 本文件 |
| core/environment/low_level_env_3d_v2_ablation.py | 消融用环境，支持 success_threshold |
| training/train_3d_v2_ablation_v3.py | 消融训练脚本，10 组单变量消融 |

未修改：train_3d_v2.py、low_level_env_3d_v2.py 等原有代码。

## 二、3D 运动学：双环 PID + RL

RL 8 动作 -> LowLevelAUVEnv -> IntegratedController(双环PID) -> AUVDynamics 6DOF。简化：roll/pitch=0，8 动作，10 维观测，到达 0.5m。

## 三、消融 10 组

| 名称 | 描述 | 说明 |
|------|------|------|
| 3d_v2_abl_bp06 | backward_penalty=0.6 | 后退惩罚略增 |
| 3d_v2_abl_bp04 | backward_penalty=0.4 | 后退惩罚略减 |
| 3d_v2_abl_h08 | heading_coef=0.8 | 朝向奖励系数 |
| 3d_v2_abl_wp02 | heading_penalty_wrong=0.2 | 反朝向惩罚 |
| 3d_v2_abl_prec03 | success_threshold=0.3 | 到达精度 0.3m |
| 3d_v2_abl_net_large | 网络 [512,256,128] | 更大网络 |
| 3d_v2_abl_net_deep | 网络 [256,256,128,64] | 更深网络 |
| 3d_v2_abl_lr_low | lr=1e-4 | 学习率降低 |
| 3d_v2_abl_ent08 | ent_coef=0.08 | 熵系数略增 |
| 3d_v2_abl_nenvs16 | n_envs=16 | 并行环境数量（见下方说明） |

**n_envs 说明**：`n_envs` 表示训练时**并行运行的环境数量**。PPO 每轮会同时跑多个 3D 仿真环境收集样本，基数默认 8。本组设为 16，即每轮并行 16 个环境，可提高采样效率、缩短 wall-clock 训练时间。

## 四、运行方式

在项目根目录 `UnderwaterPursuitEvasion/` 下执行。

### 4.1 各组单终端训练命令

可在不同 terminal 中分别运行下列命令，每组占一个终端：

```bash
# 组 1：backward_penalty=0.6
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_bp06

# 组 2：backward_penalty=0.4
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_bp04

# 组 3：heading_coef=0.8
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_h08

# 组 4：heading_penalty_wrong=0.2
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_wp02

# 组 5：success_threshold=0.3（高精度，使用 ablation env）
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_prec03

# 组 6：网络 [512,256,128]
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_net_large

# 组 7：网络 [256,256,128,64]
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_net_deep

# 组 8：lr=1e-4
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_lr_low

# 组 9：ent_coef=0.08
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_ent08

# 组 10：n_envs=16
cd UnderwaterPursuitEvasion
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_nenvs16
```

### 4.2 其他用法

```bash
# 一次运行全部 10 组（4 进程并行）
python training/train_3d_v2_ablation_v3.py --all --max-parallel 4

# 一次运行多组（例如 3 组）
python training/train_3d_v2_ablation_v3.py --variants 3d_v2_abl_bp06 3d_v2_abl_prec03 3d_v2_abl_nenvs16

# 查看所有 variant 及帮助
python training/train_3d_v2_ablation_v3.py
```

### 4.3 可选参数

- `--total-timesteps N`：总训练步数（默认 2000000）
- `--seed N`：随机种子（默认 42）

### 4.4 输出目录

- 模型 checkpoint：`checkpoints/<variant_name>/`
- 训练曲线与轨迹图：`visualization/logs/<variant_name>/plots/`
