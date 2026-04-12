# 2D 追逃分层训练（完整动力学，与之前动作空间一致）

## 思路

1. **阶段一**：参考之前 2D 最好的模型，在**完整动力学**（`pe_2d_dynamics.Dynamics2D`，不简化）上训练追击者追移动目标；**动作空间与之前一致**：9 动作 = 0(无推力) + 8 方向满推力。
2. **阶段二**：用阶段一训练好的追击者（可选加载初始化），在**追逃环境**里训练追逃策略；物理与动作空间均与之前一致（速度劣势、加速度优势，9 动作）。

所有相关代码均为**新文件**，未修改原有 `env_asymmetric_pe_v2_dynamics`、`train_pe_v2_dynamics`、`pe_2d_dynamics` 等。

## 新增文件

| 文件 | 说明 |
|------|------|
| `env_pursuer_low_level_dynamics.py` | 阶段一环境：单智能体追移动目标，**9 动作**（与之前一致），观测 18 维（含 pos/vel/acc 等） |
| `train_pursuer_stage1_dynamics.py` | 阶段一 PPO 训练脚本，保存到 `checkpoints/pursuer_stage1_dynamics/` |
| `env_asymmetric_pe_v2_dynamics_accel.py` | 阶段二环境：追逃，追击者 **9 动作**，观测 22 维，物理为速度劣势/加速度优势 |
| `train_pe_stage2_dynamics.py` | 阶段二 PPO 训练脚本，可选从阶段一 checkpoint 初始化，保存到 `checkpoints/pe2d_stage2_dynamics_accel/` |

## 使用方法

在项目根目录 `UnderwaterPursuitEvasion/` 下执行。默认总步数与之前 2D 一致（阶段一 2M、阶段二 2M），训练时长应接近 `train_pe_v2_dynamics` / `train_2d_dynamic`；需更长可传 `--total-timesteps 5000000`。

### 阶段一：训练追击者（追移动目标）

```bash
# 默认：2M steps（与之前 2D 训练量一致），恒定速度目标
python pursuit_evasion_2d/train_pursuer_stage1_dynamics.py

# 指定实验名、目标运动方式；可加大总步数
python pursuit_evasion_2d/train_pursuer_stage1_dynamics.py \
  --experiment-name pursuer_const \
  --target-motion constant \
  --target-speed 0.5 \
  --total-timesteps 2000000

# 目标会逃逸（evasive）
python pursuit_evasion_2d/train_pursuer_stage1_dynamics.py \
  --target-motion evasive \
  --experiment-name pursuer_evasive
```

阶段一结束后，最佳模型在：
`checkpoints/pursuer_stage1_dynamics/<experiment_name>/best.pth`

### 阶段二：训练追逃策略

```bash
# 从阶段一 best.pth 初始化（推荐）
python pursuit_evasion_2d/train_pe_stage2_dynamics.py \
  --stage1-checkpoint checkpoints/pursuer_stage1_dynamics/pursuer_const/best.pth \
  --experiment-name pe_from_stage1 \
  --total-timesteps 2000000

# 不从阶段一加载（从零训练追逃策略）
python pursuit_evasion_2d/train_pe_stage2_dynamics.py \
  --experiment-name pe_scratch
```

阶段二最佳模型在：
`checkpoints/pe2d_stage2_dynamics_accel/<experiment_name>/best.pth`

## Option A（高层离散 + 底层连续 PID）

如果你要运行“高层离散宏动作，底层连续 PID 跟踪子目标”的版本，可直接使用：

```bash
python pursuit_evasion_2d/train_hrl_option_a_pid.py \
  --experiment-name optionA_pid \
  --total-timesteps 600000 \
  --n-envs 8 \
  --macro-steps 5 \
  --difficulty 0.6 \
  --evader-mode smart
```

对应环境类：`PursuitEvasion2DAsymmetricEnvV2PIDOptionA`（文件 `env_asymmetric_pe_v2_pid_option_a.py`）。

### 推荐两阶段流程（先低层连续，再高层追逃）

1) 先训练连续低层目标到达（验证控制到达质量）：

```bash
python pursuit_evasion_2d/train_low_level_continuous.py \
  --experiment-name ll_cont_stage1 \
  --total-timesteps 800000 \
  --n-envs 8
```

2) 再训练高层追逃（底层由连续 PID 执行 waypoint）：

```bash
python pursuit_evasion_2d/train_hrl_option_a_pid.py \
  --experiment-name optionA_stage2 \
  --total-timesteps 600000 \
  --macro-steps 5 \
  --difficulty 0.6
```

## 动力学说明

- 与 `pe_2d_dynamics` 一致：`M*a = τ - D_linear*v - D_quadratic*|v|*v`，RK4 积分，无简化。
- 追击者 / 逃脱者仍通过 `create_pursuer_dynamics`、`create_evader_dynamics` 创建（速度劣势、加速度优势已内置）。

## 动作空间（与之前 env_asymmetric_pe_v2_dynamics 一致）

- **0**：无推力  
- **1–8**：8 个方向，满推力  

共 9 个离散动作。
