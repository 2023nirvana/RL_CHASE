# 三维 AUV 到达任务：运动学架构与简化约束

本文档说明 3D 到达任务（low_level_env_3d / low_level_env_3d_v2）的运动学架构、与原版的简化/约束，以及理论上的 relax 建议。

---

## 一、三维运动学架构：双环 PID + RL

**结论：是的，当前 3D 到达任务基于 双环PID + RL。**

### 1.1 控制链路

```
RL 策略 (离散 8 动作)
    ↓
LowLevelAUVEnv.action_map: 8→13 动作映射
    ↓
IntegratedController.compute_thrust(action, eta, nu, target)
    ├─ 外环 PI：角度误差 → 参考角速度
    ├─ 内环 PD：角速度误差 → 反馈力矩
    └─ 推力分配器 → 8 推进器推力 τ
    ↓
AUVDynamics.step(τ, external_force)
    M·ν̇ + C(ν)·ν + D(ν)·ν + g(η) = τ
    RK4 积分 → (η, ν)
```

### 1.2 关键代码位置

| 组件 | 文件 | 说明 |
|------|------|------|
| 动力学 | `core/dynamics/auv_dynamics.py` | 6DOF 非线性，M·ν̇+C·ν+D·ν+g=τ |
| 双环 PID | `core/control/dual_loop_pid.py` | IntegratedController, DualLoopPIDController |
| 底层环境 | `core/environment/low_level_env.py` | LowLevelAUVEnv，step 中调用 IntegratedController |
| 3D 包装 | `core/environment/low_level_env_3d.py` / `low_level_env_3d_v2.py` | 8 动作、10 维观测、自定义奖励 |

### 1.3 8 动作与 13 动作映射

3D 环境仅使用 8 个动作（0~7），映射到 LowLevelAUVEnv 的 13 动作中的 8 个：

| 3D 动作 | 名称 | LowLevel 动作 ID |
|---------|------|------------------|
| 0 | forward | 1 (surge_forward) |
| 1 | backward | 2 (surge_backward) |
| 2 | left | 3 (sway_left) |
| 3 | right | 4 (sway_right) |
| 4 | yaw_left | 7 (yaw_left) |
| 5 | yaw_right | 8 (yaw_right) |
| 6 | heave_up | 5 (heave_up) |
| 7 | heave_down | 6 (heave_down) |

未使用：0(hover), 9(pitch_up), 10(pitch_down), 11(roll_left), 12(roll_right)。

---

## 二、与原版相比的简化与约束

### 2.1 姿态约束（_enforce_3d_constraints）

**简化**：每步强制 roll=0、pitch=0、p=0、q=0，仅保留 yaw 自由度。

- `eta[3] = 0`  (roll)
- `eta[4] = 0`  (pitch)
- `nu[3] = 0`   (p, roll rate)
- `nu[4] = 0`   (q, pitch rate)
- `eta[2] = np.clip(eta[2], z_range[0], z_range[1])`  (z 限幅)

**目的**：将问题简化为水平面 + 深度，避免 roll/pitch 耦合带来的学习难度。

### 2.2 动作空间简化

- 去掉 pitch、roll 相关动作（9~12），仅保留平移 + yaw + heave。
- 等价于假设 AUV 始终“水平”，仅靠 yaw 转向、heave 升降。

### 2.3 观测空间简化

- 10 维：pos_error_body_x/y/z, heading_error, target_angle_xy, vx, vy, vz, yaw_rate, distance。
- 原 LowLevelAUVEnv 观测维度更大（含完整姿态、目标姿态等），3D 版本只保留与到达任务相关的量。

### 2.4 到达判定

- 成功条件：`pos_error < 0.5`（欧氏距离 < 0.5 m）。
- proximity 与 goal_bonus 也以 0.5 m 为界。

### 2.5 深度范围

- z 限制在 `z_range`（默认 [2, 4]）内，避免过浅/过深。

---

## 三、理论分析：哪些简化可以考虑适度放开

### 3.1 到达精度阈值（0.5 m → 0.3 m）

- **当前**：0.5 m 判定成功，对 12 m 任务相对宽松。
- **建议**：可做消融，将 `success_threshold` 调为 0.3 m，提高到达精度。
- **风险**：训练难度略增，建议在已收敛的 3d_v2 配置上尝试。

### 3.2 roll/pitch 强制归零

- **当前**：完全锁定 roll、pitch。
- **建议**：暂时不建议放开。原因：
  - 3D 任务主要依赖 yaw + 平移 + heave，roll/pitch 耦合会增加策略空间；
  - 若放开，需引入 pitch 动作与观测，改动较大；
  - 保持水平有利于稳定到达。
- **若将来放开**：可先放宽为小范围（如 |roll|, |pitch| < 0.1 rad），并增加相应观测与惩罚。

### 3.3 z 范围限制

- **当前**：z 严格裁剪到 [2, 4]。
- **建议**：保持。放宽 z 会增加无效探索，对到达任务收益有限。

### 3.4 动作空间（加入 pitch）

- **当前**：无 pitch。
- **建议**：暂不加入。若目标存在明显垂直分量，heave 已能覆盖主要需求；pitch 会增加动作与观测维度，宜先验证现有 8 动作表现。

### 3.5 观测归一化尺度

- **当前**：pos_error 除以 10，distance 除以 15，适用于 ~12 m 目标。
- **建议**：若固定 target_distance，现有尺度合理；若目标距离变化较大，可考虑按 target_distance 动态归一化。

---

## 四、优先尝试的 Relax 方向（消融建议）

1. **到达精度**：`success_threshold` 从 0.5 降至 0.3 m（独立消融）。
2. **奖励与惩罚**：backward_penalty、heading_coef、heading_penalty_wrong 的进一步调参。
3. **网络结构**：更大/更深的 MLP、不同 hidden sizes 的消融。
4. **训练超参**：lr、ent_coef、n_envs、n_steps 的消融。

roll/pitch 约束、动作空间等结构类简化，建议在到达成功率与精度稳定后再考虑。

---

## 五、代码位置速查

| 文件 | 内容 |
|------|------|
| `core/dynamics/auv_dynamics.py` | 6DOF 动力学，RK4 积分 |
| `core/control/dual_loop_pid.py` | 双环 PID + IntegratedController |
| `core/environment/low_level_env.py` | 底层环境，IntegratedController + AUVDynamics |
| `core/environment/low_level_env_3d.py` | 3D 包装 v1 |
| `core/environment/low_level_env_3d_v2.py` | 3D 包装 v2（含 backward_penalty） |
| `training/train_3d_v2.py` | 3D v2 训练脚本 |
