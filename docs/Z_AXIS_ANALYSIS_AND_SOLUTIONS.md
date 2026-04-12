# 3D 消融实验：Z 轴（深度）控制不佳分析与改进方案

## 一、十组消融实验结果概况

| 组别 | 名称 | 最终 eval success_rate | 备注 |
|------|------|------------------------|------|
| 1 | bp06 | ~30–70% 波动 | backward_penalty=0.6 |
| 2 | bp04 | ~30–70% 波动 | backward_penalty=0.4 |
| 3 | h08 | ~30–90% 波动 | heading_coef=0.8 |
| 4 | wp02 | ~20–80% 波动 | heading_penalty_wrong=0.2 |
| 5 | prec03 | ~20–70% 波动 | success_threshold=0.3 |
| 6 | net_large | ~30–80% 波动 | 网络 [512,256,128] |
| 7 | net_deep | ~30–70% 波动 | 网络 [256,256,128,64] |
| 8 | lr_low | 类似 | lr=1e-4 |
| 9 | ent08 | 类似 | ent_coef=0.08 |
| 10 | nenvs16 | 类似 | n_envs=16 |

整体成功率在 30–70% 区间波动，且 eval 曲线不够稳定。

---

## 二、观察到的现象

1. **XY 平面已到位，Z 未到位**  
   俯视图（XY / heading）显示智能体已接近目标，但深度未收敛。
2. **到达 XY 后剧烈振荡**  
   XY 接近目标后，轨迹在水平面附近振荡，难以稳定到目标深度。
3. **Z 偏差偏大**  
   即使“成功”到达，Z 方向仍有明显偏差。
4. **垂直推进器利用不足**  
   策略更偏向 forward/sway/yaw，heave_up/heave_down 使用较少。

---

## 三、根因分析（多角度）

### 3.1 奖励结构：Z 分量权重过低

| 奖励项 | 系数 | 主要作用维度 |
|--------|------|--------------|
| heading_coef | 0.7 | XY（yaw） |
| direction_coef | 0.6 | XY（前进速度） |
| z_direction | 0.5 | Z（heave 速度） |
| position | -0.3 | 3D（pos_error） |
| improvement | 10.0 | 3D |

- XY 有明确的 heading 对齐奖励和 direction 速度奖励。
- Z 只有 `z_direction_reward = 0.5 * z_vel`，且仅在 `abs(z_error) > 0.1` 时生效。
- 相比 XY 的 heading + direction 双重激励，Z 的激励明显更弱，策略易偏向 XY。

### 3.2 奖励时序：XY 优先、Z 滞后

- 早期：XY 误差大，heading + direction 占主导，策略先学“对准并前进”。
- 后期：XY 接近目标后，应切换到 heave，但：
  - 没有“XY 足够接近时加强 Z 奖励”的机制；
  - `improvement_reward` 用 3D `pos_error`，XY 小幅振荡也会带来变化，可能掩盖 Z 的贡献；
  - 时间惩罚 `-0.02` 与弱 Z 奖励叠加，易导致策略在 XY 附近“打转”而不主动下潜/上浮。

### 3.3 观测空间：Z 误差相对不突出

```python
# low_level_env_3d_v2.py _get_obs
pos_error_body_x / 10.0,
pos_error_body_y / 10.0,
pos_error_body_z / 5.0,   # Z 用 5.0 归一化
```

- XY 误差典型范围 ~12 m，Z 范围 2–4 m，误差量级本身不同。
- Z 用 5.0 归一化，XY 用 10.0，在相似误差下 Z 的观测值会偏大，但：
  - 策略更早依赖 XY 信息（heading、direction），易形成 XY 主导的表示；
  - 没有单独的“当前主要误差在 Z”的显式特征。

### 3.4 动作空间：heave 与 XY 动作竞争

- 8 动作：forward, backward, left, right, yaw_left, yaw_right, heave_up, heave_down。
- XY 有 6 个动作，Z 只有 2 个；且 forward 等动作对 `pos_error` 的改善更直接。
- 当 XY 接近目标时，策略更倾向微调 yaw/sway 来“对准”，而不是切换为 heave。

### 3.5 动力学：heave 响应可能偏慢

- `configs/low_level_config_2d.yaml` 中 `heave: 0.0`（2D 不用），3D 通常用 `low_level_config.yaml` 的 `heave: 0.04`。
- surge/sway 的 `action_distance` 为 0.03，heave 为 0.04，但 heave 可能受附加质量、阻尼等影响，响应更慢。
- 若 heave 响应慢，策略更可能“放弃”深度控制，专注 XY。

### 3.6 振荡成因：XY 对准奖励的副作用

- `strict_direction_heading` 要求 heading_error < 30° 才给 direction_reward。
- 接近目标时，微小 yaw 偏差就会失去 direction 奖励，策略不断微调 yaw。
- 这种微调叠加动力学滞后，易产生 XY 平面振荡；振荡期间 Z 误差可能变化不大，Z 奖励信号弱，进一步削弱 heave 使用。

---

## 四、改进方案（按优先级）

### 方案 A：奖励层面 —— 提高 Z 权重与阶段性激励（推荐）

1. **提高 z_direction 系数**  
   - 将 `z_direction_reward` 系数从 0.5 提到 0.8–1.0，或增加可调 `z_coef` 参数。

2. **XY 接近时加强 Z 奖励（阶段性）**  
   - 当 `xy_error < 1.0`（或 0.5）时，将 Z 相关奖励乘以 1.5–2.0，引导策略在 XY 到位后优先调整深度。

3. **增加 Z 专用 proximity 奖励**  
   - 类似 XY 的 proximity_reward，当 `abs(z_error) < 0.5` 时给额外奖励，鼓励最终收敛到目标深度。

### 方案 B：奖励层面 —— 解耦 XY 与 Z 的 improvement

1. **分轴 improvement_reward**  
   - 将 improvement 拆成 `improvement_xy` 与 `improvement_z`，分别用 `xy_error` 和 `abs(z_error)` 计算，并为 `improvement_z` 设单独系数（如 1.5–2.0），避免被 XY 主导。

2. **Z 改进额外奖励**  
   - 当 `z_error` 减小且 `xy_error` 已较小时，额外奖励，强化“先 XY 后 Z”的策略。

### 方案 C：观测层面 —— 突出 Z 状态

1. **显式 Z 主导标志**  
   - 增加特征：`z_error / (xy_error + 1e-6)`，或 `1 if abs(z_error) > xy_error else 0`，让网络意识到“当前主要误差在 Z”。

2. **调整 Z 观测归一化**  
   - 将 `pos_error_body_z / 5.0` 改为与 XY 一致的尺度（如 `/ 10.0`），或根据 z_range 做专门归一化，使 Z 误差在观测中更易被利用。

### 方案 D：课程 / 阶段性训练

1. **两阶段课程**  
   - Stage 1：成功条件放宽为 `xy_error < 0.5`（不要求 Z），专门学 XY 到达；
   - Stage 2：在 Stage 1 策略基础上，收紧为 3D `pos_error < 0.5`，重点学 heave。

2. **渐进 Z 范围**  
   - 早期缩小 z_range，减少 Z 误差，先建立 heave 使用习惯；后期再扩大 z_range。

### 方案 E：动作 / 物理层面

1. **增强 heave 推力**  
   - 在 `configs` 中提高 heave 的 `action_distance` 或推力，使 heave 响应更明显。

2. **复合动作（可选）**  
   - 增加 “forward+heave_up”“forward+heave_down” 等复合动作，减少“先 XY 再 Z”的切换成本（改动较大）。

### 方案 F：振荡抑制

1. **振荡惩罚**  
   - 检测 `pos_error` 在近 N 步内无明显下降或反复增大，给予小幅惩罚，鼓励稳定收敛。

2. **放宽 strict_direction_heading**  
   - 当 `xy_error < 0.5` 时，放宽到 45° 或 60°，减少无谓的 yaw 微调，为 heave 留出动作空间。

---

## 五、推荐实施顺序

1. **先做奖励改动（A、B）**  
   - 提高 `z_direction_coef`；  
   - XY 接近时加强 Z 奖励；  
   - 可选：分轴 improvement 或 Z proximity。

2. **再尝试观测改动（C）**  
   - 增加 Z 主导标志或调整 Z 归一化。

3. **若仍不足，再考虑课程（D）和物理/动作（E、F）**  
   - 课程需改训练流程；  
   - 物理与动作改动需验证动力学与安全性。

---

## 六、已实施改动（2025-02）

方案 1、2、4 已并入 `low_level_env_3d_v2.py` 和 `low_level_env_3d_v2_ablation.py`，默认开启：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `z_coef` | 1.0 | Z 方向奖励系数（原 0.5） |
| `z_boost_when_xy_close` | 1.5 | XY 接近时 Z 奖励放大倍数 |
| `xy_close_threshold` | 1.0 | 判定「XY 接近」的阈值 (m) |
| `relaxed_heading_when_xy_close` | True | XY 接近时放宽 heading 到 60° |

如需恢复旧行为：`z_coef=0.5`, `z_boost_when_xy_close=1.0`, `relaxed_heading_when_xy_close=False`。
