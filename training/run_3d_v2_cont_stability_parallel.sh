#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 用法示例：
#   MAX_PARALLEL=3 TOTAL_TIMESTEPS=2000000 N_ENVS=4 bash training/run_3d_v2_cont_stability_parallel.sh
# 说明：
#   - 只跑训练，不做可视化
#   - 每组日志写入 visualization/logs/<exp_name>/train.out

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-2000000}
N_ENVS=${N_ENVS:-4}
MAX_PARALLEL=${MAX_PARALLEL:-3}

launch_case () {
  local exp_name="$1"
  shift

  local log_dir="visualization/logs/${exp_name}"
  mkdir -p "${log_dir}"

  echo "[LAUNCH] ${exp_name}"
  nohup python training/train_low_level_3d_v2_continuous.py \
    --experiment-name "${exp_name}" \
    --config configs/low_level_config_2d.yaml \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --n-envs "${N_ENVS}" \
    --target-distance 12 \
    --z-min 1.0 \
    --z-max 6.0 \
    "$@" > "${log_dir}/train.out" 2>&1 &
}

wait_for_slot () {
  while true; do
    local running
    running=$(jobs -pr | wc -l)
    if [[ "${running}" -lt "${MAX_PARALLEL}" ]]; then
      break
    fi
    sleep 2
  done
}

# A: 大改平衡基线
wait_for_slot
launch_case "3d_v2_cont_big_a" \
  --success-radius 0.65 \
  --min-target-z-gap 1.2 \
  --z-progress-coef 1.4 \
  --z-progress-clip 0.08 \
  --xy-gate-strictness 1.0 \
  --spiral-penalty-coef 0.30 \
  --z-coef 1.8 \
  --z-boost-when-xy-close 2.4 \
  --xy-close-threshold 1.2 \
  --heading-coef 0.42 \
  --direction-coef 0.48 \
  --heading-bonus-coef 0.05 \
  --improvement-coef 6.0 \
  --improvement-clip 0.08 \
  --yaw-rate-penalty-coef 0.05 \
  --yaw-action-penalty-coef 0.03 \
  --yaw-action-flip-penalty-coef 0.055 \
  --action-smooth-coef 0.045 \
  --z-lock-coef 0.9

# B: 强Z驱动
wait_for_slot
launch_case "3d_v2_cont_big_b" \
  --success-radius 0.60 \
  --min-target-z-gap 1.4 \
  --z-progress-coef 1.8 \
  --z-progress-clip 0.10 \
  --xy-gate-strictness 1.1 \
  --spiral-penalty-coef 0.40 \
  --z-coef 2.2 \
  --z-boost-when-xy-close 2.8 \
  --xy-close-threshold 1.4 \
  --heading-coef 0.38 \
  --direction-coef 0.44 \
  --heading-bonus-coef 0.04 \
  --improvement-coef 5.5 \
  --improvement-clip 0.07 \
  --yaw-rate-penalty-coef 0.055 \
  --yaw-action-penalty-coef 0.035 \
  --yaw-action-flip-penalty-coef 0.06 \
  --action-smooth-coef 0.05 \
  --z-lock-coef 1.1

# C: 成功率优先
wait_for_slot
launch_case "3d_v2_cont_big_c" \
  --success-radius 0.70 \
  --min-target-z-gap 1.0 \
  --z-progress-coef 1.3 \
  --z-progress-clip 0.08 \
  --xy-gate-strictness 0.9 \
  --spiral-penalty-coef 0.28 \
  --z-coef 1.6 \
  --z-boost-when-xy-close 2.1 \
  --xy-close-threshold 1.0 \
  --heading-coef 0.46 \
  --direction-coef 0.52 \
  --heading-bonus-coef 0.06 \
  --improvement-coef 6.5 \
  --improvement-clip 0.09 \
  --yaw-rate-penalty-coef 0.045 \
  --yaw-action-penalty-coef 0.025 \
  --yaw-action-flip-penalty-coef 0.045 \
  --action-smooth-coef 0.04 \
  --z-lock-coef 0.8

# D: 抑制摆头加强
wait_for_slot
launch_case "3d_v2_cont_big_d" \
  --success-radius 0.65 \
  --min-target-z-gap 1.2 \
  --z-progress-coef 1.5 \
  --z-progress-clip 0.09 \
  --xy-gate-strictness 1.0 \
  --spiral-penalty-coef 0.45 \
  --z-coef 1.9 \
  --z-boost-when-xy-close 2.3 \
  --xy-close-threshold 1.2 \
  --heading-coef 0.40 \
  --direction-coef 0.50 \
  --heading-bonus-coef 0.05 \
  --improvement-coef 6.0 \
  --improvement-clip 0.08 \
  --yaw-rate-penalty-coef 0.06 \
  --yaw-action-penalty-coef 0.04 \
  --yaw-action-flip-penalty-coef 0.07 \
  --action-smooth-coef 0.055 \
  --z-lock-coef 1.0

# E: 终止更宽松
wait_for_slot
launch_case "3d_v2_cont_big_e" \
  --success-radius 0.78 \
  --min-target-z-gap 1.1 \
  --z-progress-coef 1.2 \
  --z-progress-clip 0.07 \
  --xy-gate-strictness 0.95 \
  --spiral-penalty-coef 0.25 \
  --z-coef 1.7 \
  --z-boost-when-xy-close 2.0 \
  --xy-close-threshold 1.1 \
  --heading-coef 0.36 \
  --direction-coef 0.56 \
  --heading-bonus-coef 0.04 \
  --improvement-coef 5.8 \
  --improvement-clip 0.08 \
  --yaw-rate-penalty-coef 0.05 \
  --yaw-action-penalty-coef 0.03 \
  --yaw-action-flip-penalty-coef 0.05 \
  --action-smooth-coef 0.045 \
  --z-lock-coef 0.85

# F: 去掉近距离朝向放松
wait_for_slot
launch_case "3d_v2_cont_big_f" \
  --success-radius 0.66 \
  --min-target-z-gap 1.2 \
  --z-progress-coef 1.6 \
  --z-progress-clip 0.09 \
  --xy-gate-strictness 1.2 \
  --spiral-penalty-coef 0.38 \
  --z-coef 2.0 \
  --z-boost-when-xy-close 2.5 \
  --xy-close-threshold 1.2 \
  --disable-relaxed-heading-when-xy-close \
  --heading-coef 0.50 \
  --direction-coef 0.46 \
  --heading-bonus-coef 0.05 \
  --improvement-coef 6.2 \
  --improvement-clip 0.09 \
  --yaw-rate-penalty-coef 0.05 \
  --yaw-action-penalty-coef 0.03 \
  --yaw-action-flip-penalty-coef 0.055 \
  --action-smooth-coef 0.05 \
  --z-lock-coef 1.0

# G: 俯仰能力加强（更大 pitch 余量）
wait_for_slot
launch_case "3d_v2_cont_big_g" \
  --success-radius 0.64 \
  --pitch-limit-deg 35 \
  --min-target-z-gap 1.4 \
  --z-progress-coef 1.8 \
  --z-progress-clip 0.10 \
  --xy-gate-strictness 1.1 \
  --spiral-penalty-coef 0.42 \
  --z-coef 2.1 \
  --z-boost-when-xy-close 2.7 \
  --xy-close-threshold 1.3 \
  --heading-coef 0.34 \
  --direction-coef 0.42 \
  --heading-bonus-coef 0.03 \
  --improvement-coef 5.0 \
  --improvement-clip 0.07 \
  --yaw-rate-penalty-coef 0.06 \
  --yaw-action-penalty-coef 0.04 \
  --yaw-action-flip-penalty-coef 0.07 \
  --action-smooth-coef 0.06 \
  --z-lock-coef 1.2

# H: 保守稳态（防抖优先）
wait_for_slot
launch_case "3d_v2_cont_big_h" \
  --success-radius 0.68 \
  --min-target-z-gap 1.0 \
  --z-progress-coef 1.1 \
  --z-progress-clip 0.06 \
  --xy-gate-strictness 1.0 \
  --spiral-penalty-coef 0.35 \
  --z-coef 1.5 \
  --z-boost-when-xy-close 1.9 \
  --xy-close-threshold 1.0 \
  --heading-coef 0.44 \
  --direction-coef 0.46 \
  --heading-bonus-coef 0.04 \
  --improvement-coef 5.8 \
  --improvement-clip 0.07 \
  --yaw-rate-penalty-coef 0.065 \
  --yaw-action-penalty-coef 0.045 \
  --yaw-action-flip-penalty-coef 0.075 \
  --action-smooth-coef 0.065 \
  --z-lock-coef 1.1

# I: 激进机动（验证上限）
wait_for_slot
launch_case "3d_v2_cont_big_i" \
  --success-radius 0.60 \
  --pitch-limit-deg 35 \
  --min-target-z-gap 1.5 \
  --z-progress-coef 2.0 \
  --z-progress-clip 0.12 \
  --xy-gate-strictness 1.3 \
  --spiral-penalty-coef 0.30 \
  --z-coef 2.4 \
  --z-boost-when-xy-close 3.0 \
  --xy-close-threshold 1.5 \
  --heading-coef 0.30 \
  --direction-coef 0.40 \
  --heading-bonus-coef 0.02 \
  --improvement-coef 5.0 \
  --improvement-clip 0.06 \
  --yaw-rate-penalty-coef 0.04 \
  --yaw-action-penalty-coef 0.025 \
  --yaw-action-flip-penalty-coef 0.04 \
  --action-smooth-coef 0.035 \
  --z-lock-coef 0.9

echo "All jobs launched. Current background jobs:"
jobs -l

echo "Tip: monitor with: tail -f visualization/logs/3d_v2_cont_big_a/train.out"
