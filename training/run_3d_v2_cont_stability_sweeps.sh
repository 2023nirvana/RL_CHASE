#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 用法：
#   bash training/run_3d_v2_cont_stability_sweeps.sh
# 可按需修改 total timesteps / n-envs。

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1200000}
N_ENVS=${N_ENVS:-4}

run_case () {
  local exp_name="$1"
  shift
  echo "[RUN] ${exp_name}"
  python training/train_low_level_3d_v2_continuous.py \
    --experiment-name "${exp_name}" \
    --config configs/low_level_config_2d.yaml \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --n-envs "${N_ENVS}" \
    "$@"

  python training/visualize_low_level_3d_v2_continuous_enhanced.py \
    --ckpt-dir "checkpoints/${exp_name}" \
    --train-log-dir "visualization/logs/${exp_name}" \
    --out-dir "visualization/logs/${exp_name}_enhanced" \
    --episodes 12 \
    --gif-count 3 \
    --target-distance 12 \
    --z-min 1.0 \
    --z-max 6.0 \
    --min-z-gap 1.2 \
    --z-visual-scale 2.2
}

# A: 稳定优先（建议先跑）
run_case "3d_v2_cont_stab_a" \
  --success-radius 0.65 \
  --heading-coef 0.45 \
  --direction-coef 0.55 \
  --heading-bonus-coef 0.05 \
  --improvement-coef 6.0 \
  --improvement-clip 0.08 \
  --yaw-rate-penalty-coef 0.05 \
  --yaw-action-penalty-coef 0.03 \
  --yaw-action-flip-penalty-coef 0.05 \
  --action-smooth-coef 0.04 \
  --z-lock-coef 0.8

# B: 平衡型（保留一定机动）
run_case "3d_v2_cont_stab_b" \
  --success-radius 0.60 \
  --heading-coef 0.50 \
  --direction-coef 0.50 \
  --heading-bonus-coef 0.06 \
  --improvement-coef 7.0 \
  --improvement-clip 0.10 \
  --yaw-rate-penalty-coef 0.045 \
  --yaw-action-penalty-coef 0.025 \
  --yaw-action-flip-penalty-coef 0.04 \
  --action-smooth-coef 0.03 \
  --z-lock-coef 0.7

# C: 成功率优先（更宽容终止）
run_case "3d_v2_cont_stab_c" \
  --success-radius 0.75 \
  --heading-coef 0.40 \
  --direction-coef 0.60 \
  --heading-bonus-coef 0.04 \
  --improvement-coef 5.5 \
  --improvement-clip 0.07 \
  --yaw-rate-penalty-coef 0.055 \
  --yaw-action-penalty-coef 0.035 \
  --yaw-action-flip-penalty-coef 0.06 \
  --action-smooth-coef 0.05 \
  --z-lock-coef 0.9

echo "All sweep runs completed."
