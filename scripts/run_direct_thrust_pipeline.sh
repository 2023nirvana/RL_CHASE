#!/bin/bash
# 直接推力版 Pipeline: 经典策略采数据 → Offline PPO → Online PPO
# ================================================================
# 用法: bash scripts/run_direct_thrust_pipeline.sh
#
# 三个阶段:
# Phase 1: 用 PID 版模型跑环境，记录 8 维推力作为 expert data
# Phase 2: Offline PPO 在 expert data 上预训练
# Phase 3: Online PPO 在环境中微调

set -e
cd "$(dirname "$0")/.." || exit 1

PID_MODEL="low_near/v4_tight_heading_terminal/checkpoints/best.pth"
EXPERT_DATA="data/expert_direct_thrust/expert_dataset.npz"
OFFLINE_EXP="offline_direct_thrust_v1"
ONLINE_EXP="online_direct_thrust_v1"

echo "=========================================="
echo " Direct Thrust Pipeline"
echo "=========================================="

# ===================== Phase 1 =====================
echo ""
echo "[Phase 1] Collecting expert data with PID model..."

if [ ! -f "$PID_MODEL" ]; then
    echo "ERROR: PID model not found: $PID_MODEL"
    echo "Please train v4_tight_heading_terminal first."
    exit 1
fi

if [ -f "$EXPERT_DATA" ]; then
    echo "  Expert data already exists, skipping."
else
    python training/collect_expert_direct_thrust.py \
        --pid-model "$PID_MODEL" \
        --episodes 2000 \
        --output "$EXPERT_DATA" \
        --only-success \
        --seed 0
    echo "[Phase 1] Done."
fi

# ===================== Phase 2 =====================
echo ""
echo "[Phase 2] Offline PPO pretraining..."

OFFLINE_CKPT="checkpoints/$OFFLINE_EXP/best.pth"
if [ -f "$OFFLINE_CKPT" ]; then
    echo "  Offline checkpoint exists, skipping."
else
    python training/offline_ppo_direct_thrust.py \
        --dataset "$EXPERT_DATA" \
        --experiment-name "$OFFLINE_EXP" \
        --obs-dim 8 \
        --act-dim 8 \
        --hidden 256 128 \
        --lr 3e-4 \
        --n-iters 500 \
        --batch-size 512
    echo "[Phase 2] Done."
fi

# ===================== Phase 3 =====================
echo ""
echo "[Phase 3] Online PPO fine-tuning..."

python training/online_ppo_direct_thrust.py \
    --pretrain-ckpt "$OFFLINE_CKPT" \
    --experiment-name "$ONLINE_EXP" \
    --config configs/low_level_config_2d.yaml \
    --total-timesteps 2000000 \
    --lr 1e-4 \
    --lr-end 1e-5 \
    --beta-kl 0.3 \
    --kl-anneal-frac 0.3 \
    --reach-threshold 0.10 \
    --reward-heading-terminal 0.5 \
    --init-velocity-range 1.0 \
    --thrust-scale 40.0 \
    --seed 42

echo ""
echo "=========================================="
echo " Pipeline Complete!"
echo "=========================================="
echo "  Expert data:    $EXPERT_DATA"
echo "  Offline model:  $OFFLINE_CKPT"
echo "  Online model:   low_near/$ONLINE_EXP/checkpoints/best.pth"
