#!/bin/bash
# 高层 Pipeline: 经典策略采数据 → Offline PPO → Online PPO
# =========================================================
# 用法: bash scripts/run_high_level_pipeline.sh [--skip-phase1] [--skip-phase2]

set -e
cd "$(dirname "$0")/.." || exit 1

EXPERT_DATA="data/expert_high_level/expert_dataset.npz"
OFFLINE_EXP="offline_high_level_v1"
ONLINE_EXP="online_high_level_v1"
NN_CKPT="low_near/v5_direct_thrust/checkpoints/best.pth"

SKIP_PHASE1=0
SKIP_PHASE2=0
for arg in "$@"; do
    case $arg in
        --skip-phase1) SKIP_PHASE1=1 ;;
        --skip-phase2) SKIP_PHASE2=1 ;;
    esac
done

echo "=========================================="
echo " High-Level Pipeline"
echo "=========================================="

# ===================== Phase 1 =====================
if [ $SKIP_PHASE1 -eq 0 ] && [ ! -f "$EXPERT_DATA" ]; then
    echo ""
    echo "[Phase 1] DAgger collecting high-level expert data..."
    python -u training/dagger_collect_high_level.py \
        --nn-ckpt "$NN_CKPT" \
        --output "$EXPERT_DATA" \
        --n-rounds 4 \
        --episodes-per-round 500 \
        --beta-decay 0.3 \
        --beta-min 0.1 \
        --only-success-round0 \
        --strategies expert expert_1.0 expert_3.0 lead pn \
        --bc-epochs 60 --bc-lr 1e-4 \
        --evader-mode medium \
        --v-max-pursuer 0.65 --a-max-pursuer 0.5 \
        --v-max-evader 0.85 --a-max-evader 0.25 \
        --seed 0 \
        2>&1 | tee logs/collect_high_level.log
    echo "[Phase 1] Done."
else
    echo "[Phase 1] SKIP (data exists or --skip-phase1)."
fi

# ===================== Phase 2 =====================
OFFLINE_CKPT="checkpoints/$OFFLINE_EXP/best.pth"
if [ $SKIP_PHASE2 -eq 0 ] && [ ! -f "$OFFLINE_CKPT" ]; then
    echo ""
    echo "[Phase 2] Offline PPO pretraining..."
    python -u training/offline_ppo_high_level.py \
        --dataset "$EXPERT_DATA" \
        --experiment-name "$OFFLINE_EXP" \
        --obs-dim 21 --act-dim 2 \
        --hidden 256 128 \
        --lr 1e-4 \
        --n-iters 500 \
        --n-epochs 2 \
        --batch-size 512 \
        --bc-coef 0.5 \
        --max-grad-norm 0.25 \
        --normalize-reward \
        --target-kl 0.05 \
        2>&1 | tee logs/offline_high_level.log
    echo "[Phase 2] Done."
else
    echo "[Phase 2] SKIP (ckpt exists or --skip-phase2)."
fi

# ===================== Phase 3 =====================
echo ""
echo "[Phase 3] Online PPO fine-tuning..."
python -u training/online_ppo_high_level.py \
    --pretrain-ckpt "$OFFLINE_CKPT" \
    --experiment-name "$ONLINE_EXP" \
    --nn-ckpt "$NN_CKPT" \
    --total-timesteps 500000 \
    --n-steps 1024 \
    --batch-size 256 \
    --n-epochs 4 \
    --hidden 256 128 \
    --lr 1e-4 --lr-end 1e-5 \
    --beta-kl 0.3 --kl-anneal-frac 0.3 \
    --evader-mode medium \
    --v-max-pursuer 0.65 --a-max-pursuer 0.5 \
    --v-max-evader 0.85 --a-max-evader 0.25 \
    --log-every 5 --eval-every 20 --save-every 50 --eval-episodes 20 \
    --seed 42 \
    2>&1 | tee logs/online_high_level.log

echo ""
echo "=========================================="
echo " Pipeline Complete"
echo "=========================================="
echo "  Expert data:    $EXPERT_DATA"
echo "  Offline model:  $OFFLINE_CKPT"
echo "  Online model:   low_near/$ONLINE_EXP/checkpoints/best.pth"
