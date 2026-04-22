#!/bin/bash
# 守护脚本：等待 Phase 1 数据就绪，然后自动跑 Phase 2 + Phase 3
set -e
cd "$(dirname "$0")/.." || exit 1

EXPERT_DATA="data/expert_high_level/expert_dataset.npz"
COLLECT_LOG="logs/collect_high_level.log"
OFFLINE_EXP="offline_high_level_v1"
ONLINE_EXP="online_high_level_v1"
OFFLINE_CKPT="checkpoints/$OFFLINE_EXP/best.pth"
NN_CKPT="low_near/v5_direct_thrust/checkpoints/best.pth"

echo "=========================================="
echo " Pipeline Watcher Started: $(date)"
echo "=========================================="

# --- 等待 Phase 1 完成 ---
echo "[watch] Waiting for $EXPERT_DATA ..."
while [ ! -f "$EXPERT_DATA" ]; do
    if [ -f "$COLLECT_LOG" ]; then
        echo "[watch] $(date +%H:%M:%S) | last: $(tail -1 $COLLECT_LOG)"
    fi
    sleep 120
done
echo "[watch] Phase 1 data ready at $(date): $EXPERT_DATA"
ls -lh "$EXPERT_DATA"

# --- Phase 2 ---
echo ""
echo "=========================================="
echo " [Phase 2] Offline PPO @ $(date)"
echo "=========================================="
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

if [ ! -f "$OFFLINE_CKPT" ]; then
    echo "[watch] ERROR: offline ckpt not created!"
    exit 1
fi

# --- Phase 3 ---
echo ""
echo "=========================================="
echo " [Phase 3] Online PPO @ $(date)"
echo "=========================================="
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
echo " Pipeline Complete @ $(date)"
echo "=========================================="
echo "  Offline: $OFFLINE_CKPT"
echo "  Online : low_near/$ONLINE_EXP/checkpoints/best.pth"
