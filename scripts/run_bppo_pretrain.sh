#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "===== BPPO Offline Pretrain ====="
echo "Phase 1: V(100k) + Q(100k) → Phase 2: BC(100k) → Phase 3: BPPO(1000)"
echo ""

python -u training/bppo_pretrain.py \
    --dataset data/expert_trajectories/expert_dataset.npz \
    --experiment-name bppo_phase1_v2 \
    --normalize-reward \
    --v-steps 100000 \
    --q-steps 100000 \
    --bc-steps 100000 \
    --bppo-steps 1000 \
    --batch-size 256 \
    --value-lr 3e-4 \
    --critic-lr 3e-4 \
    --bc-lr 3e-4 \
    --bppo-lr 1e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --clip-ratio 0.25 \
    --entropy-weight 0.01 \
    --decay 0.96 \
    --omega 0.9 \
    --is-clip-decay \
    --is-lr-decay \
    --is-update-old-policy \
    --bppo-eval-every 200 \
    --eval-every 10000 \
    --eval-episodes 15 \
    --print-every 5000 \
    --save-every 50000 \
    --low-level-ckpt low_near/v4_tight_heading_terminal/checkpoints/best.pth \
    --world-size 30.0 \
    --catch-radius 1.0 \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --evader-mode medium \
    --seed 42

echo ""
echo "===== BPPO 训练完成 ====="
