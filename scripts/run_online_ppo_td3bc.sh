#!/bin/bash
# Phase 2: Online PPO 微调 (用 TD3+BC v1 最好的 checkpoint)
set -e
cd "$(dirname "$0")/.."

LOW_CKPT="low_near/v4_tight_heading_terminal/checkpoints/best.pth"
PRETRAIN="checkpoints/td3bc_phase1/best.pth"

echo "===== Online PPO Fine-tune (from TD3+BC v1) ====="
echo "Pretrained: $PRETRAIN"
echo ""

python -u training/online_ppo_finetune.py \
    --low-level-ckpt "$LOW_CKPT" \
    --pretrain-ckpt "$PRETRAIN" \
    --experiment-name "online_ppo_td3bc_v1" \
    --hidden 256 128 \
    --total-timesteps 500000 \
    --n-steps 256 \
    --batch-size 256 \
    --n-epochs 4 \
    --lr 3e-5 \
    --clip-range 0.15 \
    --beta-kl 0.5 \
    --kl-anneal-frac 0.5 \
    --eval-freq 10000 \
    --eval-episodes 10 \
    --save-freq 50000 \
    --evader-mode medium \
    --world-size 30.0 \
    --catch-radius 1.0 \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --max-steps 6000 \
    --seed 42

echo ""
echo "===== Online PPO 训练完成 ====="
echo "Checkpoints: checkpoints/online_ppo_td3bc_v1/"
