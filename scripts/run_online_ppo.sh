#!/bin/bash
# Phase 2: Online PPO 微调
set -e
cd "$(dirname "$0")/.."

LOW_CKPT="low_near/v4_tight_heading_terminal/checkpoints/best.pth"
PRETRAIN="checkpoints/offline_ppo_phase1/best.pth"

python training/online_ppo_finetune.py \
    --low-level-ckpt "$LOW_CKPT" \
    --pretrain-ckpt "$PRETRAIN" \
    --experiment-name "online_ppo_phase2" \
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
    --evader-mode medium \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --seed 42

echo "Done. Checkpoints in checkpoints/online_ppo_phase2/"
