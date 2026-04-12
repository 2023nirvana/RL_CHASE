#!/bin/bash
# AWAC 离线预训练
cd /root/autodl-tmp/pursue_20260302

python training/awac_pretrain.py \
    --dataset data/expert_trajectories/expert_dataset.npz \
    --experiment-name awac_phase1_v1 \
    --n-steps 150000 \
    --batch-size 256 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --beta 1.0 \
    --eval-every 5000 \
    --eval-episodes 15 \
    --low-level-ckpt low_near/v4_tight_heading_terminal/checkpoints/best.pth \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --evader-mode medium \
    --seed 42
