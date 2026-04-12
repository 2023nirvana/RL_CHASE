#!/bin/bash
# IQL 离线预训练
cd /root/autodl-tmp/pursue_20260302

python training/iql_pretrain.py \
    --dataset data/expert_trajectories/expert_dataset.npz \
    --experiment-name iql_phase1_v1 \
    --n-steps 150000 \
    --batch-size 256 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --value-lr 3e-4 \
    --gamma 0.99 \
    --tau 0.005 \
    --expectile 0.7 \
    --beta 3.0 \
    --normalize-reward \
    --eval-every 5000 \
    --eval-episodes 15 \
    --low-level-ckpt low_near/v4_tight_heading_terminal/checkpoints/best.pth \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --evader-mode medium \
    --seed 42
