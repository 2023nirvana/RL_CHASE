#!/bin/bash
# Online PPO v2: 从 TD3+BC v4 预训练恢复, 使用改进的超参
set -e
cd "$(dirname "$0")/.."

LOW_CKPT="low_near/v4_tight_heading_terminal/checkpoints/best.pth"
PRETRAIN="checkpoints/td3bc_v4_2m/best.pth"
RESUME="checkpoints/online_ppo_td3bc_v3/latest.pth"

echo "===== Online PPO v2 (TD3+BC v3, improved) ====="

python -u training/online_ppo_finetune.py \
    --low-level-ckpt "$LOW_CKPT" \
    --pretrain-ckpt "$PRETRAIN" \
    --resume "$RESUME" \
    --experiment-name "online_ppo_td3bc_v3" \
    --hidden 256 128 \
    --total-timesteps 500000 \
    --n-steps 256 \
    --batch-size 256 \
    --n-epochs 3 \
    --lr 1e-5 \
    --lr-end 1e-6 \
    --clip-range 0.1 \
    --beta-kl 0.3 \
    --kl-anneal-frac 0.3 \
    --eval-freq 10000 \
    --eval-episodes 20 \
    --save-freq 50000 \
    --evader-mode medium \
    --world-size 30.0 \
    --catch-radius 1.0 \
    --subgoal-range 0.7 \
    --low-steps 50 \
    --max-steps 6000 \
    --normalize-reward \
    --num-envs 4 \
    --r-catch 150.0 \
    --r-fail -50.0 \
    --alpha-shaping 5.0 \
    --c-sg 10.0 \
    --c-path 5.0 \
    --lam-time 0.02 \
    --seed 42 \
    2>&1 | tee logs/online_ppo_td3bc_v3_cont.log

echo "===== TD3+BC v3 Online PPO Done ====="
