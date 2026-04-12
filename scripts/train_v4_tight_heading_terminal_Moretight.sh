#!/bin/bash
# v4_tight_heading_terminal_Moretight
# 基于 v4_tight_heading_terminal，将捕获阈值从 0.10m 缩小到 0.05m
# 其余参数与原模型完全一致

cd "$(dirname "$0")/.." || exit 1

nohup python training/train_2d_v3_continuous.py \
    --experiment-name v4_tight_heading_terminal_Moretight \
    --config configs/low_level_config_2d.yaml \
    --reach-threshold 0.05 \
    --reward-heading-terminal 0.5 \
    --reward-time-penalty -0.2 \
    --init-velocity-range 1.0 \
    --seed 42 \
    > low_near/v4_tight_heading_terminal_Moretight_train.log 2>&1 &

echo "PID: $!"
echo "日志: low_near/v4_tight_heading_terminal_Moretight_train.log"
echo "监控: tail -f low_near/v4_tight_heading_terminal_Moretight_train.log"
