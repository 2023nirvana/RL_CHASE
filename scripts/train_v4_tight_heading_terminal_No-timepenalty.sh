#!/bin/bash
# v4_tight_heading_terminal_No-timepenalty
# 基于 v4_tight_heading_terminal，取消时间惩罚 (reward_time_penalty=0.0)
# 保留其他三项奖励: 到达奖励+速度奖金、距离改善奖励、航向终端缩放
# 目的: 观察去掉时间惩罚后能否收敛，以及最终朝向是否改善

cd "$(dirname "$0")/.." || exit 1

nohup python training/train_2d_v3_continuous.py \
    --experiment-name v4_tight_heading_terminal_No-timepenalty \
    --config configs/low_level_config_2d.yaml \
    --reach-threshold 0.10 \
    --reward-heading-terminal 0.5 \
    --reward-time-penalty 0.0 \
    --init-velocity-range 1.0 \
    --seed 42 \
    > low_near/v4_tight_heading_terminal_No-timepenalty_train.log 2>&1 &

echo "PID: $!"
echo "日志: low_near/v4_tight_heading_terminal_No-timepenalty_train.log"
echo "监控: tail -f low_near/v4_tight_heading_terminal_No-timepenalty_train.log"
