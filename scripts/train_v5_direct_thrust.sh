#!/bin/bash
# v5_direct_thrust
# 直接推力版本：RL 输出 8 个推进器推力，跳过 PID 闭环
# 任务场景与 v4_tight_heading_terminal 一致

cd "$(dirname "$0")/.." || exit 1

nohup python training/train_2d_v3_direct_thrust.py \
    --experiment-name v5_direct_thrust \
    --config configs/low_level_config_2d.yaml \
    --reach-threshold 0.10 \
    --reward-heading-terminal 0.5 \
    --reward-time-penalty -0.2 \
    --init-velocity-range 1.0 \
    --thrust-scale 40.0 \
    --total-timesteps 2000000 \
    --seed 42 \
    > low_near/v5_direct_thrust_train.log 2>&1 &

echo "PID: $!"
echo "日志: low_near/v5_direct_thrust_train.log"
echo "监控: tail -f low_near/v5_direct_thrust_train.log"
