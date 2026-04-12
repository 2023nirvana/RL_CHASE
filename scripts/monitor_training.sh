#!/bin/bash
# 监控训练进度脚本
# 用法: ./scripts/monitor_training.sh

echo "=========================================="
echo "Training Progress Monitor"
echo "=========================================="
echo ""

# 检查进程
echo "Running processes:"
ps aux | grep train_low_level_2d | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
echo ""

# 1m 训练
echo "=== 1m Distance Training ==="
if [ -f /tmp/train_1m.log ]; then
    grep -E "Update|Success Rate|Best" /tmp/train_1m.log | tail -10
else
    echo "  Log not found"
fi
echo ""

# 2m 训练
echo "=== 2m Distance Training ==="
if [ -f /tmp/train_2m.log ]; then
    grep -E "Update|Success Rate|Best" /tmp/train_2m.log | tail -10
else
    echo "  Log not found"
fi
echo ""

# 5m 训练
echo "=== 5m Distance Training ==="
if [ -f /tmp/train_5m.log ]; then
    grep -E "Update|Success Rate|Best" /tmp/train_5m.log | tail -10
else
    echo "  Log not found"
fi
echo ""

# 检查模型
echo "=== Saved Models ==="
ls -la /root/autodl-tmp/HRL_Control/backup_v0.1_discrete_action/UnderwaterPursuitEvasion/checkpoints/ | grep "2d_dist"
