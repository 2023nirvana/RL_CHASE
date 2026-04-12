#!/bin/bash
# ==============================================================================
# 运行所有水流等级的训练实验
# ==============================================================================

set -e
cd "$(dirname "$0")/.."

echo "=============================================================="
echo "Running all water current level experiments"
echo "=============================================================="

# 参数（可根据需要调整）
EXPERT_EPISODES=3000
OFFLINE_STEPS=1000000
ONLINE_STEPS=200000
EVAL_EPISODES=50

# 依次运行各水流等级
for LEVEL in 0 1 2 3; do
    echo ""
    echo "######################################################"
    echo "Starting Water Current Level ${LEVEL}"
    echo "######################################################"
    echo ""
    
    ./scripts/run_water_current_pipeline.sh \
        "$LEVEL" \
        "$EXPERT_EPISODES" \
        "$OFFLINE_STEPS" \
        "$ONLINE_STEPS" \
        "$EVAL_EPISODES"
    
    echo ""
    echo "Level ${LEVEL} complete!"
    echo ""
done

echo "=============================================================="
echo "All water current experiments complete!"
echo "=============================================================="

# 比较结果
echo ""
echo "Comparing results across water current levels..."
python -c "
import os
import json
import numpy as np

results = []
for level in range(4):
    exp_name = f'water_current_level_{level}_online'
    json_path = f'checkpoints/{exp_name}/eval_visuals/best_eval_results.json'
    
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        summary = data.get('summary', {})
        results.append({
            'level': level,
            'success_rate': summary.get('success_rate', 0) * 100,
            'avg_steps': summary.get('avg_steps_success', 0),
            'avg_reward': summary.get('avg_reward', 0),
        })
    else:
        results.append({'level': level, 'success_rate': None, 'avg_steps': None, 'avg_reward': None})

print()
print('=' * 60)
print('Water Current Experiment Results')
print('=' * 60)
print(f\"{'Level':<10} {'Success%':>12} {'Avg Steps':>12} {'Avg Reward':>12}\")
print('-' * 50)
for r in results:
    if r['success_rate'] is not None:
        print(f\"{r['level']:<10} {r['success_rate']:>11.1f}% {r['avg_steps']:>12.1f} {r['avg_reward']:>12.2f}\")
    else:
        print(f\"{r['level']:<10} {'N/A':>12} {'N/A':>12} {'N/A':>12}\")
print('=' * 60)
"
