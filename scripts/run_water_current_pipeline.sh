#!/bin/bash
# ==============================================================================
# 水流干扰训练 - 完整流程
# ==============================================================================
# 本脚本执行带水流干扰的完整HRL训练流程：
# 1. Phase 1: 专家数据收集 (底层RL + 高层专家策略)
# 2. Phase 2: Offline预训练 (TD3+BC)
# 3. Phase 3: Online微调 (PPO)
# 4. 评估并生成可视化
# ==============================================================================

set -e
cd "$(dirname "$0")/.."

# 参数
CURRENT_LEVEL=${1:-1}  # 水流等级: 0=无, 1=弱, 2=中, 3=强
TOTAL_EXPERT_EPISODES=${2:-5000}
OFFLINE_STEPS=${3:-2000000}
ONLINE_STEPS=${4:-300000}
EVAL_EPISODES=${5:-100}

# 路径
LOW_CKPT="low_near/v4_tight_heading_terminal/checkpoints/best.pth"
EXPERIMENT_NAME="water_current_level_${CURRENT_LEVEL}"
CURRENT_CONFIG="configs/current_level_${CURRENT_LEVEL}.yaml"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="logs"

# 计算每个策略的episode数 (5个策略均分)
EPISODES_PER_STRATEGY=$((TOTAL_EXPERT_EPISODES / 5))

echo "=============================================================="
echo "Water Current Training Pipeline"
echo "=============================================================="
echo "Current Level: ${CURRENT_LEVEL}"
echo "Current Config: ${CURRENT_CONFIG}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Episodes per strategy: ${EPISODES_PER_STRATEGY}"
echo "=============================================================="

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# ==============================================================================
# Phase 1: 专家数据收集
# ==============================================================================
echo ""
echo "[Phase 1] Collecting expert trajectories..."
echo "=============================================================="

EXPERT_DATA_DIR="${CHECKPOINT_DIR}/expert_data"
mkdir -p "$EXPERT_DATA_DIR"

python -u training/collect_expert_trajectories.py \
    --low-level-ckpt "$LOW_CKPT" \
    --output-dir "$EXPERT_DATA_DIR" \
    --episodes-per-strategy "$EPISODES_PER_STRATEGY" \
    --evader-mode medium \
    --world-size 30.0 \
    --catch-radius 1.0 \
    --subgoal-range 0.7 \
    --seed 42 \
    --strategies lead pn apn cb expert \
    2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_phase1.log"

echo "[Phase 1] Expert data collection complete!"

# ==============================================================================
# Phase 2: Offline预训练 (TD3+BC)
# ==============================================================================
echo ""
echo "[Phase 2] Offline pretraining with TD3+BC..."
echo "=============================================================="

# 找到收集的数据集文件
DATASET_FILE=$(ls -t "${EXPERT_DATA_DIR}"/*.npz 2>/dev/null | head -n1)
if [ -z "$DATASET_FILE" ]; then
    echo "Error: No dataset found in ${EXPERT_DATA_DIR}"
    exit 1
fi
echo "Using dataset: $DATASET_FILE"

python -u training/td3bc_pretrain.py \
    --dataset "$DATASET_FILE" \
    --experiment-name "${EXPERIMENT_NAME}_offline" \
    --n-steps "$OFFLINE_STEPS" \
    --batch-size 256 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --alpha-ratio 2.5 \
    --hidden 256 128 \
    --save-every 50000 \
    --print-every 5000 \
    2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_phase2.log"

echo "[Phase 2] Offline pretraining complete!"

# ==============================================================================
# Phase 3: Online微调 (PPO)
# ==============================================================================
echo ""
echo "[Phase 3] Online fine-tuning with PPO..."
echo "=============================================================="

# td3bc_pretrain.py 输出到 checkpoints/{experiment_name}/best.pth
PRETRAIN_CKPT="checkpoints/${EXPERIMENT_NAME}_offline/best.pth"

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Warning: Pretrain checkpoint not found at $PRETRAIN_CKPT"
    echo "Skipping online fine-tuning..."
else
    python -u training/online_ppo_finetune.py \
        --low-level-ckpt "$LOW_CKPT" \
        --pretrain-ckpt "$PRETRAIN_CKPT" \
        --experiment-name "${EXPERIMENT_NAME}_online" \
        --hidden 256 128 \
        --total-timesteps "$ONLINE_STEPS" \
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
    2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_phase3.log"
fi

echo "[Phase 3] Online fine-tuning complete!"

# ==============================================================================
# Phase 4: 评估与可视化
# ==============================================================================
echo ""
echo "[Phase 4] Evaluation and visualization..."
echo "=============================================================="

FINAL_CKPT="checkpoints/${EXPERIMENT_NAME}_online/best.pth"

if [ -f "$FINAL_CKPT" ]; then
    python evaluation/eval_and_visualize.py \
        --checkpoint "$FINAL_CKPT" \
        --n-episodes "$EVAL_EPISODES" \
        --n-cases 5 \
        --fps 10 \
        --skip-frames 1 \
        --save-json \
        2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}_eval.log"
    
    echo "[Phase 4] Evaluation complete!"
else
    echo "[Phase 4] Warning: Checkpoint not found at $FINAL_CKPT"
fi

# ==============================================================================
# 总结
# ==============================================================================
echo ""
echo "=============================================================="
echo "Training Pipeline Complete!"
echo "=============================================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""
echo "Final model: ${FINAL_CKPT}"
echo "Visualizations: checkpoints/${EXPERIMENT_NAME}_online/eval_visuals/"
echo "=============================================================="
