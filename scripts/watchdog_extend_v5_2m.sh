#!/bin/bash
# V5 Online 续训到 2M — 从 latest.pth 续，自守护：被杀后等 60s 自动重启
# 用法:  nohup bash scripts/watchdog_extend_v5_2m.sh > logs/v5/watchdog.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/v5

LOW="low_near/v5_direct_thrust/checkpoints/best.pth"
TARGET_STEPS=2000000
MAX_RETRIES=30
SLEEP_BETWEEN=60

log() { echo "[$(date '+%F %T')] $*"; }

COMMON="--env-version v5 \
  --v-max-pursuer 0.65 --a-max-pursuer 0.5 \
  --v-max-evader 0.85  --a-max-evader 0.25 \
  --obs-dim 21 --act-dim 2 --hidden 256 128 \
  --total-timesteps $TARGET_STEPS --n-steps 2048 --num-envs 4 \
  --gamma 0.99 --gae-lambda 0.95 --max-grad-norm 0.5 \
  --normalize-reward --r-catch 200.0 --r-fail -50.0 \
  --alpha-shaping 5.0 --d-sg-safe 1.5 \
  --eval-freq 20000 --eval-episodes 20 --save-freq 50000 \
  --world-size 30.0 --catch-radius 1.0 --subgoal-range 0.7 \
  --low-steps 50 --max-steps 6000 --evader-mode medium"

GA="--lr 1e-5 --lr-end 5e-7 --clip-range 0.1 --n-epochs 3 \
  --beta-kl 0.2 --kl-anneal-frac 0.5 --ent-coef 0.01 \
  --c-sg 3.0 --c-path 2.0 --lam-time 0.02"

GB="--lr 3e-5 --lr-end 1e-6 --clip-range 0.15 --n-epochs 4 \
  --beta-kl 0.0 --kl-anneal-frac 0.0 --ent-coef 0.02 \
  --c-sg 1.0 --c-path 1.0 --lam-time 0.01"

# (exp_name, group)
EXPS=(
    "v5_bppo_A:A"
    "v5_bppo_B:B"
    "v5_iql_A:A"
    "v5_iql_B:B"
    "v5_td3bc_A:A"
    "v5_td3bc_B:B"
)

# 获取当前 step 数（从 latest.pth 读 total_steps）
get_step() {
    local exp=$1
    python - <<EOF 2>/dev/null
import torch
try:
    c = torch.load("checkpoints/$exp/latest.pth", map_location="cpu", weights_only=False)
    print(c.get("total_steps", 0))
except Exception:
    print(0)
EOF
}

launch_one() {
    local exp=$1
    local group=$2
    local short=${exp#v5_}   # e.g. bppo_A
    local logf="logs/v5/online/${short}.log"
    local args_grp
    [ "$group" = "A" ] && args_grp="$GA" || args_grp="$GB"

    python -u training/online_ppo_finetune.py \
        --low-level-ckpt "$LOW" \
        --resume "checkpoints/$exp/latest.pth" \
        --experiment-name "$exp" --seed 42 \
        $COMMON $args_grp >> "$logf" 2>&1 &
    echo $!
}

log "============================================"
log " V5 Watchdog: resume→2M with auto-restart"
log "============================================"

for attempt in $(seq 1 $MAX_RETRIES); do
    log ""
    log "=== Attempt $attempt / $MAX_RETRIES ==="

    # 查每个实验还差多少
    all_done=true
    declare -A PIDS
    for entry in "${EXPS[@]}"; do
        exp=${entry%:*}; grp=${entry#*:}
        cur=$(get_step "$exp")
        log "  $exp: step=$cur / $TARGET_STEPS"
        if [ "$cur" -lt "$TARGET_STEPS" ]; then
            all_done=false
            pid=$(launch_one "$exp" "$grp")
            PIDS[$exp]=$pid
            log "    -> launched PID=$pid"
        else
            log "    -> already done, skip"
        fi
    done

    if $all_done; then
        log ""
        log "所有实验都已到 ${TARGET_STEPS} 步，退出 watchdog"
        exit 0
    fi

    # 等当前这批都结束（正常完成 or 被杀）
    log "等待 ${#PIDS[@]} 个进程..."
    for exp in "${!PIDS[@]}"; do
        pid=${PIDS[$exp]}
        while kill -0 "$pid" 2>/dev/null; do
            sleep 30
        done
        log "  $exp (PID=$pid) 结束"
    done

    log "本轮结束，sleep $SLEEP_BETWEEN s 后再检查..."
    sleep $SLEEP_BETWEEN
    unset PIDS
    declare -A PIDS
done

log "达到最大重试次数 $MAX_RETRIES，退出"
