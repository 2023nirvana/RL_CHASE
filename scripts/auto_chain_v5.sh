#!/bin/bash
# V5 全流程自动链：等当前 offline (PID 文件) 完成 → online 500k → extend 2M
# 用法：nohup bash scripts/auto_chain_v5.sh > logs/v5/auto_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

mkdir -p logs/v5
CHAIN_LOG="logs/v5/auto_chain.log"

log() { echo "[$(date '+%F %T')] $*"; }

PIDS_FILE="logs/v5/offline/pids_offline.txt"
if [ ! -f "$PIDS_FILE" ]; then
    log "ERROR: $PIDS_FILE not found. offline 是否在跑？"
    exit 1
fi

log "============================================"
log " V5 Auto Chain: offline-wait → online → extend"
log "============================================"
log "当前 offline PIDs:"
cat "$PIDS_FILE"

# ─── 1) 等 offline 3 个进程全部结束 ───
log "Phase 1/3: 等离线预训练完成 ..."
while read -r name pid; do
    [ -z "$pid" ] && continue
    log "  waiting $name (PID=$pid) ..."
    # 轮询直到进程消失（kill -0 失败）
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    log "  $name (PID=$pid) 已结束"
done < "$PIDS_FILE"

log "离线预训练全部结束。检查 ckpt 是否就绪："
for f in checkpoints/td3bc_v5_2m/best.pth \
         checkpoints/iql_v5_2m/best.pth \
         checkpoints/bppo_v5_2m/best.pth; do
    if [ -f "$f" ]; then
        log "  OK  $f"
    else
        log "  MISSING $f  —— online 可能会失败，但仍尝试继续"
    fi
done

# ─── 2) online 500k × 6 并行 ───
log ""
log "Phase 2/3: 启动 online 微调 (500k × 6) ..."
bash scripts/run_online_v5_all.sh
ONLINE_RC=$?
log "online 阶段退出码: $ONLINE_RC"

if [ "$ONLINE_RC" -ne 0 ]; then
    log "online 阶段失败，终止链路"
    exit $ONLINE_RC
fi

# ─── 3) extend 2M × 6 并行 ───
log ""
log "Phase 3/3: 启动 extend 续训 (2M × 6) ..."
bash scripts/extend_v5_2m.sh
EXTEND_RC=$?
log "extend 阶段退出码: $EXTEND_RC"

log "============================================"
log " V5 Auto Chain 全部完成"
log "============================================"
