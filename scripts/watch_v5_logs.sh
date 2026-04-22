#!/bin/bash
# 一键查看 v5 6 个 online 实验的实时进度
# 用法:
#   bash scripts/watch_v5_logs.sh            # 显示各 log 最后 20 行
#   bash scripts/watch_v5_logs.sh follow     # 实时 tail -f 所有 log
#   bash scripts/watch_v5_logs.sh offline    # 查看离线预训练 log
#   bash scripts/watch_v5_logs.sh status     # 只看运行状态 (PID 存活 / 最新一行)

cd "$(dirname "$0")/.."
ONLINE_DIR="logs/v5/online"
OFFLINE_DIR="logs/v5/offline"
MODE="${1:-summary}"

case "$MODE" in
  follow)
    exec tail -n 50 -F "$ONLINE_DIR"/*.log
    ;;
  offline)
    exec tail -n 50 -F "$OFFLINE_DIR"/*.log
    ;;
  status)
    PIDFILE="$ONLINE_DIR/pids.txt"
    [[ -f "$PIDFILE" ]] || { echo "$PIDFILE 不存在"; exit 1; }
    printf "%-10s %-8s %-7s %s\n" NAME PID STATE LAST_LINE
    while read name pid; do
      if kill -0 "$pid" 2>/dev/null; then st="RUN"; else st="DEAD"; fi
      last=$(tail -n 1 "$ONLINE_DIR/${name}.log" 2>/dev/null | cut -c1-80)
      printf "%-10s %-8s %-7s %s\n" "$name" "$pid" "$st" "$last"
    done < "$PIDFILE"
    ;;
  *)
    for f in "$ONLINE_DIR"/*.log; do
      [[ -e "$f" ]] || continue
      echo "========== $f =========="
      tail -n 20 "$f"
      echo
    done
    ;;
esac
#!/bin/bash
# 实时查看 v5 online 6 个实验的日志滚动输出
# 用法:
#   bash scripts/watch_v5_logs.sh           # online 全部 6 个
#   bash scripts/watch_v5_logs.sh offline   # offline 3 个
set -e
cd "$(dirname "$0")/.."

MODE="${1:-online}"
DIR="logs/v5/$MODE"

if [[ ! -d "$DIR" ]]; then
    echo "日志目录不存在: $DIR"
    echo "请先运行对应的训练脚本。"
    exit 1
fi

FILES=( "$DIR"/*.log )
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "$DIR 下暂无 .log 文件"
    exit 1
fi

echo "===== tail -F 以下文件 (Ctrl-C 退出) ====="
for f in "${FILES[@]}"; do echo "  $f"; done
echo ""
tail -F "${FILES[@]}"
