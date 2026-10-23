#!/bin/bash
# 近实时写入：每小时把最近 75 分钟的聊天抽成记忆，不用等到半夜。
# 框架第四节「说完就进候选」那条断路的最小补法：还是同一个抽取器（DeepSeek），只是跑得勤。
# 窗口比间隔多 15 分钟，跨边界的话不会漏；抽取按内容去重，重叠部分不会记两遍。
# 不判新旧顶替（--no-timeline，9/23 的决定）；新旧冲突交给召回时的 newer.py。
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK=$DIR/.ingest.lock
LOG=$DIR/ingest-cron.log

exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0          # 半夜那次全量在跑就让开
cd "$DIR" || exit 0

SINCE=$(date -u -d '75 minutes ago' +%FT%H:%M)
UNTIL=$(date -u +%FT%H:%M)
FILES=$(find "${ENGRAM_TRANSCRIPT_DIR:-$HOME/.claude/projects}" -maxdepth 1 -name '*.jsonl' -mmin -80)
[ -n "$FILES" ] || exit 0
{
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') 每小时抽 $SINCE → $UNTIL ==="
  # shellcheck disable=SC2086
  timeout 1200 python3 -m extract.from_evidence --no-timeline --jsonl $FILES --since "$SINCE" --until "$UNTIL" 2>&1 | grep -E "留下|失败"
  timeout 300 python3 vectors.py build 2>&1 | tail -1
} >> "$LOG" 2>&1
