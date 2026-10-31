#!/bin/bash
# 近实时写入：每小时把最近 75 分钟的聊天抽成记忆，不用等到半夜。
# 框架第四节「说完就进候选」那条断路的最小补法：还是同一个抽取器（DeepSeek），只是跑得勤。
# 窗口比间隔多 15 分钟，跨边界的话不会漏；抽取按内容去重，重叠部分不会记两遍。
# 不判新旧顶替（--no-timeline，9/23 的决定）；新旧冲突交给召回时的 newer.py。
set -u
_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=${ENGRAM_DIR:-$_HERE}   # 缺省＝脚本所在目录；ENGRAM_DIR 可覆盖（测试用）
# 会话 jsonl 目录：环境变量 ENGRAM_TRANSCRIPT_DIR > persona.env 里的同名项 > ~/.claude/projects
_PE=${ENGRAM_PERSONA_FILE:-$DIR/persona.env}
if [ -z "${ENGRAM_TRANSCRIPT_DIR:-}" ] && [ -f "$_PE" ]; then
  ENGRAM_TRANSCRIPT_DIR=$(sed -n "s/^ENGRAM_TRANSCRIPT_DIR=//p" "$_PE" | head -1 | tr -d "'\"")
fi
TRANSCRIPTS=${ENGRAM_TRANSCRIPT_DIR:-$HOME/.claude/projects}
LOCK=$DIR/.ingest.lock
LOG=$DIR/ingest-cron.log

exec 9>"$LOCK" || exit 0
flock -n 9 || exit 0          # 半夜那次全量在跑就让开
cd "$DIR" || exit 0

# 任务三：ENGRAM_DISTILL_SLICE=1 时走「采集原话 → 会话安静 ≥3 小时才整理 → 补派生层」新流程（建议 cron 改成每 30 分钟一趟）。
# 默认关＝下面旧的 75 分钟窗口一字不差。读开关用 flags.py（环境变量 > .env），读不出来按关处理。
if python3 -c 'import flags,sys; sys.exit(0 if flags.on("DISTILL_SLICE") else 1)' >/dev/null 2>&1; then
  {
    echo "=== $(TZ=Asia/Shanghai date '+%F %T') 切片（采集→整理→补派生） ==="
    timeout 600 python3 -m extract.collect 2>&1 | tail -2
    timeout 1500 python3 -m extract.distill_slice --apply 2>&1 | tail -3
    timeout 300 python3 -m derive_repair --apply 2>&1 | tail -2
    timeout 300 python3 vectors.py build 2>&1 | tail -1
  } >> "$LOG" 2>&1
  exit 0
fi

SINCE=$(date -u -d '75 minutes ago' +%FT%H:%M)
UNTIL=$(date -u +%FT%H:%M)
FILES=$(find "$TRANSCRIPTS" -maxdepth 1 -name '*.jsonl' -mmin -80)
[ -n "$FILES" ] || exit 0
{
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') 每小时抽 $SINCE → $UNTIL ==="
  # shellcheck disable=SC2086
  timeout 1200 python3 -m extract.from_evidence --no-timeline --jsonl $FILES --since "$SINCE" --until "$UNTIL" 2>&1 | grep -E "留下|失败"
  timeout 300 python3 vectors.py build 2>&1 | tail -1
} >> "$LOG" 2>&1
