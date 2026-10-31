#!/bin/bash
# 每天把 DeepSeek 写的会话摘要灌进 engram 库。
# 由 cron 在北京时间 00:00 触发（crontab 里用 CRON_TZ=Asia/Shanghai 声明，不受夏令时影响）。
# 那一刻「昨天」刚封档、是完整的一天，「今天」通常还是空的或只有零星几条。
# 仍然两天都灌，是为了兜住 digest.mjs 延迟写入的情况；ingest.py 自带去重，重复灌是安全的。
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
flock -n 9 || exit 0

cd "$DIR" || exit 0

# 任务三：ENGRAM_DISTILL_SLICE=1 时夜里只做「补漏（扫最近 3 天的会话文件）、重试失败的、审计（列死信箱）」，
# 不再重抽整天、不再灌每日摘要。默认关＝下面旧流程一字不差。
if python3 -c 'import flags,sys; sys.exit(0 if flags.on("DISTILL_SLICE") else 1)' >/dev/null 2>&1; then
  {
    echo "=== $(TZ=Asia/Shanghai date '+%F %T') 夜间：补漏/重试/审计 ==="
    timeout 900 python3 -m extract.collect --since-days 3 2>&1 | tail -2
    timeout 1800 python3 -m extract.distill_slice --apply 2>&1 | tail -3
    timeout 300 python3 -m extract.distill_slice --audit 2>&1 | tail -20
    timeout 300 python3 -m derive_repair --apply 2>&1 | tail -2
    timeout 300 python3 -m dup_report 2>&1 | tail -5
    timeout 900 python3 vectors.py build 2>&1 | tail -1
  } >> "$LOG" 2>&1
  tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  exit 0
fi

# 按北京时间取日期——摘要文件是按她那边的日子命名的，服务器是太平洋时区。
for d in $(TZ=Asia/Shanghai date -d yesterday +%F) $(TZ=Asia/Shanghai date +%F); do
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') 灌 $d ==="
  timeout 900 python3 ingest.py "$d" 2>&1 | grep -E "晋升|合计|库：|错误|Error"
done >> "$LOG" 2>&1

# 原话 → 事：抽北京时间「昨天」整天。按 UTC 算就是前天 16:00 到昨天 16:00。
# 只读最近两天动过的会话文件；promote 按内容去重，重跑安全。
SINCE=$(date -u -d "$(TZ=Asia/Shanghai date -d yesterday +%F) 00:00 +0800" +%FT%H:%M)
UNTIL=$(date -u -d "$(TZ=Asia/Shanghai date +%F) 00:00 +0800" +%FT%H:%M)
FILES=$(find "$TRANSCRIPTS" -maxdepth 1 -name '*.jsonl' -mtime -2)
if [ -n "$FILES" ]; then
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') 抽事 $SINCE → $UNTIL ===" >> "$LOG"
  # shellcheck disable=SC2086
  timeout 1800 python3 -m extract.from_evidence --no-timeline --jsonl $FILES --since "$SINCE" --until "$UNTIL" \
    2>&1 | grep -E "留下|失败" >> "$LOG"
fi

# 日志别无限长
tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"

# 新抽出来的事补向量（按意思找用，见 vectors.py）。只补缺的，失败不影响别的。
echo "=== $(TZ=Asia/Shanghai date '+%F %T') 补向量 ===" >> "$LOG"
timeout 900 python3 vectors.py build 2>&1 | tail -1 >> "$LOG"
