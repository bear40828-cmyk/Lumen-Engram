#!/bin/bash
# 日志轮转：hook.log、llm-route.log 超过 2MB 就整份挪到 archive/logs/ 并压缩（不删、不截断）。
# 为什么对钩子安全：钩子和 llm.py 每次写日志都是「按名字 open(..., 'a')、写完就关」，写失败也是静默的；
# 所以这里用「改名」而不是「truncate」——改名之后下一次写会自己新建同名文件，不会丢行，也不需要动钩子代码。
# 用法（给 cron）：maintenance/rotate_logs.sh        环境变量：ENGRAM_DIR（默认＝本脚本上一级目录）、ENGRAM_LOG_MAX_BYTES（默认 2097152）
set -u
DIR=${ENGRAM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MAX=${ENGRAM_LOG_MAX_BYTES:-2097152}
DEST=$DIR/archive/logs
mkdir -p "$DEST" || exit 0
for f in hook.log llm-route.log; do
  p=$DIR/$f
  [ -f "$p" ] || continue
  sz=$(stat -c %s "$p" 2>/dev/null || echo 0)
  [ "$sz" -gt "$MAX" ] || continue
  ts=$(date +%Y%m%d-%H%M%S)
  out=$DEST/$f.$ts
  mv "$p" "$out" || continue
  [ -e "$p" ] || touch "$p"          # 读日志的人别撞上「文件不在」；touch 不会截断刚被新写入的内容
  sleep 1                            # 让正在写旧文件的那一两行落完
  gzip -9 "$out" && chmod 640 "$out.gz"
  echo "$(date '+%F %T') 轮转 $f（$sz 字节）→ archive/logs/$f.$ts.gz"
done
exit 0
