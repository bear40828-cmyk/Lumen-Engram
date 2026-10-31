#!/bin/bash
# 夜间聚类归纳（任务四·步骤3）。只是脚本，crontab 由助手装（建议每天 4:30）：
#   30 4 * * * /srv/engram-live/maintenance/nightly-cluster.sh
# 模型：.env 里必须配 CLUSTER_MODEL（Flash 档）；没配 cluster_summarize 会拒绝（否则会继承 EXTRACT_MODEL=Pro）。
# 每晚调用上限缺省 20（ENGRAM_CLUSTER_MAX_CALLS 可改，稳了再放到 40）。
# 开关 ENGRAM_CLUSTER_SUMMARY=1（.env 或环境）没开＝什么都不做。numpy 只装在 venv：
#   python3 -m venv --system-site-packages .venv && .venv/bin/pip install numpy
# （系统没装 python3-venv 时：python3 -m venv --system-site-packages --without-pip .venv
#   && python3 -m pip --python .venv/bin/python install numpy ；--system-site-packages 是为了沿用系统的 yaml 等包）
set -u
_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIR=${ENGRAM_DIR:-$_HERE}
PY=${ENGRAM_CLUSTER_PY:-$DIR/.venv/bin/python}
LOG=$DIR/cluster-cron.log
exec 9>"$DIR/.ingest.lock" || exit 0
flock -w 600 9 || exit 0          # 跟 0:00 全量 / 每 30 分钟切片共用一把锁，等不到就让开
cd "$DIR" || exit 0
[ -x "$PY" ] || { echo "$(date '+%F %T') 没有 venv（$PY），跳过" >> "$LOG"; exit 0; }
{
  echo "=== $(TZ=Asia/Shanghai date '+%F %T') 聚类归纳 ==="
  timeout 300 "$PY" vectors.py build 2>&1 | tail -1          # 先补齐新事实的向量
  timeout 1800 "$PY" maintenance/cluster_summarize.py --apply --max-calls "${ENGRAM_CLUSTER_MAX_CALLS:-20}" 2>&1 | tail -3
} >> "$LOG" 2>&1
