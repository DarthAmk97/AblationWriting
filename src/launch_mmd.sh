#!/bin/bash
set -euo pipefail

ROOT=/root/AblationWriting
PIDFILE="$ROOT/.mmd_embed.pid"
LOG="$ROOT/logs/mmd_embed.log"

if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "MMD already running pid=$pid"
    exit 0
  fi
fi

if pgrep -af '/root/AblationWriting/src/(mmd_embed|mmd_score|control).*\.py' >/dev/null; then
  echo "Refusing to launch beside an existing MMD/control worker" >&2
  pgrep -af '/root/AblationWriting/src/(mmd_embed|mmd_score|control).*\.py' >&2
  exit 1
fi

available_kb=$(df --output=avail / | tail -1)
if (( available_kb < 20 * 1024 * 1024 )); then
  echo "Refusing MMD launch with less than 20 GiB free" >&2
  exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/.hf_cache"
cd "$ROOT"
nohup env HF_HOME="$ROOT/.hf_cache" /venv/main/bin/python "$ROOT/src/mmd_embed.py" \
  --models L1 O1 Q08 Q20 G2 \
  --splits val2 test \
  --batch-size 1 \
  --max-length 4096 \
  --device cuda >>"$LOG" 2>&1 &
pid=$!
echo "$pid" > "$PIDFILE"
sleep 2
kill -0 "$pid"
echo "MMD launched pid=$pid log=$LOG"
