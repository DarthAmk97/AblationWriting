#!/bin/bash
# launch_controls_pack.sh — CONTAINED EXPERIMENT: run O1+Q08 control generation concurrently.
# Same v4 epoch, same code, same seeds as serial; separate OS processes have
# independent RNG streams, so outputs are identical to serial runs. Separate
# per-model logs. Any failure (incl. OOM) kills the survivor and exits
# nonzero so the loop falls back to the serial launcher, which resumes from
# atomic 16-prompt checkpoints. Never use for Q20/G2/embedder (solo class).
set -euo pipefail

ROOT=/root/AblationWriting
PYTHON=/venv/main/bin/python
PIDFILE="$ROOT/.controls.pid"
LOCKFILE="$ROOT/.controls.lock"
LOG="$ROOT/logs/controls_pack.log"
PACK_CACHE="$ROOT/.controls_pack_cache"
MODELS=(O1 Q08)
O1_REPO="allenai/OLMo-2-0425-1B-Instruct"
O1_REV="48d788eca847d4d7548f375ad03d3c9312f6139e"
Q08_REPO="Qwen/Qwen3.5-0.8B"
Q08_REV="2fc06364715b967f1860aea9cf38778875588b17"
WORKER_RE='/root/AblationWriting/src/(full_h2|preserve_full|test_headline|mmd_embed|mmd_score|h2_controls|control_embed|control_stats)\.py'

if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE" | head -n 1)
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Refusing overlap: control pipeline already running pid=$pid" >&2
    exit 1
  fi
fi

if pgrep -af "$WORKER_RE" >/dev/null; then
  echo "Refusing overlap with an active core, MMD, or control worker:" >&2
  pgrep -af "$WORKER_RE" >&2
  exit 1
fi

if ! command -v flock >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "flock and nvidia-smi are required" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$ROOT/.hf_token" ]]; then
  echo "Missing Hugging Face token: $ROOT/.hf_token" >&2
  exit 1
fi
HF_TOKEN=$(tr -d '\r\n' < "$ROOT/.hf_token")
if [[ -z "$HF_TOKEN" ]]; then
  echo "Hugging Face token is empty" >&2
  exit 1
fi
export HF_TOKEN

gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$gpu_pids" ]]; then
  echo "Refusing overlap with existing GPU compute processes: $gpu_pids" >&2
  exit 1
fi

# VRAM gate: two small models need ~9GB + headroom.
free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if (( free_vram < 16384 )); then
  echo "Refusing packed launch with less than 16 GiB free VRAM (have ${free_vram} MiB)" >&2
  exit 1
fi

available_kb=$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')
if (( available_kb < 6 * 1024 * 1024 )); then
  echo "Refusing packed launch with less than 6 GiB free disk" >&2
  exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/.hf_cache" "$PACK_CACHE"

# Offline archive/plan preflight for both models (no model load).
HF_HOME="$ROOT/.hf_cache" HF_HUB_CACHE="$ROOT/.hf_cache/hub" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 H2_CONTROLS_ALLOW_NETWORK=0 \
  "$PYTHON" "$ROOT/src/h2_controls.py" --root "$ROOT" --models "${MODELS[@]}" --preflight

# Serial prefetch of both pinned snapshots into the shared pack cache,
# then compute runs fully offline (load_target uses local_files_only=True).
HF_HOME="$PACK_CACHE" HF_HUB_CACHE="$PACK_CACHE/hub" \
  "$PYTHON" - "$O1_REPO" "$O1_REV" "$Q08_REPO" "$Q08_REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
for repo, rev in [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])]:
    snapshot_download(repo_id=repo, revision=rev, max_workers=4)
    print(f"prefetched {repo}@{rev[:12]}", flush=True)
EOF

nohup bash -c '
  set -euo pipefail
  ROOT=$1
  PYTHON=$2
  PIDFILE=$3
  LOCKFILE=$4
  PACK_CACHE=$5
  shift 5
  exec 9>"$LOCKFILE"
  flock -n 9 || { echo "Refusing overlap: control lock is held" >&2; exit 1; }
  OVERLAP_RE="/root/AblationWriting/src/(full_h2|preserve_full|test_headline|mmd_embed|mmd_score|h2_controls|control_embed|control_stats)\\.py"
  overlap=$(pgrep -af "$OVERLAP_RE" | grep -v "^$$ " || true)
  if [[ -n "$overlap" ]]; then
    echo "Refusing overlap detected after preflight:" >&2
    printf "%s\n" "$overlap" >&2
    exit 1
  fi
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed "/^[[:space:]]*$/d" || true)
  if [[ -n "$gpu_pids" ]]; then
    echo "Refusing overlap with GPU compute processes detected after preflight: $gpu_pids" >&2
    exit 1
  fi
  echo $$ > "$PIDFILE"
  children=""
  cleanup() {
    for c in $children; do kill -0 "$c" 2>/dev/null && kill "$c" 2>/dev/null || true; done
    if [[ "$(head -n 1 "$PIDFILE" 2>/dev/null || true)" == "$$" ]]; then
      rm -f "$PIDFILE"
    fi
  }
  trap cleanup EXIT INT TERM
  export CUDA_VISIBLE_DEVICES=0
  export HF_HOME="$PACK_CACHE"
  export HF_HUB_CACHE="$PACK_CACHE/hub"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=0
  export H2_CONTROLS_ALLOW_NETWORK=0
  # Eval tokenizer is Qwen/Qwen3.5-0.8B = Q08 repo/rev, already in the pack cache.
  export H2_CONTROLS_EVAL_CACHE="$PACK_CACHE"
  for model in "$@"; do
    "$PYTHON" "$ROOT/src/h2_controls.py" --root "$ROOT" --models "$model" \
      >>"$ROOT/logs/controls_pack_${model}.log" 2>&1 &
    children="$children $!"
    echo "$! pid_for=$model" >> "$PIDFILE"
  done
  failed=0
  for _ in "$@"; do
    if ! wait -n; then
      failed=1
      for c in $children; do kill -0 "$c" 2>/dev/null && kill "$c" 2>/dev/null || true; done
    fi
  done
  wait || true
  if (( failed )); then
    echo "PACKED_RUN_FAILED: inspect logs/controls_pack_O1.log logs/controls_pack_Q08.log; fall back to serial launcher (resumes from checkpoints)" >&2
    exit 1
  fi
  echo "PACKED_RUN_COMPLETE O1 Q08"
' controls-pack-worker "$ROOT" "$PYTHON" "$PIDFILE" "$LOCKFILE" "$PACK_CACHE" "${MODELS[@]}" >>"$LOG" 2>&1 &

pid=$!
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Packed worker exited during startup; inspect $LOG" >&2
  exit 1
fi
echo "Packed control pipeline launched pid=$pid gpu=0 models=${MODELS[*]} log=$LOG"
