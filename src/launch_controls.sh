#!/bin/bash
set -euo pipefail

ROOT=/root/AblationWriting
PYTHON=/venv/main/bin/python
PIDFILE="$ROOT/.controls.pid"
LOCKFILE="$ROOT/.controls.lock"
LOG="$ROOT/logs/controls.log"
TARGET_CACHE="$ROOT/.controls_target_cache"
EVAL_CACHE="$ROOT/.controls_eval_cache"
MODELS=(L1 O1 Q08 Q20 G2)
WORKER_RE='/root/AblationWriting/src/(full_h2|preserve_full|test_headline|mmd_embed|mmd_score|h2_controls|control_embed|control_stats)\.py'

if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE")
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

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$gpu_count" -lt 1 ]]; then
  echo "No NVIDIA GPU is available" >&2
  exit 1
fi
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$gpu_pids" ]]; then
  echo "Refusing overlap with existing GPU compute processes: $gpu_pids" >&2
  exit 1
fi

available_kb=$(df --output=avail "$ROOT" | tail -1 | tr -d ' ')
if (( available_kb < 8 * 1024 * 1024 )); then
  echo "Refusing control launch with less than 8 GiB free" >&2
  exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/.hf_cache" "$EVAL_CACHE"

# Prior-epoch -> v5 migration: archive blocked prior-epoch state safely before v5.
# Prior suites (H2-MRSC-400 through H2-MRSC-400-v4) share paths with v5.
# v4 progress/anchors files are ADOPTED by the v5 runner (text hashes verified,
# epoch bookkeeping rewritten); only v4 plans, markers, and policies are archived.
# Refuse only on FINAL artifacts, which must never exist without a matching done marker.
for model in "${MODELS[@]}"; do
  plan="$ROOT/artifacts/geometry/H2/$model/controls_plan.json"
  if [[ -f "$plan" ]]; then
    if ! grep -q '"suite": "H2-MRSC-400-v5"' "$plan"; then
      geom="$ROOT/artifacts/geometry/H2/$model"
      if [[ -e "$geom/controls_generations.parquet" || -e "$geom/control_generation_done.json" || -e "$geom/control_embeddings.parquet" || -e "$geom/control_embed_done.json" || -e "$geom/control_metrics.json" || -e "$geom/controls_done.json" ]]; then
        echo "Refusing migration for $model: FINAL prior-epoch artifacts exist without a matching done marker path; manual review required" >&2
        exit 1
      fi
      stamp=$(date -u +%Y%m%dT%H%M%SZ)
      prior=$(grep -o '"suite": "[^"]*"' "$plan" | head -n 1)
      mv -- "$plan" "$geom/controls_plan.blocked-$stamp.json"
      echo "Archived blocked prior-epoch plan for $model ($prior)"
      for transient in "$geom/controls_reconstruction_preflight.json" "$geom/controls_calibration.progress.json" "$geom/controls_calibration.json" "$geom/controls_lexical_policy.json"; do
        if [[ -f "$transient" ]]; then
          mv -- "$transient" "$transient.blocked-$stamp"
          echo "Archived blocked transient $(basename "$transient") for $model"
        fi
      done
      echo "Kept for v5 adoption: controls_generations.progress.parquet, controls_val1_anchors.parquet (if present)"
    fi
  fi
done

# Archive and plan validation runs before nohup and before any model is loaded.
HF_HOME="$ROOT/.hf_cache" HF_HUB_CACHE="$ROOT/.hf_cache/hub" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 H2_CONTROLS_ALLOW_NETWORK=0 \
  "$PYTHON" "$ROOT/src/h2_controls.py" --root "$ROOT" --models "${MODELS[@]}" --preflight

nohup bash -c '
  set -euo pipefail
  ROOT=$1
  PYTHON=$2
  PIDFILE=$3
  LOCKFILE=$4
  shift 4
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
  TARGET_CACHE="$ROOT/.controls_target_cache"
  EVAL_CACHE="$ROOT/.controls_eval_cache"
  require_space() {
    local available_kb
    available_kb=$(df --output=avail "$ROOT" | tail -1 | tr -d " ")
    if (( available_kb < 8 * 1024 * 1024 )); then
      echo "Refusing control stage with less than 8 GiB free" >&2
      exit 1
    fi
  }
  clear_target_cache() {
    if [[ "$TARGET_CACHE" != "$ROOT/.controls_target_cache" || "$TARGET_CACHE" == "/" || -L "$TARGET_CACHE" ]]; then
      echo "Unsafe target cache path: $TARGET_CACHE" >&2
      exit 1
    fi
    rm -rf -- "$TARGET_CACHE"
    mkdir -p "$TARGET_CACHE"
  }
  remove_target_cache() {
    if [[ "$TARGET_CACHE" == "$ROOT/.controls_target_cache" && "$TARGET_CACHE" != "/" && ! -L "$TARGET_CACHE" ]]; then
      rm -rf -- "$TARGET_CACHE"
    fi
  }
  echo $$ > "$PIDFILE"
  cleanup() {
    remove_target_cache
    if [[ "$(cat "$PIDFILE" 2>/dev/null || true)" == "$$" ]]; then
      rm -f "$PIDFILE"
    fi
  }
  trap cleanup EXIT INT TERM
  export CUDA_VISIBLE_DEVICES=0
  export H2_CONTROLS_EVAL_CACHE="$EVAL_CACHE"
  for model in "$@"; do
    require_space
    clear_target_cache
    export HF_HOME="$TARGET_CACHE"
    export HF_HUB_CACHE="$TARGET_CACHE/hub"
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
    export H2_CONTROLS_ALLOW_NETWORK=1
    "$PYTHON" "$ROOT/src/h2_controls.py" --root "$ROOT" --models "$model"
    remove_target_cache
  done
  require_space
  export HF_HOME="$ROOT/.hf_cache"
  export HF_HUB_CACHE="$ROOT/.hf_cache/hub"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export H2_CONTROLS_ALLOW_NETWORK=0
  "$PYTHON" "$ROOT/src/control_embed.py" --root "$ROOT" --models "$@"
  require_space
  export HF_HOME="$EVAL_CACHE"
  export HF_HUB_CACHE="$EVAL_CACHE/hub"
  export HF_HUB_OFFLINE=0
  export TRANSFORMERS_OFFLINE=0
  export H2_CONTROLS_ALLOW_NETWORK=1
  "$PYTHON" "$ROOT/src/control_stats.py" --root "$ROOT"
' controls-worker "$ROOT" "$PYTHON" "$PIDFILE" "$LOCKFILE" "${MODELS[@]}" >>"$LOG" 2>&1 &

pid=$!
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Control worker exited during startup; inspect $LOG" >&2
  exit 1
fi
echo "Control pipeline launched pid=$pid gpu=0 log=$LOG"
