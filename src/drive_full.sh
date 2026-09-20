#!/bin/bash
# drive_full.sh — VAST-side FULL-H2 state machine. Enforces the ORDERING PRINCIPLE in code, not memory:
#   GPU-first chain on VAST (geometry->sweeps->freeze->preservation->TEST->MMD-embed->controls-gen);
#   LOCAL chain fed by dump (JMQ/stats/figures/paper); dump pushed on every stage transition; figure-data incremental.
# Completion markers per (model,stage); never relaunches finished/done stages; never touches TEST before frozen.json.
# OOM learning inherited: pack down on OOM, solo relaunch (max 2 tries), park chronic OOMers as BLOCKED.
# Usage: nohup bash src/drive_full.sh > logs/driver_full_nohup.log 2>&1 &
set -u
ROOT=/root/AblationWriting
LOG=$ROOT/logs/driver_full.log
PROG=$ROOT/PROGRESS.md
PACK_FILE=$ROOT/.pack_max_full
[ -f "$PACK_FILE" ] || echo 3 > "$PACK_FILE"
mkdir -p $ROOT/logs
exec >>$LOG 2>&1
echo "=== FULL DRIVER START $(date -u) ==="

# QUEUE in dependency order. preserve/test scripts now exist (preserve_full.py, test_headline.py).
QUEUE="Q20:screen0 Q20:s1 Q20:s2 Q20:s3 G2:screen0 G2:s1 G2:s2 G2:s3 S3:s3 O1:s3 Q08:s3 L1:preserve S3:preserve O1:preserve Q08:preserve Q20:preserve G2:preserve L1:test S3:test O1:test Q08:test Q20:test G2:test"
echo "QUEUE: $QUEUE"

marker_done() { # $1 MID $2 STAGE -> 0 if complete
  local m=$1 s=$2
  case $s in
    screen0) [ -f "$ROOT/artifacts/geometry/H2/$m/screen0_windows.json" ];;
    s1) [ -f "$ROOT/artifacts/geometry/H2/$m/s1_top12.json" ];;
    s2) [ -f "$ROOT/artifacts/geometry/H2/$m/s2_top4.json" ];;
    s3) [ -f "$ROOT/artifacts/geometry/H2/$m/frozen.json" ];;
    preserve) [ -f "$ROOT/artifacts/geometry/H2/$m/preservation_full.json" ];;
    test) [ -f "$ROOT/artifacts/geometry/H2/$m/test_done" ];;
  esac
}
running() { # $1 MID $2 STAGE (stage maps to its script)
  case $2 in
    preserve) pgrep -f "preserve_full.py --models $1" >/dev/null 2>&1;;
    test) pgrep -f "test_headline.py --models $1" >/dev/null 2>&1;;
    *) pgrep -f "full_h2.py --models $1 --stage $2" >/dev/null 2>&1;;
  esac
}
running_any() { [ "$(nrun)" -gt 0 ]; }
nrun() { pgrep -fc '(full_h2|preserve_full|test_headline)\.py --models' 2>/dev/null | tr -cd '0-9'; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -cd '0-9'; }
blocked() { grep -qx "$1" "$ROOT/.blocked_models" 2>/dev/null; }

launch() { # $1 MID $2 STAGE (dispatches to the stage's script)
  local cmd
  case $2 in
    preserve) cmd="python $ROOT/src/preserve_full.py --models $1";;
    test) cmd="python $ROOT/src/test_headline.py --models $1";;
    *) cmd="python $ROOT/src/full_h2.py --models $1 --stage $2";;
  esac
  echo "--- LAUNCH $1 $2 $(date -u) ---"
  echo "- [$1:$2] launched $(date -u)" >> $PROG
  # shellcheck disable=SC2086
  # APPEND logs (never overwrite — overwrites destroyed two preserve crash tracebacks, lesson 2026-09-10)
  echo "===== LAUNCH $1 $2 $(date -u) =====" >> $ROOT/logs/full_${1}_${2}.log
  nohup bash -c "source /venv/main/bin/activate; HF_HOME=$ROOT/.hf_cache HF_HUB_CACHE=$ROOT/.hf_cache/hub $cmd" >> $ROOT/logs/full_${1}_${2}.log 2>&1 &
  echo "$(date -u) LAUNCH $1 $2 PID $!" >> $LOG
  sleep 20
}
push_dump() { # fire-and-forget content push (proves ordering: dump follows GPU, feeds local)
  nohup bash -c "source /venv/main/bin/activate; HF_HOME=$ROOT/.hf_cache python $ROOT/src/stage_dump.py" > $ROOT/logs/dump_auto.log 2>&1 &
  echo "$(date -u) DUMP push triggered" >> $LOG
}
need() { # $1 MID $2 STAGE -> 0 if launchable now (deps met, not done/running/blocked)
  local m=$1 s=$2
  blocked "$m" && return 1
  marker_done "$m" "$s" && return 1
  running "$m" "$s" && return 1
  case $s in
    s1) marker_done "$m" screen0 || return 1;;
    s2) marker_done "$m" s1 || return 1;;
    s3) marker_done "$m" s2 || return 1;;
    preserve) marker_done "$m" s3 || return 1;
      # frozen must be a real config (reverted NONE-weak models like S3/LF12 have the file but null content)
      grep -q '"frozen": "H2' "$ROOT/artifacts/geometry/H2/$m/frozen.json" 2>/dev/null || return 1;;
    test) marker_done "$m" s3 || return 1;
      grep -q '"frozen": "H2' "$ROOT/artifacts/geometry/H2/$m/frozen.json" 2>/dev/null || return 1;
      # preservation NOT a gate for TEST (2026-09-11 user: safety overlap is insight for H3, not a block)
      [ -f "$ROOT/artifacts/geometry/H2/$m/preservation_full.json" ] || return 1;;
  esac
  return 0
}

last_push_sig=""
# Ignore historical OOMs at startup; a new OOM changes the last matching line number/content.
OOM_RE="out of memory|OutOfMemoryError|CUDA OOM|CUBLAS_STATUS_ALLOC_FAILED"
for l in $ROOT/logs/full_*.log; do
  match=$(grep -inE "$OOM_RE" "$l" 2>/dev/null | tail -n 1)
  [ -n "$match" ] && echo "$l|$(printf '%s' "$match" | md5sum | cut -d' ' -f1)" >> "$ROOT/.oom_handled"
done
while true; do
  echo "$(date -u) TICK running=$(nrun) free_mb=$(free_mb)"
  # OOM scan (IDEMPOTENT: only NEW evidence since last handled signature; stale text must never refire —
  # lesson 2026-09-11: refire loop killed fresh G2-S3 relaunches every 2 min)
  oom=""
  for l in $ROOT/logs/full_*.log; do
    match=$(grep -inE "$OOM_RE" "$l" 2>/dev/null | tail -n 1)
    if [ -n "$match" ]; then
      sig="$(printf '%s' "$match" | md5sum | cut -d' ' -f1)"
      if ! grep -qxF "$l|$sig" "$ROOT/.oom_handled" 2>/dev/null; then
        oom="$oom $l"
        echo "$l|$sig" >> "$ROOT/.oom_handled"
      fi
    fi
  done
  # (old non-idempotent scan removed here — see above; stale text refired every loop)
  if [ -n "$(echo "$oom" | tr -d ' ')" ]; then
    echo "$(date -u) OOM in:$oom pack->1, selective kill (victims only — lesson: pkill-all burned innocent G2-S3 twice)" >> $LOG
    echo 1 > $PACK_FILE
    for l in $oom; do
      base=$(basename "$l" .log)  # full_<MID>_<stage> | smoke_<MID>
      mid=$(echo "$base" | sed -E 's/^(full|smoke)_([A-Za-z0-9]+)_.*/\2/')
      if [ -n "$mid" ]; then
        pkill -f "full_h2.py --models $mid" 2>/dev/null
        pkill -f "smoke_h2.py --models $mid" 2>/dev/null
        pkill -f "preserve_full.py --models $mid" 2>/dev/null
        pkill -f "test_headline.py --models $mid" 2>/dev/null
      fi
    done
    sleep 20
  fi
  PACK_MAX=$(head -n 1 "$PACK_FILE" 2>/dev/null | tr -cd '0-9')
  [ -z "$PACK_MAX" ] && PACK_MAX=3
  # dump push: frozen-set change OR 6h elapsed (not every run-line — avoids commit spam)
  froz_now=$(ls $ROOT/artifacts/geometry/H2/*/frozen.json 2>/dev/null | sort | md5sum | cut -d' ' -f1)
  last_push=$(cat $ROOT/.last_auto_push 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ "$froz_now" != "$(cat $ROOT/.last_froz 2>/dev/null)" ] || [ $((now - last_push)) -gt 21600 ]; then
    echo "$froz_now" > $ROOT/.last_froz
    echo "$now" > $ROOT/.last_auto_push
    push_dump
  fi
  # find launchable (dependency order = queue order)
  launched=""
  for item in $QUEUE; do
    m=${item%%:*}; s=${item##*:}
    if need "$m" "$s"; then
      n=$(nrun); [ -z "$n" ] && n=0
      free=$(free_mb); [ -z "$free" ] && free=0
      if [ "$n" -ge "$PACK_MAX" ]; then break; fi
      # heavy guards: s3/preserve/test need >=9000 free when something runs; screen0/s1/s2 need >=6000
      case $s in
        s3|preserve|test) [ "$n" -ge 1 ] && [ "$free" -lt 9000 ] && continue;;
        *) [ "$n" -ge 1 ] && [ "$free" -lt 6000 ] && continue;;
      esac
      # big-model solo for HEAVY stages only (screen0/s1 teacher-force + light gens pack freely;
      # EXCEPT Gemma-4-E4B (8B, ~16GB weights): solo at EVERY stage — lesson 2026-09-11, OOMed screen0 packed)
      # 2026-09-11 relax per user: G2:s3 packs with lights (heavy-only = G4:* + S3:s3 + Q20:s3); G2 now 2-way.
      case "$m:$s" in
        S3:s3|S3:preserve|G4:*|Q20:s3|Q20:preserve) [ "$n" -ge 1 ] && continue;;
      esac
      launch "$m" "$s"
      launched="$launched $item"
      n=$(nrun)
      [ "$n" -ge "$PACK_MAX" ] && break
    fi
  done
  # completion: queue drained (done or blocked) and nothing running
  pending=0
  for item in $QUEUE; do
    m=${item%%:*}; s=${item##*:}
    if need "$m" "$s"; then pending=$((pending + 1)); fi
  done
  if [ "$pending" -eq 0 ] && ! running_any; then
    echo "=== FULL QUEUE DRAINED $(date -u) (done/blocked/parked) ===" >> $PROG
    echo "=== FULL DRIVER EXIT $(date -u) ==="
    break
  fi
  sleep 120
done
