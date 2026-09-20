#!/bin/bash
# drive_to_end.sh — VAST-side autonomous H2 driver. Runs to completion, never idle.
# Queue: remaining smokes LF12,Q20,S3,G2,G4 (O1/L1/Q08 done). Packs max 2 concurrent small jobs.
# Resume-safe: skips models with >=12 smoke runs in runs.jsonl. Logs to logs/driver.log, progress to PROGRESS.md.
# Scope: /root/AblationWriting ONLY. Usage: nohup bash /root/AblationWriting/src/drive_to_end.sh > /root/AblationWriting/logs/driver_nohup.log 2>&1 &
set -u
ROOT=/root/AblationWriting
LOG=$ROOT/logs/driver.log
PROG=$ROOT/PROGRESS.md
mkdir -p $ROOT/logs
exec >>$LOG 2>&1
echo "=== DRIVER START $(date -u) ==="

smoke_done() {
  # $1 = MID -> 0 if >=12 runs (grep -c always prints a number; no || echo)
  local mid=$1
  local n=0
  if [ -f $ROOT/runs.jsonl ]; then
    n=$(grep -c "\"model\": \"$mid\"" $ROOT/runs.jsonl 2>/dev/null)
    n=${n:-0}
    n=$(echo "$n" | tr -cd '0-9')
    n=${n:-0}
  fi
  [ "$n" -ge 12 ]
}

running() {
  # $1 = MID -> 0 if process alive
  pgrep -f "smoke_h2.py --models $1" >/dev/null 2>&1
}

running_count() {
  local n
  n=$(pgrep -fc "smoke_h2.py --models" 2>/dev/null)
  n=${n:-0}
  echo "$n" | tr -cd '0-9'
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -cd '0-9'
}

launch() {
  local mid=$1
  echo "--- LAUNCH $mid $(date -u) ---"
  echo "- [$mid] smoke launched $(date -u)" >> $PROG
  nohup bash -c "source /venv/main/bin/activate; HF_HOME=$ROOT/.hf_cache HF_HUB_CACHE=$ROOT/.hf_cache/hub python $ROOT/src/smoke_h2.py --models $mid --cal_n 512 --val_n 128 --max_new 256" > $ROOT/logs/smoke_${mid}.log 2>&1 &
  echo "PID $! for $mid"
  echo "$(date -u) LAUNCH $mid PID $!" >> $LOG
  sleep 20
}

# Ordered queue: small/text first (packable), big multimodal solo last
QUEUE="LF12 Q20 S3 G2 G4"
PACK_FILE=$ROOT/.pack_max
[ -f "$PACK_FILE" ] || echo 2 > "$PACK_FILE"
echo "QUEUE: $QUEUE"

oom_victims() {
  # print MIDs whose CURRENT log shows OOM (logs are removed on relaunch, so no stale triggers; no marker files)
  for m in LF12 Q20 S3 G2 G4 O1 L1 Q08; do
    if [ -f "$ROOT/logs/smoke_${m}.log" ] && grep -qiE "out of memory|OutOfMemoryError|CUDA OOM|CUBLAS_STATUS_ALLOC_FAILED" "$ROOT/logs/smoke_${m}.log" 2>/dev/null; then
      echo "$m"
    fi
  done
}

blocked_models() {
  cat "$ROOT/.blocked_models" 2>/dev/null | tr ' ' '\n' | sort -u | tr '\n' ' '
}

handle_oom() {
  local victims="$1"
  [ -z "$(echo "$victims" | tr -d ' ')" ] && return 1
  echo "$(date -u) OOM DETECTED in:$victims — learning: pack down, solo relaunch (max 2 tries, then park)"
  # kill all smoke workers to free VRAM
  pkill -f "smoke_h2.py --models" 2>/dev/null
  sleep 15
  # learn: force pack_max=1 (solo); pack 3 permanently forbidden
  echo 1 > "$PACK_FILE"
  echo "pack3-forbidden-after-oom $(date -u)" >> "$PACK_FILE"
  for m in $victims; do
    cnt=$(cat "$ROOT/.oom_count_${m}" 2>/dev/null | tr -cd '0-9')
    cnt=${cnt:-0}
    cnt=$((cnt + 1))
    echo "$cnt" > "$ROOT/.oom_count_${m}"
    if [ "$cnt" -gt 2 ]; then
      echo "$(date -u) $m OOMed $cnt times — PARKING as BLOCKED (needs smaller batch/quant, operator call)" >> $LOG
      echo "- [$m] BLOCKED $(date -u): OOM x$cnt even solo. Needs batch shrink or quant; operator call." >> $PROG
      echo "## $(date -u) $m BLOCKED after $cnt OOMs (even solo)" >> $ROOT/failures.md
      echo "$m" >> "$ROOT/.blocked_models"
      continue
    fi
    # clean partial duplicate runs for victim so relaunch is clean (backup first)
    if [ -f $ROOT/runs.jsonl ]; then
      cp $ROOT/runs.jsonl $ROOT/runs.jsonl.bak-oom-$(date +%s) 2>/dev/null
      grep -v "\"model\": \"$m\"" $ROOT/runs.jsonl > $ROOT/runs.jsonl.tmp 2>/dev/null && mv $ROOT/runs.jsonl.tmp $ROOT/runs.jsonl
    fi
    rm -f "$ROOT/logs/smoke_${m}.log"
    echo "$(date -u) RELAUNCH $m SOLO after OOM (try $cnt)" >> $LOG
    echo "- [$m] OOM $(date -u): packed run died, cleaned partials, relaunching SOLO (try $cnt). Pack forced to 1." >> $PROG
    launch "$m"
    sleep 30
  done
  return 0
}

while true; do
  # OOM learning first: if any worker OOMed, pack down + relaunch solo, skip normal scheduling this tick
  victims=$(oom_victims)
  if [ -n "$(echo "$victims" | tr -d ' ')" ]; then
    handle_oom "$victims"
    sleep 60
    continue
  fi
  PACK_MAX=$(head -n 1 "$PACK_FILE" 2>/dev/null | tr -cd '0-9')
  [ -z "$PACK_MAX" ] && PACK_MAX=2
  # check completion
  all_done=1
  for m in $QUEUE O1 L1 Q08; do
    if ! smoke_done $m; then
      case $m in O1|L1|Q08) ;; # already done, skip check noise
        *) all_done=0;;
      esac
    fi
  done
  # remaining queue (skip BLOCKED-parked models)
  blocked=" $(blocked_models) "
  remaining=""
  for m in $QUEUE; do
    case "$blocked" in *" $m "*) continue;; esac
    if ! smoke_done $m && ! running $m; then
      remaining="$remaining $m"
    fi
  done
  echo "$(date -u) TICK running=$(running_count) free_mb=$(gpu_free_mb) remaining:$remaining"
  if [ -z "$(echo $remaining | tr -d ' ')" ]; then
    nrun=$(running_count)
    if [ "$nrun" -eq 0 ]; then
      echo "=== SMOKE PANEL COMPLETE $(date -u) ==="
      echo "- SMOKE PANEL COMPLETE $(date -u): O1/L1/Q08/LF12/Q20/S3/G2/G4 all >=12 runs" >> $PROG
      echo "NEXT: full H2 VAL2 halving + preservation + MMD/JMQ/StoryScope (implement full_h2.py, then extend driver). Driver parking (no idle re-launch)." >> $PROG
      break
    else
      echo "$(date -u) waiting for $nrun running to finish..."
      sleep 120
      continue
    fi
  fi
  # packing: PACK_MAX concurrent (learned; 2 default, 1 after OOM; 3 only via manual override); big models solo
  nrun=$(running_count)
  free=$(gpu_free_mb)
  if [ "$nrun" -ge "$PACK_MAX" ]; then
    echo "$(date -u) $nrun running (max $PACK_MAX), waiting..."
    sleep 120
    continue
  fi
  # pick next: prefer small if a big is running and vice versa? simple FIFO with solo rule for G2/G4/S3
  next=$(echo $remaining | awk '{print $1}')
  if [ "$nrun" -ge 1 ]; then
    # if running includes big or next is big, wait for slot to free (solo rule)
    if pgrep -f "smoke_h2.py --models G[24]" >/dev/null 2>&1 || pgrep -f "smoke_h2.py --models S3" >/dev/null 2>&1; then
      echo "$(date -u) big job running solo, waiting..."
      sleep 120
      continue
    fi
    case $next in
      G2|G4|S3) echo "$(date -u) $next needs solo slot, waiting..."; sleep 120; continue;;
    esac
  fi
  # VRAM guard: need >=9000 MB free to launch
  if [ "${free:-0}" -lt 9000 ] && [ "$nrun" -ge 1 ]; then
    echo "$(date -u) low VRAM ${free}MB with $nrun running, waiting..."
    sleep 120
    continue
  fi
  launch $next
  sleep 60
done
echo "=== DRIVER EXIT $(date -u) ==="
