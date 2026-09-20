#!/bin/bash
# reseed_restart.sh — seed OOM memory with stale evidence, restore pack 3, restart driver on hardened code.
# Running workers (G2 S3, Q08/Q20 preserve) are separate nohups and survive; driver skips them via pgrep.
ROOT=/root/AblationWriting
bash -n $ROOT/src/drive_full.sh || { echo "SYNTAX FAIL"; exit 1; }
echo "SYNTAX OK"
for l in $ROOT/logs/full_*.log; do
  if grep -qiE "out of memory|OutOfMemoryError|CUDA OOM|CUBLAS_STATUS_ALLOC_FAILED" "$l" 2>/dev/null; then
    sig="$(stat -c '%i %s' "$l" 2>/dev/null)"
    grep -qxF "$l|$sig" $ROOT/.oom_handled 2>/dev/null || echo "$l|$sig" >> $ROOT/.oom_handled
  fi
done
echo "seeded:"
cat $ROOT/.oom_handled
echo 3 > $ROOT/.pack_max_full
pkill -f "drive_full.sh" 2>/dev/null
sleep 3
touch $ROOT/PROGRESS.md
nohup bash $ROOT/src/drive_full.sh > $ROOT/logs/driver_full_nohup.log 2>&1 &
echo "FULLDRV $!"
sleep 150
tail -n 6 $ROOT/logs/driver_full.log
