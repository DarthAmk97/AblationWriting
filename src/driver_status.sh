#!/bin/bash
# driver_status.sh — read-only status for driver (no quoting hell)
date -u
echo ===PS===
ps -o pid,etime,time,args -e | grep -E '(full_h2|preserve_full|test_headline|h2_controls|mmd_embed|mmd_score|stage_dump)\.py|drive_full\.sh|ticker_loop\.sh' | grep -v grep | head -n 30
echo ===GPU===
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv | head -n 3
echo ===LOGS===
ls -lht /root/AblationWriting/logs/ | head -n 20
echo ===RUNS===
if [[ -f /root/AblationWriting/src/runs_count.py && -f /root/AblationWriting/runs.jsonl ]]; then
  python3 /root/AblationWriting/src/runs_count.py
else
  echo "minimal reconstruction; core H2 archived locally/HF"
  find /root/AblationWriting/artifacts/geometry/H2 -name 'mmd_embeddings_*.parquet' -type f 2>/dev/null | wc -l | xargs echo "MMD embeddings:"
  find /root/AblationWriting/artifacts/geometry/H2 -name 'mmd_metrics.json' -type f 2>/dev/null | wc -l | xargs echo "MMD scored models:"
fi
echo ===DISK===
df -h / | head -n 3
du -sh /root/AblationWriting/.hf_cache
