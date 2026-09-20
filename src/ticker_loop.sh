#!/bin/bash
# 30-min stable-run ticker — appends to /root/AblationWriting/logs/ticker.log
# Run via nohup in background. Also checks stall.
while true; do
  source /venv/main/bin/activate
  HF_HOME=/root/AblationWriting/.hf_cache python /root/AblationWriting/src/watchdog_remote.py >> /root/AblationWriting/logs/ticker.log 2>&1
  echo "--- sleep 1800 ---" >> /root/AblationWriting/logs/ticker.log
  sleep 1800
done
