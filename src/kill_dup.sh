#!/bin/bash
# kill_dup.sh — kill duplicate L1 S3 (64342, 28min, still in baseline, nothing logged yet); keep original 64114
kill 64342 2>/dev/null
sleep 5
ps -o pid,etime,args -e | grep full_h2 | grep -v grep | head -n 6
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv | head -n 3
