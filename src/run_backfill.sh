#!/bin/bash
# run_backfill.sh — CPU-only backfill, niced, correct log path
source /venv/main/bin/activate
export HF_HOME=/root/AblationWriting/.hf_cache
exec nice -n 15 python /root/AblationWriting/src/backfill_metrics.py
