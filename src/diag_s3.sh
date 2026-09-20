#!/bin/bash
tail -n 60 /root/AblationWriting/logs/full_L1_s3.log
echo ===CORRECT-S3-LF12===
source /venv/main/bin/activate
HF_HOME=/root/AblationWriting/.hf_cache python /root/AblationWriting/src/correct_s1.py S3 LF12 2>&1 | head -n 30
