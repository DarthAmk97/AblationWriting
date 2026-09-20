#!/bin/bash
# fix_shadow.sh — remove stdlib-shadowing numbers.py (broke numpy: module 'numbers' has no attribute 'Integral')
rm -f /root/AblationWriting/src/numbers.py
rm -rf /root/AblationWriting/src/__pycache__
ls /root/AblationWriting/src/ | grep -iE '^(numbers|json|sys|re|code|token|inspect|typing|collections|pathlib|random|hashlib|time|os)\.py' || echo "no shadows remain"
source /venv/main/bin/activate
export HF_HOME=/root/AblationWriting/.hf_cache
nohup bash -c 'source /venv/main/bin/activate; HF_HOME=/root/AblationWriting/.hf_cache nice -n 15 python /root/AblationWriting/src/backfill_metrics.py' > /root/AblationWriting/logs/backfill.log 2>&1 &
echo BACKFILL:$!
