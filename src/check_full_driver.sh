#!/bin/bash
# check_full_driver.sh — syntax-check drive_full, dry-run its first tick logic read-only, then start it ONLY if safe
bash -n /root/AblationWriting/src/drive_full.sh || { echo "SYNTAX FAIL"; exit 1; }
echo "SYNTAX OK"
echo "--- would-launch preview (no launches) ---"
ROOT=/root/AblationWriting
for item in Q20:screen0 Q20:s1 Q20:s2 Q20:s3 G2:screen0 G2:s1 G2:s2 G2:s3 G4:screen0 G4:s1 G4:s2 G4:s3 S3:s3 O1:s3 Q08:s3; do
  m=${item%%:*}; s=${item##*:}
  echo "$item running=$(pgrep -f "full_h2.py --models $m --stage $s" >/dev/null 2>&1 && echo Y || echo n)"
done
echo "--- running full_h2 procs ---"
ps -o pid,etime,args -e | grep full_h2 | grep -v grep | head -n 6
