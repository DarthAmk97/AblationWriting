#!/bin/bash
# restore_pack3.sh — first OOM analyzed: G4-screen0 (~16GB weights) packed with G2-S3 -> OOM, correct pack-down fired.
# Steady state fix: G4 solo at EVERY stage (rule patched + uploaded); restore PACK_MAX=3; restart driver (workers untouched).
echo 3 > /root/AblationWriting/.pack_max_full
bash -n /root/AblationWriting/src/drive_full.sh || { echo "SYNTAX FAIL"; exit 1; }
echo "SYNTAX OK"
pkill -f "drive_full.sh" 2>/dev/null
sleep 3
touch /root/AblationWriting/PROGRESS.md
nohup bash /root/AblationWriting/src/drive_full.sh > /root/AblationWriting/logs/driver_full_nohup.log 2>&1 &
echo "FULLDRV $!"
sleep 150
tail -n 8 /root/AblationWriting/logs/driver_full.log
