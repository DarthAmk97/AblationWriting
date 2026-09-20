#!/bin/bash
bash -n /root/AblationWriting/src/drive_full.sh || { echo "SYNTAX FAIL"; exit 1; }
echo "SYNTAX OK"
pkill -f "drive_full.sh" 2>/dev/null
sleep 3
touch /root/AblationWriting/PROGRESS.md
nohup bash /root/AblationWriting/src/drive_full.sh > /root/AblationWriting/logs/driver_full_nohup.log 2>&1 &
echo "FULLDRV $!"
sleep 150
tail -n 12 /root/AblationWriting/logs/driver_full.log
