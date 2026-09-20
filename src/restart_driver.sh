#!/bin/bash
pkill -f drive_to_end.sh 2>/dev/null
sleep 3
touch /root/AblationWriting/PROGRESS.md
nohup bash /root/AblationWriting/src/drive_to_end.sh > /root/AblationWriting/logs/driver_nohup.log 2>&1 &
echo "DRIVER PID $!"
sleep 20
tail -n 12 /root/AblationWriting/logs/driver.log
