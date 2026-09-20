#!/bin/bash
# launch_driver.sh — start drive_to_end (idempotent)
chmod +x /root/AblationWriting/src/drive_to_end.sh
if pgrep -f drive_to_end.sh >/dev/null 2>&1; then
  echo "DRIVER ALREADY RUNNING"
  ps -o pid,etime,args -e | grep drive_to_end | grep -v grep | head -n 3
else
  touch /root/AblationWriting/PROGRESS.md
  nohup bash /root/AblationWriting/src/drive_to_end.sh > /root/AblationWriting/logs/driver_nohup.log 2>&1 &
  echo "DRIVER PID $!"
  sleep 15
  tail -n 20 /root/AblationWriting/logs/driver.log
fi
