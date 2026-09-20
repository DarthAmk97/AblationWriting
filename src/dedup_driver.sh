#!/bin/bash
ps -o pid,etime,args -e | grep drive_to_end | grep -v grep
echo ---KILL-NEWER-DUPLICATE---
# keep oldest (6649), kill any newer duplicates
for p in $(pgrep -f drive_to_end.sh); do
  if [ "$p" != "6649" ]; then
    echo "killing duplicate driver $p"
    kill "$p" 2>/dev/null
  fi
done
sleep 2
ps -o pid,etime,args -e | grep drive_to_end | grep -v grep
