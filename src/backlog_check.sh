#!/bin/bash
bash /root/AblationWriting/src/frozen_check.sh
for m in L1 O1 S3 Q08 Q20 G2 LF12; do
  f=/root/AblationWriting/artifacts/geometry/H2/$m/preservation_full.json
  if [ -f "$f" ]; then
    echo "$m preserve DONE"
  else
    echo "$m preserve --"
  fi
done
