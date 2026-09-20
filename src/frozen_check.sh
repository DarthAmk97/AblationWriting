#!/bin/bash
for m in S3 Q08 Q20 G2 G4 O1 L1 LF12; do
  f=/root/AblationWriting/artifacts/geometry/H2/$m/frozen.json
  if [ -f "$f" ]; then
    echo "$m: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("frozen", "NONE"))' "$f")"
  else
    echo "$m: (no frozen file)"
  fi
done
