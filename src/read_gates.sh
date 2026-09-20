#!/bin/bash
for m in O1 Q08; do
  echo "=== $m ==="
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("pass:", d.get("pass"), "gates:", json.dumps(d.get("gates", {}))); print("kl:", d.get("kl_mean"), "safety:", json.dumps(d.get("safety", {})))' /root/AblationWriting/artifacts/geometry/H2/$m/preservation_full.json
done
