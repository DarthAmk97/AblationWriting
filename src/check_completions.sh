#!/bin/bash
for f in /root/AblationWriting/artifacts/geometry/H2/Q20/frozen.json /root/AblationWriting/artifacts/geometry/H2/Q08/preservation_full.json /root/AblationWriting/artifacts/geometry/H2/G2/s2_top4.json; do
  echo "=== $f ==="
  ls -l --time-style=+%H:%M "$f" 2>&1
done
echo "=== Q20 frozen content ==="
cat /root/AblationWriting/artifacts/geometry/H2/Q20/frozen.json 2>&1 | head -n 8
echo "=== Q08 preserve tail ==="
tail -n 4 /root/AblationWriting/logs/full_Q08_preserve.log 2>&1 | head -n 6
echo "=== Q20 S3 tail ==="
tail -n 3 /root/AblationWriting/logs/full_Q20_s3.log 2>&1 | head -n 5
