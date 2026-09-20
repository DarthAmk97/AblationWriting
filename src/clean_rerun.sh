#!/bin/bash
# clean_rerun.sh — remove invalid S1 W-BEST rows (O1 dead-hook x5, L1 partial-hook x5), backup first
cd /root/AblationWriting
cp runs.jsonl runs.jsonl.bak-wbest-bug-$(date +%s)
grep -v 'H2-full-s1-O1-W-BEST' runs.jsonl | grep -v 'H2-full-s1-L1-W-BEST' > runs.jsonl.tmp && mv runs.jsonl.tmp runs.jsonl
echo "kept lines: $(wc -l < runs.jsonl)"
grep -c W-BEST runs.jsonl || echo "0 W-BEST rows remain"
