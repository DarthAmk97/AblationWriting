#!/bin/bash
# revert_s3_freeze.sh — S3 VAL2 best is R-0.054 (S2 +0.38 did not survive VAL2): freeze rule now requires R>0.
# PRE-TEST legal (TEST unopened). Keeps eligible list for the record.
python3 - <<'PYEOF'
import json
p = "/root/AblationWriting/artifacts/geometry/H2/S3/frozen.json"
d = json.load(open(p))
print("was:", d.get("frozen"))
d["frozen"] = None
d["rule"] = "eligible(cone,mini+path gates,R>0 on VAL2) max R; tie lower rank,alpha"
d["reverted"] = "2026-09-10: best VAL2 R-0.054 (S2 +0.38 selection-bias shrinkage); no R>0 eligible -> weak verdict, no TEST"
d["eligible"] = []
json.dump(d, open(p, "w"), indent=2)
print("now: NONE-eligible (weak)")
PYEOF
cat >> /root/AblationWriting/failures.md << 'EOF'

## 2026-09-10 — S3 VAL2 contradicts S2 (+0.38 -> -0.05): selection-bias shrinkage, freeze REVERTED (PRE-TEST)
- S2 top (96 VAL1 prompts, the selection set) does not survive VAL2-512: exact frozen-row recompute R_all -0.054, R_clean -0.055.
- This is the funnel working as designed (VAL2 exists to kill selection winners), not a bug. Freeze rule amended R>0 requirement (TEST unopened: PRE-TEST legal).
- S3 joins LiquidAI as weak/no-freeze. Refusal stratification on same rows: clean==all, contamination ruled out everywhere.
EOF
echo DONE
