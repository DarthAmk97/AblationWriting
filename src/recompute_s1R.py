"""recompute_s1R.py — rewrite R_L2 for H2-full-s1 rows with frozen VAL1 floor (old rows carry noisy 16v16 R)."""
import json
from pathlib import Path
ROOT = Path("/root/AblationWriting")
floor = json.load(open(ROOT / "artifacts" / "metrics" / "val1_floor.json"))
F = float(floor["L2_1gram_hh"])
lines = [l for l in open(ROOT / "runs.jsonl") if l.strip()]
fixed, n = [], 0
for l in lines:
    r = json.loads(l)
    if r.get("stage") == "s1" and "L2_base" in r and "L2_abl" in r:
        denom = r["L2_base"] - F
        r["R_L2"] = float((r["L2_base"] - r["L2_abl"]) / denom) if abs(denom) > 1e-12 else 0.0
        r["L2_hh"] = F
        r["floor"] = "VAL1-256-frozen"
        n += 1
    fixed.append(r)
open(ROOT / "runs.jsonl", "w").write("\n".join(json.dumps(r) for r in fixed) + "\n")
print(f"patched {n} s1 rows to frozen floor {F}", flush=True)
