"""numbers.py — full runs ledger table (CPU-only). Best R per model/stage + frozen + gates."""
import json
from pathlib import Path
from collections import defaultdict
ROOT = Path("/root/AblationWriting")
runs = [json.loads(l) for l in open(ROOT / "runs.jsonl") if l.strip()]
print(f"TOTAL RUNS: {len(runs)}", flush=True)
by = defaultdict(list)
for r in runs:
    by[(r.get("model"), r.get("stage", r.get("hypothesis")))].append(r)
for (mid, st) in sorted(by.keys(), key=str):
    rs = by[(mid, st)]
    b = max(rs, key=lambda x: x.get("R_L2", -9))
    npos = sum(1 for x in rs if x.get("R_L2", 0) > 0)
    gates = sum(1 for x in rs if x.get("gate_ok"))
    print(f"{mid:5s} {str(st):8s} n={len(rs):3d} best={b['run_id']} R={b.get('R_L2', 0):+.3f} "
          f"L2 {b.get('L2_base', 0):.5f}->{b.get('L2_abl', 0):.5f} npos={npos} gated={gates}", flush=True)
print("--- frozen ---", flush=True)
for f in sorted((ROOT / "artifacts" / "geometry").glob("H2/*/frozen.json")):
    try:
        j = json.load(open(f))
        print(f"{f.parent.name}: {j.get('frozen')} eligible={len(j.get('eligible', []))}", flush=True)
    except Exception as e:
        print(f"{f.parent.name}: ERR {e}", flush=True)
