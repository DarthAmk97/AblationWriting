"""fix_windows.py — recompute CONTIGUOUS screen0 windows from saved screen0.json (no GPU)."""
import json, math
from pathlib import Path
ROOT = Path("/root/AblationWriting")
for mid in ["O1", "L1"]:
    geom = ROOT / "artifacts" / "geometry" / "H2" / mid
    s0 = json.load(open(geom / "screen0.json"))
    out = {int(k): v for k, v in s0["layers"].items()}
    L = s0["L"]
    S = 4 if L <= 16 else (5 if L <= 24 else 6)
    ls = sorted(out.keys())
    cands = [ls[i:i + S] for i in range(len(ls) - S + 1)]
    scored = []
    for w in cands:
        conds = [out[li]["cond"] for li in w]
        if max(conds) > 1e7:
            continue
        score = sum(out[li]["eff_rank"] for li in w) / len(w) + sum(math.log10(max(c, 1)) for c in conds) / len(w)
        scored.append((score, w))
    scored.sort(key=lambda x: x[0])
    picked = []
    for _, w in scored:
        if all(set(w).isdisjoint(set(p)) for p in picked):
            picked.append(w)
        if len(picked) == 2:
            break
    print(mid, "L=", L, "picked:", picked, flush=True)
    json.dump({"W-A": picked[0], "W-B": picked[1], "ranks": [2, 4, 8, 16, 32]},
              open(geom / "screen0_windows.json", "w"), indent=2)
print("FIXED", flush=True)
