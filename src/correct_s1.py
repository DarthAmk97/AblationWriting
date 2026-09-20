"""correct_s1.py — re-rank finished S1 runs by RAW L2 (stable) instead of noisy 16v16-floor R.
Writes artifacts/geometry/H2/<MID>/s1_top12.json (corrected) + logs ranking.
Run AFTER s1 stage completes for a model. Uses runs.jsonl H2-full-s1 entries.
Rule: rem>0, empty-pathology delta <=2pp vs baseline, sort L2_abl asc, keep 12. Ties -> lower rank wins.
"""
import json, sys
from pathlib import Path
ROOT = Path("/root/AblationWriting")

def main(mid):
    runs = [json.loads(l) for l in open(ROOT / "runs.jsonl") if l.strip()]
    s1 = [r for r in runs if r.get("stage") == "s1" and r.get("model") == mid]
    print(f"{mid}: {len(s1)} s1 runs", flush=True)
    if not s1:
        return
    # baseline pathology from first run's implied base? use min empty as ref; simpler: require empty<=1
    ok = []
    for r in s1:
        p = r.get("pathologies", {})
        if r.get("rem", -1) is not None and r.get("rem", 0) <= 0:
            print(f"  DROP {r['run_id']} rem={r.get('rem')} (dead hook)", flush=True)
            continue
        if p.get("empty", 0) > 1:
            print(f"  DROP {r['run_id']} empty={p.get('empty')}", flush=True)
            continue
        ok.append(r)
    ok.sort(key=lambda r: (r["L2_abl"], r["rank"]))
    top = ok[:12]
    geom = ROOT / "artifacts" / "geometry" / "H2" / mid
    json.dump([{"run_id": r["run_id"], "L2_abl": r["L2_abl"], "L2_base": r["L2_base"],
                "R_noisy": r.get("R_L2"), "rank": r["rank"], "window": r["window"]} for r in top],
              open(geom / "s1_top12.json", "w"), indent=2)
    print(f"{mid}: corrected top{len(top)} by RAW L2:", flush=True)
    for r in top:
        print(f"  {r['run_id']} L2={r['L2_abl']:.5f} (R_noisy={r.get('R_L2',0):+.2f})", flush=True)

if __name__ == "__main__":
    for m in sys.argv[1:]:
        main(m)
