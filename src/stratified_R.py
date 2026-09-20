"""stratified_R.py — CPU-only. Recompute L2 recovery excluding refusal-borderline baselines.
Answers: how much of R is refusal-removal vs style? Uses saved parquets + splits."""
import re, json
from pathlib import Path
import pandas as pd
ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))
REF = re.compile(r"(i'?m sorry|i cannot|i can'?t|cannot comply|unable to|not able to|against policy|helpline|trusted adult|mental health professional|substance abuse)", re.I)
splits = {}
for name in ["val1", "val2"]:
    try:
        for line in open(ROOT / "data" / "splits" / f"{name}.jsonl"):
            r = json.loads(line)
            splits[str(r["prompt_id"])] = r["human_text"]
    except Exception:
        pass
from h2_lib import l2_ngram
import sys as _s
_sys_tok = None
def tok():
    global _sys_tok
    if _sys_tok is None:
        from transformers import AutoTokenizer as AT
        ht = (ROOT / ".hf_token").read_text().strip() if (ROOT / ".hf_token").exists() else None
        _sys_tok = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=ht)
    return _sys_tok
t = tok()
for pq in sorted((ROOT / "artifacts" / "geometry").glob("H2/*/s*_generations.parquet")):
    df = pd.read_parquet(pq, columns=["prompt_id", "baseline", "ablated", "config"])
    # focus: best-config rows only to keep it fast (largest effect = most at stake)
    runs = [json.loads(l) for l in open(ROOT / "runs.jsonl") if l.strip()]
    mid = pq.parent.name
    stage = "s2" if "s2_" in pq.name else "s3"
    cands = [r for r in runs if r.get("model") == mid and r.get("stage") == stage]
    if not cands:
        continue
    best = max(cands, key=lambda x: x.get("R_L2", -9))
    cfg = best["run_id"].split(mid + "-", 1)[-1] if mid + "-" in best["run_id"] else None
    # match parquet config string: stored as window-r..-a..-rho..[-form]
    g = df[df["config"].str.contains(f"r{best['rank']}-a{best['alpha']}")].copy() if "rank" in best else df
    if len(g) == 0:
        g = df
    hum = [splits.get(str(i), "") for i in g["prompt_id"]]
    base, abl = g["baseline"].tolist(), g["ablated"].tolist()
    keep = [not bool(REF.search(b[:600])) for b in base]
    def R(b, a, h):
        db, _ = l2_ngram(b, h, t, n=1)
        da, _ = l2_ngram(a, h, t, n=1)
        h1, h2 = h[:len(h) // 2], h[len(h) // 2:]
        dh, _ = l2_ngram(h1, h2, t, n=1)
        return (db - da) / max(db - dh, 1e-12), db, da, dh
    Rall, dba, daa, dha = R(base, abl, hum)
    bc = [x for x, k in zip(base, keep) if k]
    ac = [x for x, k in zip(abl, keep) if k]
    hc = [x for x, k in zip(hum, keep) if k]
    Rcl, dbc, dac, dhc = R(bc, ac, hc)
    print(f"{mid} {best['run_id']}: n={len(base)} refused={len(base)-len(bc)} "
          f"R_all={Rall:+.3f} R_clean={Rcl:+.3f} (base {dba:.5f}/{dbc:.5f})", flush=True)
