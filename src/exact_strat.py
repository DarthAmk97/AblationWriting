"""exact_strat.py — exact-config stratified R (no substring pooling): frozen rows only, all vs clean."""
import re, json
from pathlib import Path
import pandas as pd
ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))
REF = re.compile(r"(i'?m sorry|i cannot|i can'?t|cannot comply|unable to|not able to|against policy|helpline|trusted adult|mental health professional|substance abuse)", re.I)
splits = {}
for name in ["val1", "val2"]:
    for line in open(ROOT / "data" / "splits" / f"{name}.jsonl"):
        r = json.loads(line)
        splits[str(r["prompt_id"])] = r["human_text"]
from h2_lib import l2_ngram
from transformers import AutoTokenizer as AT
ht = (ROOT / ".hf_token").read_text().strip() if (ROOT / ".hf_token").exists() else None
t = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=ht)
TARGETS = {
    "L1": ("s3_generations.parquet", "W-BEST-r4-a0.75-rho0.5-cone"),
    "O1": ("s3_generations.parquet", "W-BEST-r2-a0.75-rho0.5-cone"),
    "S3": ("s3_generations.parquet", "W-B-r32-a1.0-rho0.5-cone"),
    "Q08": ("s3_generations.parquet", "W-A-r8-a1.0-rho0.5-cone"),
}
for mid, (pqname, cfg) in TARGETS.items():
    p = ROOT / "artifacts" / "geometry" / "H2" / mid / pqname
    if not p.exists():
        print(f"{mid}: {pqname} MISSING (stage not done)", flush=True)
        continue
    df = pd.read_parquet(p)
    g = df[df["config"] == cfg]
    if len(g) == 0:
        print(f"{mid}: config {cfg} not in parquet ({df['config'].nunique()} others)", flush=True)
        continue
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
    print(f"{mid} {cfg}: n={len(base)} refused={len(base)-len(bc)} R_all={Rall:+.3f} R_clean={Rcl:+.3f}", flush=True)
