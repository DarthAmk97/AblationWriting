"""
backfill_metrics.py — CPU-only upgrade of runs.jsonl from saved generation parquets (no GPU).
For each s2/s3 parquet config: L2-1/2/3, JSD, top unigram contributors, mean lengths (base/abl/human).
Patches matching runs rows in place (backup runs.jsonl.bak-metrics-*). ALSO the length-confound audit.
Usage: python src/backfill_metrics.py [MID...] (default all with parquets)
"""
import json
from pathlib import Path
from collections import defaultdict
ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))

TOK = None
def get_tok():
    global TOK
    if TOK is None:
        from transformers import AutoTokenizer as AT
        ht = (ROOT / ".hf_token").read_text().strip() if (ROOT / ".hf_token").exists() else None
        TOK = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=ht)
    return TOK

def main(mids):
    from h2_lib import load_splits, l2_ngram, jsd_1gram
    tok = get_tok()
    # ensure VAL2 floors exist for all four headline metrics (S3 wrote L2-1 only at first)
    splits = load_splits(["val1", "val2"])
    vf = ROOT / "artifacts" / "metrics" / "val2_floor.json"
    try:
        _v = json.load(open(vf)) if vf.exists() else {}
    except Exception:
        _v = {}
    if not all(k in _v for k in ["L2_1gram_hh", "L2_2gram_hh", "L2_3gram_hh", "JSD_hh"]):
        vh = [r["human_text"] for r in splits["val2"]]
        h1, h2 = vh[:256], vh[256:]
        d1, _ = l2_ngram(h1, h2, tok, n=1)
        d2, _ = l2_ngram(h1, h2, tok, n=2)
        d3, _ = l2_ngram(h1, h2, tok, n=3)
        dj = jsd_1gram(h1, h2, tok)
        _v.update({"L2_1gram_hh": float(d1), "L2_2gram_hh": float(d2), "L2_3gram_hh": float(d3),
                   "JSD_hh": float(dj), "n": len(vh)})
        json.dump(_v, open(vf, "w"), indent=2)
        print("VAL2 floors:", {k: round(v, 6) for k, v in _v.items() if k != "n"}, flush=True)
    href = {}
    for name in ["val1", "val2"]:
        for r in splits[name]:
            href[str(r["prompt_id"])] = r["human_text"]
    import pandas as pd
    runs = [json.loads(l) for l in open(ROOT / "runs.jsonl") if l.strip()]
    by_id = {r["run_id"]: r for r in runs}
    n_patch = 0
    for pq in sorted((ROOT / "artifacts" / "geometry").glob("H2/*/s*_generations.parquet")):
        mid = pq.parent.name
        if mids and mid not in mids:
            continue
        stage = "s2" if "_s2_" in pq.name or pq.name.startswith("s2_") else "s3"
        df = pd.read_parquet(pq)
        print(f"{mid} {pq.name}: {len(df)} rows, {df['config'].nunique()} configs", flush=True)
        for cfg, g in df.groupby("config"):
            # find matching run(s): same model+stage, config string contained in run_id
            cands = [r for r in runs if r.get("model") == mid and r.get("stage") == stage and cfg in r.get("run_id", "")]
            if not cands:
                print(f"  {cfg}: NO matching run (skipped)", flush=True)
                continue
            abl = g["ablated"].tolist()
            base = g["baseline"].tolist()
            hum = [href.get(str(i), "") for i in g["prompt_id"].tolist()]
            hum = [h for h in hum if h]
            l2a, contrib = l2_ngram(abl, hum, tok, n=1)
            l2a2, _ = l2_ngram(abl, hum, tok, n=2)
            l2a3, _ = l2_ngram(abl, hum, tok, n=3)
            jsda = jsd_1gram(abl, hum, tok)
            l2b, _ = l2_ngram(base, hum, tok, n=1)
            l2b2, _ = l2_ngram(base, hum, tok, n=2)
            l2b3, _ = l2_ngram(base, hum, tok, n=3)
            jsdb = jsd_1gram(base, hum, tok)
            def mlen(ts):
                return float(sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in ts) / max(len(ts), 1))
            top = [tok.decode([k[0][0]]) if isinstance(k[0], tuple) else str(k[0]) for k in contrib[:10]]
            for r in cands:
                r["L2_2_base"] = l2b2
                r["L2_2_abl"] = l2a2
                r["L2_3_base"] = l2b3
                r["L2_3_abl"] = l2a3
                r["JSD_base"] = jsdb
                r["JSD_abl"] = jsda
                r["top_words"] = top
                r["len_base"] = mlen(base)
                r["len_abl"] = mlen(abl)
                r["len_human"] = mlen(hum)
                n_patch += 1
            print(f"  {cfg}: L2 {l2b:.5f}->{l2a:.5f} L2-2 {l2a2:.5f} JSD {jsda:.4f} len b/a/h "
                  f"{mlen(base):.0f}/{mlen(abl):.0f}/{mlen(hum):.0f} top={top[:5]}", flush=True)
    import time
    open(ROOT / f"runs.jsonl.bak-metrics-{int(time.time())}", "w").write("\n".join(json.dumps(r) for r in runs) + "\n")
    open(ROOT / "runs.jsonl", "w").write("\n".join(json.dumps(r) for r in runs) + "\n")
    print(f"PATCHED {n_patch} rows (backups kept)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
