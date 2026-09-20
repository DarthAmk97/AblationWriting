"""
test_headline.py — TEST headline generation after frozen.json + preservation_full exist.
Per model: baseline + frozen-cone on TEST 2000 (cap 1024, seed-matched) -> test_generations.parquet,
headline L2-1/2/3 + JSD + lengths, run rows hypothesis H2-test. Also builds judging/inputs/test_jmq_pairs.parquet:
  - overall 400 TEST-JMQ: (H2 vs human) + (baseline vs human) pairs
  - direct 200-subset: H2 vs baseline
  - fine-grained 5 dims x 100: H2 vs baseline (clarity/coherence/creativity/depth/relevance judged separately locally)
Preservation failures are recorded but do not block TEST. NEVER used for selection.
Usage: python src/test_headline.py --models L1
"""
import json, argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))
from h2_lib import load_splits, l2_ngram, jsd_1gram, recovery, pathologies
from smoke_h2 import load_model, generate_batch
from full_h2 import generate_with_hooks_form
from preserve_full import build_hooks

EVAL_REV_NOTE = "Qwen/Qwen3.5-0.8B frozen eval tokenizer"


def save_progress(df, path, timing_path, timings):
    df.to_parquet(path)
    timing_path.write_text(json.dumps(timings))


def complete_generations(gp, ids):
    if not gp.exists():
        return None
    df = pd.read_parquet(gp)
    required = {"prompt_id", "prompt", "human", "baseline", "ablated"}
    if required <= set(df.columns) and len(df) == len(ids) and set(df["prompt_id"].astype(str)) == set(ids):
        return df.assign(prompt_id=df["prompt_id"].astype(str)).set_index("prompt_id").loc[ids].reset_index()
    return None


def upsert_runs(path, records):
    replacements = {record["run_id"]: record for record in records}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    output, written = [], set()
    for row in rows:
        run_id = row.get("run_id")
        if run_id in replacements:
            if run_id not in written:
                output.append(replacements[run_id])
                written.add(run_id)
        else:
            output.append(row)
    output.extend(record for run_id, record in replacements.items() if run_id not in written)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in output))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=25)
    a = ap.parse_args()
    from transformers import AutoTokenizer as AT
    TOK_PATH = ROOT / ".hf_token"
    HT = TOK_PATH.read_text().strip() if TOK_PATH.exists() else None
    eval_tok = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=HT)
    splits = load_splits(["test", "test_jmq"])
    floors_path = ROOT / "artifacts/metrics/val2_floor.json"
    floors = json.loads(floors_path.read_text())
    test = splits["test"][:a.n]
    tjmq = splits["test_jmq"]
    completed = []
    for mid in a.models:
        geom = ROOT / "artifacts" / "geometry" / "H2" / mid
        fr = json.load(open(geom / "frozen.json"))
        assert fr.get("frozen"), f"{mid}: no frozen config — TEST stays locked"
        pf = json.load(open(geom / "preservation_full.json")) if (geom / "preservation_full.json").exists() else {"pass": None}
        if pf.get("pass"):
            print(f"[{mid}] TEST {len(test)} prompts (frozen {fr['frozen']}, preserve pass)", flush=True)
        else:
            print(f"[{mid}] TEST {len(test)} prompts (frozen {fr['frozen']}, preserve {pf.get('pass')} — insight for H3, not block)", flush=True)
        prompts = [r["prompt"] for r in test]
        ids = [str(r["prompt_id"]) for r in test]
        hum = [r["human_text"] for r in test]
        gp = geom / "test_generations.parquet"
        df = complete_generations(gp, ids)
        timing_path = geom / "test_progress.json"
        timings = json.loads(timing_path.read_text()) if timing_path.exists() else {"base_s": 0.0, "ablated_s": 0.0}
        model = None
        if df is None:
            progress_path = geom / "test_progress.parquet"
            if progress_path.exists():
                df = pd.read_parquet(progress_path)
                if len(df) != len(ids) or set(df["prompt_id"].astype(str)) != set(ids):
                    df = None
            if df is None:
                df = pd.DataFrame({"prompt_id": ids, "prompt": prompts, "human": hum,
                                   "baseline": [None] * len(ids), "ablated": [None] * len(ids),
                                   "base_done": False, "ablated_done": False})
            df = df.assign(prompt_id=df["prompt_id"].astype(str)).set_index("prompt_id").loc[ids].reset_index()
            tok, model, hooks, rec = build_hooks(mid)
            hd = {li: (B, mu, al) for li, (B, mu, al) in hooks.items()}
            import time
            for column, done, generator in [
                ("baseline", "base_done", lambda ps: generate_batch(model, tok, ps, seed_base=1000, max_new=1024)),
                ("ablated", "ablated_done", lambda ps: generate_with_hooks_form(
                    model, tok, ps, hd, form="cone", seed_base=1000, max_new=1024)[0]),
            ]:
                todo = [i for i, value in enumerate(df[done].fillna(False)) if not bool(value)]
                for start in range(0, len(todo), a.chunk):
                    idx = todo[start:start + a.chunk]
                    t0 = time.time()
                    values = generator([prompts[i] for i in idx])
                    timings["base_s" if column == "baseline" else "ablated_s"] += time.time() - t0
                    df.loc[idx, column] = values
                    df.loc[idx, done] = True
                    save_progress(df, progress_path, timing_path, timings)
                    print(f"[{mid}] {column} {int(df[done].sum())}/{len(df)}", flush=True)
            df[["prompt_id", "prompt", "human", "baseline", "ablated"]].to_parquet(gp)
            progress_path.unlink(missing_ok=True)
        else:
            print(f"[{mid}] reusing complete TEST generations", flush=True)
        base, abl = df["baseline"].tolist(), df["ablated"].tolist()
        t_base, dt = timings["base_s"], timings["ablated_s"]
        old_metrics = json.loads((geom / "test_metrics.json").read_text()) if (geom / "test_metrics.json").exists() else {}
        out = {"model": mid, "frozen": fr["frozen"], "n": len(test),
               "recovery_floor_source": str(floors_path.relative_to(ROOT))}
        for n in [1, 2, 3]:
            db, _ = l2_ngram(base, hum, eval_tok, n=n)
            da, contrib = l2_ngram(abl, hum, eval_tok, n=n)
            floor = floors[f"L2_{n}gram_hh"]
            out[f"L2_{n}_base"], out[f"L2_{n}_abl"] = db, da
            out[f"L2_{n}_hh"] = floor
            out[f"R_L2_{n}"] = recovery(db, da, floor)
            if n == 1:
                out["top_words"] = [eval_tok.decode([k[0][0]]) if isinstance(k[0], tuple) else str(k[0]) for k in contrib[:15]]
        out["JSD_base"] = jsd_1gram(base, hum, eval_tok)
        out["JSD_abl"] = jsd_1gram(abl, hum, eval_tok)
        out["path_base"] = pathologies(base)
        out["path_abl"] = pathologies(abl)
        out["kappa"] = dt / max(t_base, 1e-6) if t_base else old_metrics.get("kappa")
        json.dump(out, open(geom / "test_metrics.json", "w"), indent=2)
        kappa = f"{out['kappa']:.2f}" if out["kappa"] is not None else "—"
        print(f"[{mid}] TEST R_L2_1={out['R_L2_1']:+.3f} kappa={kappa}", flush=True)
        metric_fields = {k: v for k, v in out.items() if k not in {"top_words", "model", "frozen"}}
        records = [
            dict(run_id=f"H2-test-{mid}-baseline", hypothesis="H2-test", model=mid,
                 split=f"TEST-{len(test)}", seed_base=1000,
                 sampler={"temperature": 0.8, "top_p": 0.95, "top_k": 50}, status="ok", **metric_fields),
            dict(run_id=f"H2-test-{mid}-ablated", hypothesis="H2-test", model=mid,
                 split=f"TEST-{len(test)}", seed_base=1000,
                 sampler={"temperature": 0.8, "top_p": 0.95, "top_k": 50}, status="ok",
                 frozen=fr["frozen"], **metric_fields),
        ]
        upsert_runs(ROOT / "runs.jsonl", records)
        completed.append((mid, geom, fr))
        if model is not None:
            del model
        torch.cuda.empty_cache()
    # judging feed (all TEST-complete models)
    rows = []
    for geom in sorted((ROOT / "artifacts" / "geometry" / "H2").glob("*")):
        mid = geom.name
        gp = geom / "test_generations.parquet"
        if not gp.exists():
            continue
        df = pd.read_parquet(gp).set_index("prompt_id")
        for r in tjmq:
            pid = str(r["prompt_id"])
            if pid not in df.index:
                continue
            d = df.loc[pid]
            rows.append({"pair_id": f"{mid}-HvH-{pid}", "model": mid, "matchup": "h2-vs-human",
                         "prompt": d["prompt"], "a": d["ablated"], "b": r["human_text"]})
            rows.append({"pair_id": f"{mid}-BvH-{pid}", "model": mid, "matchup": "baseline-vs-human",
                         "prompt": d["prompt"], "a": d["baseline"], "b": r["human_text"]})
        sub = tjmq[:200]
        for r in sub:
            pid = str(r["prompt_id"])
            if pid not in df.index:
                continue
            d = df.loc[pid]
            rows.append({"pair_id": f"{mid}-HvB-{pid}", "model": mid, "matchup": "h2-vs-baseline",
                         "prompt": d["prompt"], "a": d["ablated"], "b": d["baseline"]})
    jp = ROOT / "judging" / "inputs"
    jp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(jp / "test_jmq_pairs.parquet")
    print(f"JUDGING FEED {len(rows)} pairs -> {jp / 'test_jmq_pairs.parquet'} (judge LOCALLY via local_jmq.py)", flush=True)
    for mid, geom, fr in completed:
        (geom / "test_done").write_text(json.dumps({"n": len(test), "frozen": fr["frozen"]}))


if __name__ == "__main__":
    main()
