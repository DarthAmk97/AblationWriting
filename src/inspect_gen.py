"""
inspect_gen.py — inspect GEN + dmitva schema, no torch needed (needs datasets)
Writes artifacts/dataset_schema.json
"""
import os, json
from pathlib import Path
ROOT = Path("/root/AblationWriting")
os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".hf_cache" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(ROOT / ".hf_cache" / "datasets")
tok = (ROOT / ".hf_token").read_text().strip() if (ROOT / ".hf_token").exists() else None
if tok:
    os.environ["HF_TOKEN"] = tok

from datasets import load_dataset

def inspect(repo, name, rev=None):
    print(f"\n===== {name} {repo} =====", flush=True)
    try:
        ds = load_dataset(repo, revision=rev, trust_remote_code=False)
        print(f"splits: {list(ds.keys())}", flush=True)
        out = {"repo": repo, "splits": {}}
        for split, d in ds.items():
            print(f"--- split {split}: n={len(d)} columns={d.column_names} ---", flush=True)
            out["splits"][split] = {"n": len(d), "columns": d.column_names}
            # show 2 examples truncated
            for i in range(min(2, len(d))):
                ex = d[i]
                # truncate long fields
                trunc = {}
                for k, v in ex.items():
                    s = str(v)
                    trunc[k] = s[:2000] + ("...[trunc]" if len(s) > 2000 else "")
                print(json.dumps({"idx": i, **trunc}, indent=2)[:4000], flush=True)
                if i == 0:
                    out["splits"][split]["example0_keys"] = list(ex.keys())
        return out
    except Exception as e:
        import traceback
        print(f"ERROR {e}", flush=True)
        traceback.print_exc()
        return {"repo": repo, "error": str(e)}

results = {}
results["GEN"] = inspect("szyszy/GEN", "GEN")
results["PAIRED"] = inspect("dmitva/human_ai_generated_text", "PAIRED")

with open(ROOT / "artifacts" / "dataset_schema.json", "w") as f:
    json.dump(results, f, indent=2)
print("WROTE artifacts/dataset_schema.json")
