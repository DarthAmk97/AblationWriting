"""
inspect_gen2.py — deep inspect GEN 3 configs
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
for cfg in ["human", "ai_generated", "ai_edited"]:
    print(f"\n===== GEN config {cfg} =====", flush=True)
    ds = load_dataset("szyszy/GEN", cfg, trust_remote_code=False)
    print(f"splits {list(ds.keys())}", flush=True)
    for split in ds:
        d = ds[split]
        print(f"split {split} n={len(d)} cols={d.column_names}", flush=True)
        ex = d[0]
        for k, v in ex.items():
            s = str(v)
            print(f"  {k}: {s[:1500]}", flush=True)
            if len(s) > 1500:
                print("   ...[trunc]", flush=True)
        # check prompt id field?
        # try to get unique prompt ids count
        # look for prompt_id, prompt, id columns
        break
