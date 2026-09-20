"""frozen_floor.py — stable human-human floor from VAL1 256 halves (128 vs 128), fixed eval tokenizer.
Saves artifacts/metrics/val1_floor.json. CPU-only. S1's 16-vs-16 split floor is too noisy (sign flips, |R|>5)."""
import json
from pathlib import Path
ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))
from h2_lib import load_splits, l2_ngram, jsd_1gram
TOK = None
try:
    from transformers import AutoTokenizer as AT
    ht = (ROOT / ".hf_token").read_text().strip() if (ROOT / ".hf_token").exists() else None
    TOK = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=ht)
except Exception as e:
    print(f"tokenizer fail {e}", flush=True)
    raise SystemExit(1)
splits = load_splits(["val1"])
val = splits["val1"]  # 256
assert len(val) == 256, len(val)
hum = [r["human_text"] for r in val]
h1, h2 = hum[:128], hum[128:]
out = {}
for n in [1, 2, 3]:
    d, _ = l2_ngram(h1, h2, TOK, n=n)
    out[f"L2_{n}gram_hh"] = d
    print(f"L2-{n} hh={d:.6f}", flush=True)
out["JSD_hh"] = jsd_1gram(h1, h2, TOK)
print(f"JSD hh={out['JSD_hh']:.6f}", flush=True)
out["n"] = 256
p = ROOT / "artifacts" / "metrics"
p.mkdir(parents=True, exist_ok=True)
json.dump(out, open(p / "val1_floor.json", "w"), indent=2)
print("WROTE artifacts/metrics/val1_floor.json", flush=True)
