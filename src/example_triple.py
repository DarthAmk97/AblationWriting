"""example_triple.py — print one baseline/ablated/human triple for a config (trimmed for chat).
Default: most-changed readable pair. --random: seeded-random readable pair (honest, not cherry-picked).
Both modes filter refusal-borderline baselines (refusal analysis is a separate track)."""
import sys
from pathlib import Path
ROOT = Path("/root/AblationWriting")
import pandas as pd
mid, config = sys.argv[1], sys.argv[2]
RANDOM_MODE = "--random" in sys.argv
splits = {}
import json
for name in ["val2", "val1"]:
    try:
        for line in open(ROOT / "data" / "splits" / f"{name}.jsonl"):
            r = json.loads(line)
            splits[str(r["prompt_id"])] = r
    except Exception:
        pass
df = pd.read_parquet(ROOT / "artifacts" / "geometry" / "H2" / mid / "s3_generations.parquet")
g = df[df["config"] == config].copy()
import re
REF = re.compile(r"(i'?m sorry|i cannot|i can'?t|cannot comply|unable to|not able to|against policy|helpline|trusted adult|mental health professional)", re.I)
n0 = len(g)
g = g[~g["baseline"].str[:600].str.contains(REF, regex=True)]
print(f"filtered {n0 - len(g)}/{n0} refusal-borderline baselines", flush=True)
g["blen"] = g["baseline"].str.split().str.len()
g["alen"] = g["ablated"].str.split().str.len()
g = g[(g["blen"] > 60) & (g["blen"] < 300) & (g["alen"] > 60) & (g["alen"] < 300)]
if RANDOM_MODE:
    import random as _r
    idx = _r.Random(7).choice(list(g.index))
    r = g.loc[idx]
    print("mode=seeded-random", flush=True)
else:
    def dist(a, b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        return 1 - len(sa & sb) / max(len(sa | sb), 1)
    g["d"] = [dist(a, b) for a, b in zip(g["ablated"], g["baseline"])]
    g = g.sort_values("d", ascending=False)
    r = g.iloc[0]
    print("mode=most-changed", flush=True)
h = splits.get(str(r["prompt_id"]), {}).get("human_text", "[human ref missing]")
print(f"MODEL {mid} CONFIG {config} prompt_id={r['prompt_id']}", flush=True)
print(f"PROMPT: {r['prompt'] if 'prompt' in g.columns else splits.get(str(r['prompt_id']), {}).get('prompt', '')}"[:800], flush=True)
for label, t in [("BASELINE", r["baseline"]), ("ABLATED", r["ablated"]), ("HUMAN", h)]:
    print(f"\n===== {label} ({len(str(t).split())} words) =====", flush=True)
    print(str(t)[:1800], flush=True)
