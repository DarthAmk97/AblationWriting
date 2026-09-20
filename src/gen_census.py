"""gen_census.py — AI-model + domain census of GEN matched sets (CPU-only, cached)."""
from datasets import load_dataset
from collections import Counter
import os
from pathlib import Path
ROOT = Path("/root/AblationWriting")
os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
human = load_dataset("szyszy/GEN", "human", trust_remote_code=False)["train"]
ai = load_dataset("szyszy/GEN", "ai_generated", trust_remote_code=False)["train"]
print("HUMAN n=", len(human), "domains=", Counter(human["source"]), flush=True)
print("HUMAN model field:", Counter(human["model"]), flush=True)
print("AI n=", len(ai), flush=True)
print("AI by model:", Counter(ai["model"]).most_common(40), flush=True)
print("AI by source:", Counter(ai["source"]), flush=True)
print("AI by text_type:", Counter(ai["text_type"]), flush=True)
hp = set(human["prompt_id"])
ap = set(ai["prompt_id"])
print("common pids:", len(hp & ap), "human-only:", len(hp - ap), "ai-only:", len(ap - hp), flush=True)
# per-pid AI count stats
from collections import defaultdict
cnt = defaultdict(int)
for p in ai["prompt_id"]:
    cnt[p] += 1
import statistics
print("AI outputs per pid: min", min(cnt.values()), "max", max(cnt.values()), "mean %.2f" % statistics.mean(cnt.values()), flush=True)
