"""
make_splits.py — immutable GEN splits + external + storyscope placeholder
Protocol Sec 3-4. PRE-TEST. Deterministic, no TEST leakage.
Counts target: CAL 2048, VAL1 256, VAL2 512, TEST 2000, TEST-JMQ 400 (first fixed 400 TEST), INTERP 128 CAL-only
If GEN insufficient, reduce proportionally keeping TEST>=1000.
Saves data/splits/{cal,val1,val2,test,test_jmq,external,storyscope,interp}.jsonl
Each row: dataset, revision, prompt_id, duplicate_cluster_id, domain, hash, prompt, human_text, ai_text, ai_model
"""
import os, json, hashlib, random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/root/AblationWriting")
os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".hf_cache" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(ROOT / ".hf_cache" / "datasets")
tok = (ROOT / ".hf_token").read_text().strip() if (ROOT / ".hf_token").exists() else None
if tok:
    os.environ["HF_TOKEN"] = tok

from datasets import load_dataset

GEN_REV = "a0f143c1ae9c35b684898a5f9c0ab6efb49be486"
PAIRED_REV = "dacfc1bc7960967bb515a46e66b07e33bb081ddf"

print("Loading GEN human + ai_generated ...", flush=True)
human = load_dataset("szyszy/GEN", "human", trust_remote_code=False)["train"]
ai = load_dataset("szyszy/GEN", "ai_generated", trust_remote_code=False)["train"]
print(f"human n={len(human)} ai n={len(ai)}", flush=True)

# group
from collections import defaultdict
human_by_pid = defaultdict(list)
for r in human:
    human_by_pid[r["prompt_id"]].append(r)
ai_by_pid = defaultdict(list)
for r in ai:
    ai_by_pid[r["prompt_id"]].append(r)

common_pids = sorted(set(human_by_pid.keys()) & set(ai_by_pid.keys()))
print(f"common prompt_ids: {len(common_pids)}", flush=True)
print(f"human-only: {len(set(human_by_pid)-set(ai_by_pid))} ai-only: {len(set(ai_by_pid)-set(human_by_pid))}", flush=True)

# domain = source
pid_domain = {}
for pid in common_pids:
    # human source (should match ai source? take human)
    pid_domain[pid] = human_by_pid[pid][0].get("source", "unknown")

print("domain counts:", Counter(pid_domain.values()), flush=True)

# deterministic pick: one human (first sorted by text hash), one AI (sorted by model+text hash)
def hhash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

matched = []
for pid in common_pids:
    h_cands = sorted(human_by_pid[pid], key=lambda r: hhash(r["text"]))
    a_cands = sorted(ai_by_pid[pid], key=lambda r: hhash(r["model"] + r["text"]))
    h = h_cands[0]
    a = a_cands[0]
    # duplicate cluster: exact normalized text hash of prompt (to keep semantic dups together, simple exact + normalized)
    norm_prompt = " ".join(h["prompt"].lower().split())
    cluster = hhash(norm_prompt)[:16]
    matched.append({
        "dataset": "szyszy/GEN",
        "revision": GEN_REV,
        "prompt_id": int(pid),
        "duplicate_cluster_id": cluster,
        "domain": h.get("source", "unknown"),
        "hash": hhash(str(pid) + h["prompt"])[:16],
        "prompt": h["prompt"],
        "human_text": h["text"],
        "ai_text": a["text"],
        "ai_model": a.get("model", "unknown"),
        "human_source": h.get("source", ""),
    })

# enforce duplicate clusters wholly to one split: group by cluster (exact prompt duplicates share cluster)
# In GEN, prompt_id unique => cluster unique unless duplicate prompts across ids. Check collisions:
clust_counts = Counter(m["duplicate_cluster_id"] for m in matched)
multi = sum(1 for v in clust_counts.values() if v > 1)
print(f"duplicate clusters: {len(clust_counts)} unique, {multi} clusters with >1 pid (must keep together)", flush=True)

# group by cluster for split assignment
cluster_to_items = defaultdict(list)
for m in matched:
    cluster_to_items[m["duplicate_cluster_id"]].append(m)
clusters = sorted(cluster_to_items.keys())  # deterministic

# stratify by domain: sort clusters by domain then hash for determinism, then allocate proportionally
# Simple: shuffle clusters deterministically with seed 0, but keep multi-item clusters together
rng = random.Random(0)
# stratify: bucket clusters by majority domain
domain_buckets = defaultdict(list)
for c in clusters:
    doms = [x["domain"] for x in cluster_to_items[c]]
    maj = Counter(doms).most_common(1)[0][0]
    domain_buckets[maj].append(c)
for dom in domain_buckets:
    rng.shuffle(domain_buckets[dom])
    domain_buckets[dom].sort()  # ensure deterministic? Actually shuffle already deterministic with seed; keep shuffle order
    # re-shuffle deterministically: we already shuffled; don't sort after
# interleave domains round-robin for balanced splits
ordered_clusters = []
buckets = {k: sorted(v) for k, v in domain_buckets.items()}  # sort for determinism before interleave? Use shuffled?
# redo properly: deterministic shuffle per domain
for dom in buckets:
    lst = buckets[dom]
    rng2 = random.Random(hhash(dom)[:8].encode().hex()[0:8])
    # use hash-seeded shuffle
    r = random.Random(int(hhash("split-seed-"+dom)[:8], 16))
    r.shuffle(lst)
    buckets[dom] = lst
# round robin
domains_sorted = sorted(buckets.keys())
idx = {d: 0 for d in domains_sorted}
while True:
    progressed = False
    for d in domains_sorted:
        if idx[d] < len(buckets[d]):
            ordered_clusters.append(buckets[d][idx[d]])
            idx[d] += 1
            progressed = True
    if not progressed:
        break

# expand to items
ordered_items = []
for c in ordered_clusters:
    # sort items within cluster by prompt_id
    items = sorted(cluster_to_items[c], key=lambda x: x["prompt_id"])
    ordered_items.extend(items)

print(f"ordered matched items: {len(ordered_items)}", flush=True)

# target counts
targets = {"CAL": 2048, "VAL1": 256, "VAL2": 512, "TEST": 2000}
total_needed = sum(targets.values())
print(f"need {total_needed}, have {len(ordered_items)}", flush=True)
if len(ordered_items) < total_needed:
    # reduce proportionally keeping TEST>=1000
    print("INSUFFICIENT — reducing proportionally", flush=True)
    # scale factor
    # keep ratio CAL:VAL1:VAL2:TEST = 2048:256:512:2000
    # reduce until fits
    avail = len(ordered_items)
    # reserve TEST>=1000
    # simple: scale all by avail/total
    scale = avail / total_needed
    n_test = max(1000, int(2000*scale))
    remaining = avail - n_test
    # distribute remaining proportionally to CAL/VAL1/VAL2 (2048+256+512=2816)
    n_cal = int(remaining * 2048/2816)
    n_val1 = int(remaining * 256/2816)
    n_val2 = remaining - n_cal - n_val1
    targets = {"CAL": n_cal, "VAL1": n_val1, "VAL2": n_val2, "TEST": n_test}
    print(f"reduced targets: {targets}", flush=True)

# assign sequentially (already domain-interleaved, cluster-grouped)
splits = {}
cursor = 0
for name in ["CAL", "VAL1", "VAL2", "TEST"]:
    n = targets[name]
    splits[name] = ordered_items[cursor:cursor+n]
    cursor += n
print(f"assigned: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()), flush=True)

# INTERP: 128 CAL-only IDs (subset of CAL, never headline stats)
rng_interp = random.Random(1234)
cal_copy = splits["CAL"][:]
rng_interp.shuffle(cal_copy)
interp = cal_copy[:128]
print(f"INTERP={len(interp)} (subset of CAL)", flush=True)

# TEST-JMQ: first fixed 400 TEST IDs (in TEST order)
test_jmq = splits["TEST"][:400]
print(f"TEST-JMQ={len(test_jmq)}", flush=True)

# External: 1000 from dmitva, disjoint by hash (deterministic hash sampling)
print("Loading dmitva for external ...", flush=True)
from datasets import load_dataset as ld
paired = ld("dmitva/human_ai_generated_text", trust_remote_code=False)["train"]
print(f"dmitva n={len(paired)}", flush=True)
# deterministic: take every k-th by hash order? Sort by id hash, take first 1000
all_ids = []
for i, r in enumerate(paired):
    # use id field hash
    all_ids.append((hhash(r["id"]), i))
all_ids.sort()
ext_idx = [i for _, i in all_ids[:1000]]
external = []
for i in ext_idx:
    r = paired[i]
    external.append({
        "dataset": "dmitva/human_ai_generated_text",
        "revision": PAIRED_REV,
        "prompt_id": r["id"],
        "duplicate_cluster_id": hhash(r["instructions"].lower().split().__str__())[:16] if r["instructions"] else "noinstr",
        "domain": "external",
        "hash": hhash(r["id"])[:16],
        "prompt": r["instructions"] if r["instructions"] else "",
        "human_text": r["human_text"],
        "ai_text": r["ai_text"],
        "ai_model": "unknown-releasedAI",
    })
print(f"external={len(external)}", flush=True)

# StoryScope placeholder: 100 frozen IDs (prompts to be filled when StoryScope code available)
# Create deterministic placeholder IDs storyscope-000..099
storyscope = []
for i in range(100):
    storyscope.append({
        "dataset": "StoryScope",
        "revision": "TBD-code-rev",
        "prompt_id": f"storyscope-{i:03d}",
        "duplicate_cluster_id": f"ss-{i:03d}",
        "domain": "narrative",
        "hash": hhash(f"storyscope-{i}")[:16],
        "prompt": "TBD — fill from released StoryScope prompts, freeze before H2/H3 comparison",
        "human_text": "",
        "ai_text": "",
        "ai_model": "",
    })

# write
outdir = ROOT / "data" / "splits"
outdir.mkdir(parents=True, exist_ok=True)
def write(name, rows):
    p = outdir / f"{name.lower()}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"WROTE {p} n={len(rows)}", flush=True)

write("cal", splits["CAL"])
write("val1", splits["VAL1"])
write("val2", splits["VAL2"])
write("test", splits["TEST"])
write("test_jmq", test_jmq)
write("interp", interp)
write("external", external)
write("storyscope", storyscope)

# summary
counts = {k: len(v) for k, v in splits.items()}
counts.update({"INTERP": len(interp), "TEST_JMQ": len(test_jmq), "EXTERNAL": len(external), "STORYSCOPE": len(storyscope)})
summary = {
    "gen_common_pids": len(common_pids),
    "targets": targets,
    "counts": counts,
    "gen_rev": GEN_REV,
    "paired_rev": PAIRED_REV,
    "duplicate_multi_clusters": multi,
}
with open(outdir / "split_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
