import json, re
rows = [json.loads(l) for l in open("dump_mirror/data/splits/test_jmq.jsonl", encoding="utf-8")]
print(list(rows[0].keys()))
ax = [r for r in rows if r.get("prompt", "").startswith("Rephrase the abstract")]
print(len(ax), len(rows))