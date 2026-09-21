import json
d = json.load(open("exports/jmq_case_probe.json", encoding="utf-8"))
cand = []
for i, r in enumerate(d):
    bt, at = r.get("baseline_truncated"), r.get("ablated_truncated")
    if (bt or at) and (bt != at):
        cand.append((i, r.get("model"), r.get("domain"), r.get("jmq_direct_winner"), bt, at))
print(len(cand))
for c in cand[:10]:
    print(c)
anyt = [i for i, r in enumerate(d) if r.get("baseline_truncated") or r.get("ablated_truncated")]
print("any-trunc", len(anyt))