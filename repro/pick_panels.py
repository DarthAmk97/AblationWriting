import json
d = json.load(open("exports/jmq_case_probe.json", encoding="utf-8"))
print(len(d))
story_trunc = [i for i, r in enumerate(d) if r.get("domain") == "story_generation" and r.get("ablated_truncated") != r.get("baseline_truncated")]
print("story-trunc-split", len(story_trunc), [d[i]["model"] + "/" + d[i]["jmq_direct_winner"] for i in story_trunc[:8]])
fmt = [i for i, r in enumerate(d) if r.get("jmq_direct_winner") == "baseline" and "format" in (r.get("jmq_raw_rationale") or "").lower()]
print("format-wins", len(fmt), [(d[i]["model"], d[i]["domain"]) for i in fmt[:6]])
fid = [i for i, r in enumerate(d) if r.get("jmq_direct_winner") == "ablated" and ("fidel" in (r.get("jmq_raw_rationale") or "").lower() or "source" in (r.get("jmq_raw_rationale") or "").lower())]
print("fidelity-wins", len(fid), [(d[i]["model"], d[i]["domain"]) for i in fid[:6]])