import json
d = json.load(open("exports/jmq_case_probe.json", encoding="utf-8"))
out = []
gram = [i for i, r in enumerate(d) if r.get("jmq_direct_winner") == "baseline" and "grammar" in (r.get("jmq_raw_rationale") or "").lower()]
out.append("grammar-candidates " + str([(i, d[i]["model"], d[i]["domain"]) for i in gram[:5]]))
fid = [i for i, r in enumerate(d) if r.get("jmq_direct_winner") == "ablated" and ("fidel" in (r.get("jmq_raw_rationale") or "").lower())]
out.append("fidelity-ablation " + str([(i, d[i]["model"], d[i]["domain"]) for i in fid[:6]]))
for i in gram[:1] + fid[:2]:
    r = d[i]
    out.append("=" * 20 + " " + str(i) + " " + r["model"] + " " + r["domain"] + " win:" + r["jmq_direct_winner"])
    out.append("BASELINE: " + (r["baseline_output"] or "")[:300])
    out.append("ABLATED: " + (r["ablated_output"] or "")[:300])
    out.append("RATIONALE: " + (r["jmq_raw_rationale"] or "")[:400])
open("repro/panel_preview.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("preview-written")