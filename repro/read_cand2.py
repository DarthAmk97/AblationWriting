import json
d = json.load(open("exports/jmq_case_probe.json", encoding="utf-8"))
shown = 0
for i, r in enumerate(d):
    rat = (r.get("jmq_raw_rationale") or "")
    if r.get("jmq_direct_winner") == "baseline" and "format" in rat.lower() and r.get("model") == "Q08":
        print("=" * 20, i, r["model"], r["domain"])
        print("BASELINE:", r["baseline_output"][:300])
        print("ABLATED:", r["ablated_output"][:300])
        print("RATIONALE:", rat[:450])
        shown += 1
        if shown >= 1:
            break
for i, r in enumerate(d):
    rat = (r.get("jmq_raw_rationale") or "")
    if r.get("jmq_direct_winner") == "baseline" and "grammar" in rat.lower():
        print("=" * 20, "grammar", i, r["model"], r["domain"])
        print("RATIONALE:", rat[:450])
        break