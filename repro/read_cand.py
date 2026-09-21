import json
d = json.load(open("exports/jmq_case_probe.json", encoding="utf-8"))
for i in [24, 16]:
    r = d[i]
    print("=" * 20, i, r["model"], r["domain"], "winner:", r["jmq_direct_winner"])
    print("PROMPT:", r["prompt"][:200])
    print("BASELINE:", r["baseline_output"][:350])
    print("ABLATED:", r["ablated_output"][:350])
    print("RATIONALE:", (r["jmq_raw_rationale"] or "")[:400])
    print()