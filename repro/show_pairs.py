import json
jobs = {x["job"]: x for x in json.load(open("rephrase_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"] for x in json.load(open("rephrase_outs.json", encoding="utf-8"))}
for jid in ["Q20-intro", "L1-intro", "G2-fieldnote"]:
    print("=" * 30, jid, "ORIGINAL:")
    print(jobs[jid]["text"])
    print("-" * 30, "MODEL:")
    print(outs[jid].strip())
    print()