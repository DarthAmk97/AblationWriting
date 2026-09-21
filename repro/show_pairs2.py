import json
jobs = {x["job"]: x for x in json.load(open("rephrase_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"] for x in json.load(open("rephrase_outs.json", encoding="utf-8"))}
for jid in ["O1-intro", "O1-abstract", "Q08-intro", "Q08-fieldnote", "Q20-abstract", "L1-abstract", "G2-intro"]:
    print("=" * 25, jid, "ORIGINAL:")
    print(jobs[jid]["text"])
    print("-" * 25, "MODEL:")
    print(outs[jid].strip())
    print()