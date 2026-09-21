import json, difflib
jobs = {x["job"]: x for x in json.load(open("rephrase_rw2_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_rw2_outs.json", encoding="utf-8"))}
for jid in ["RW2:1", "RW2:2", "RW2:5", "RW2:6", "RW2:7", "GNEG:0"]:
    print("=" * 20, jid, "OLD:")
    print(jobs[jid]["text"][:500])
    print("-" * 20, "NEW:")
    print(outs[jid][:700])