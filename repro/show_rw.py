import json
jobs = {x["job"]: x for x in json.load(open("rephrase_rw_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_rw_outs.json", encoding="utf-8"))}
for jid in ["RW:0", "RW:4"]:
    print("=" * 20, jid)
    print("NEW:", outs[jid][:400])