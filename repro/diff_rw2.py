import json
jobs = {x["job"]: x for x in json.load(open("rephrase_rw2_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_rw2_outs.json", encoding="utf-8"))}
for jid in sorted(outs):
    print(jid, "SAME" if outs[jid] == jobs[jid]["text"] else "DIFF")