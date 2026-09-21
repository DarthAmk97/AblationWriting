import json
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs3.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs3.json", encoding="utf-8"))}
for jid in sorted(outs):
    print(jid, "IDENTICAL" if outs[jid] == jobs[jid]["text"] else "changed")
    if outs[jid] != jobs[jid]["text"]:
        print("  OLD:", jobs[jid]["text"][:200])
        print("  NEW:", outs[jid][:200])