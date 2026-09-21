import json, re
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
rep = json.load(open("repro/paper_rep_final.json", encoding="utf-8"))
def nums(t):
    return sorted(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t))
for jid, v in rep.items():
    if v == "REJECT-NUMBERS":
        print("=" * 20, jid)
        print("OLD-NUMS:", nums(jobs[jid]["text"]))
        print("NEW-NUMS:", nums(outs[jid]))
        print("NEW-TEXT:", outs[jid][:500])