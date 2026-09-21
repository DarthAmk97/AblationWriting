import json, re
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
def nums(t):
    return sorted(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t))
for jid in ["02_introduction.tex:3", "11_conclusion.tex:0"]:
    old, new = jobs[jid]["text"], outs[jid]
    print("=" * 20, jid, "OLD:")
    print(old)
    print("-" * 20, "NEW:")
    print(new[:1500])
    print("-" * 20, "old-nums:", nums(old))
    print("new-nums:", nums(new))