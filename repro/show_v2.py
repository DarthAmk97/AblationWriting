import json
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs2.json", encoding="utf-8"))}
for jid in ["01_abstract.tex:0", "02_introduction.tex:2"]:
    print("=" * 20, jid, "OLD:")
    print(jobs[jid]["text"][:600])
    print("-" * 20, "NEW:")
    print(outs[jid][:900])