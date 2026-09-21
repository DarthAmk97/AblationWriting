import json
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs2.json", encoding="utf-8"))}
rep = json.load(open("repro/paper_rep_final2.json", encoding="utf-8"))
n = 0
by_file = {}
for jid, v in rep.items():
    if v == "ACCEPT":
        by_file.setdefault(jobs[jid]["file"], []).append(jid)
for fpath, jids in by_file.items():
    t = open(fpath, encoding="utf-8").read()
    for job in jids:
        old, new = jobs[job]["text"], outs[job]
        assert t.count(old) == 1, (job, t.count(old))
        t = t.replace(old, new)
    open(fpath, "w", encoding="utf-8").write(t)
    n += len(jids)
print("replaced", n, "in", len(by_file), "files")