import json
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs2.json", encoding="utf-8"))}
rep = json.load(open("repro/paper_rep_final2.json", encoding="utf-8"))
n = skipped = 0
by_file = {}
for job, v in rep.items():
    if v == "ACCEPT":
        by_file.setdefault(jobs[job]["file"], []).append(job)
for fpath, jids in by_file.items():
    t = open(fpath, encoding="utf-8").read()
    for job in jids:
        old, new = jobs[job]["text"], outs[job]
        c = t.count(old)
        if c == 0:
            print("skip-already-replaced", job)
            skipped += 1
            continue
        assert c == 1, (job, c)
        t = t.replace(old, new)
        n += 1
    open(fpath, "w", encoding="utf-8").write(t)
print("replaced", n, "skipped", skipped)