import json
jobs = {x["job"]: x for x in json.load(open("rephrase_rw2_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_rw2_outs.json", encoding="utf-8"))}
acc = ["RW2:1", "RW2:2", "RW2:5", "RW2:6", "RW2:7", "GNEG:0"]
t = open("paper/sections/03_related_work.tex", encoding="utf-8").read()
n = 0
for job in acc[:5]:
    old = jobs[job]["head"] + "\n" + jobs[job]["text"] if jobs[job]["head"] not in jobs[job]["text"] else jobs[job]["text"]
    new = jobs[job]["head"] + "\n" + outs[job]
    assert t.count(old) == 1, (job, t.count(old))
    t = t.replace(old, new)
    n += 1
open("paper/sections/03_related_work.tex", "w", encoding="utf-8").write(t)
g = open("paper/appendix/G_negatives.tex", encoding="utf-8").read()
job = "GNEG:0"
old, new = jobs[job]["text"], outs[job]
assert g.count(old) == 1, (job, g.count(old))
open("paper/appendix/G_negatives.tex", "w", encoding="utf-8").write(g.replace(old, new))
print("merged", n + 1)