import json
jobs = {x["job"]: x for x in json.load(open("rephrase_rw_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_rw_outs.json", encoding="utf-8"))}
acc = ["RW:1", "RW:2", "RW:3", "RW:5"]
t = open("paper/sections/03_related_work.tex", encoding="utf-8").read()
n = 0
for jid in acc:
    old, new = jobs[jid]["text"], outs[jid]
    assert t.count(old) == 1, (jid, t.count(old))
    t = t.replace(old, new)
    n += 1
for jid in ["RW:0", "RW:4", "RW:6"]:
    old, new = jobs[jid]["text"], outs[job] if False else outs[jid]
    head = jobs[jid]["head"]
    assert t.count(old) == 1, (jid, t.count(old))
    t = t.replace(old, head + "\n" + new)
    n += 1
open("paper/sections/03_related_work.tex", "w", encoding="utf-8").write(t)
print("merged", n)