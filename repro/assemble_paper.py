import json
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
acc = ["H_jmq.tex:0", "H_jmq.tex:1", "02_introduction.tex:1", "04_hypotheses.tex:1", "07_results_h2.tex:4"]
# 07:3 manually verified: only comma + spelled-out differences
acc.append("07_results_h2.tex:3")
import os
for job in acc:
    m = jobs[job]["model"] if "model" in jobs[job] else None
for job in acc:
    j = jobs[job]
    fpath = j["file"]
    t = open(fpath, encoding="utf-8").read()
    old, new = j["text"], outs[job]
    assert t.count(old) == 1, (job, t.count(old))
    open(fpath, "w", encoding="utf-8").write(t.replace(old, new))
print("replaced", len(acc))