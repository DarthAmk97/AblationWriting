import json, shutil, os
jobs = {x["job"]: x for x in json.load(open("rephrase_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_outs.json", encoding="utf-8"))}
accept = ["L1-abstract", "O1-abstract", "O1-intro", "Q08-intro", "Q20-abstract", "G2-intro"]
for m in ["L1", "O1", "Q08", "Q20", "G2"]:
    os.makedirs("hf_cards_v4/" + m, exist_ok=True)
    shutil.copy("hf_cards_v3/" + m + "/README.md", "hf_cards_v4/" + m + "/README.md")
n = 0
for job in accept:
    m = jobs[job]["model"]
    p = "hf_cards_v4/" + m + "/README.md"
    t = open(p, encoding="utf-8").read()
    old, new = jobs[job]["text"], outs[job]
    assert t.count(old) == 1, (job, t.count(old))
    open(p, "w", encoding="utf-8").write(t.replace(old, new))
    n += 1
print("replaced", n)