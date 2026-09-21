import json, re
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
STALE = ("05_method.tex", "03_related_work.tex", "06_experimental_setup.tex")
def nums(t):
    return sorted(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t))
def keys(t, cmd):
    return sorted(re.findall(cmd + r"\{([^}]*)\}", t))
META = ("here are a few ways", "option 1", "option 2", "**option")
rep = {}
for jid, j in jobs.items():
    if j["file"].endswith(STALE):
        rep[jid] = "STALE-SOURCE"
        continue
    old, new = j["text"], outs[jid]
    low = new.lower()
    if any(m in low for m in META):
        rep[jid] = "REJECT-META-RESPONSE"
    elif nums(old) != nums(new):
        rep[jid] = "REJECT-NUMBERS"
    elif keys(old, "cite") != keys(new, "cite") or keys(old, "ref") != keys(new, "ref"):
        rep[jid] = "REJECT-KEYS"
    else:
        rep[jid] = "ACCEPT"
from collections import Counter
print(Counter(rep.values()))
json.dump(rep, open("repro/paper_rep_final.json", "w", encoding="utf-8"), indent=1)
acc = [k for k, v in rep.items() if v == "ACCEPT"]
print(len(acc), acc)