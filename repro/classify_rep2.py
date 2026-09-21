import json, re
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs2.json", encoding="utf-8"))}
def nums(t):
    return sorted(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t))
def keys(t, cmd):
    return sorted(re.findall(cmd + r"\{([^}]*)\}", t))
BAD = ("here are a few ways", "option 1", "option 2", "**option", "here is the rephrased",
       "here is a rephrased", "to make sense of", "what is actually happening")
rep = {}
for jid, new in outs.items():
    old = jobs[jid]["text"]
    low = new.lower()
    if any(b in low for b in BAD):
        rep[jid] = "REJECT-FORMAT"
    elif nums(old) != nums(new):
        rep[jid] = "REJECT-NUMBERS"
    elif keys(old, "cite") != keys(new, "cite") or keys(old, "ref") != keys(new, "ref"):
        rep[jid] = "REJECT-KEYS"
    else:
        rep[jid] = "ACCEPT"
from collections import Counter
print(Counter(rep.values()))
print(sorted([k for k, v in rep.items() if v == "ACCEPT"]))
json.dump(rep, open("repro/paper_rep_final2.json", "w", encoding="utf-8"), indent=1)