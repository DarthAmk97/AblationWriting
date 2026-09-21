import json, re
jobs = {x["job"]: x for x in json.load(open("rephrase_rw2_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_rw2_outs.json", encoding="utf-8"))}
def nums(t):
    return sorted(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t))
def keys(t, cmd):
    return sorted(re.findall(cmd + r"\{([^}]*)\}", t))
BAD = ("here are a few ways", "option 1", "option 2", "**option", "here is the rephrased",
       "here is a rephrased", "to make sense of", "what is actually happening")
for jid, new in outs.items():
    old = jobs[jid]["text"]
    low = new.lower()
    if any(b in low for b in BAD):
        v = "REJECT-FORMAT"
    elif nums(old) != nums(new):
        v = "REJECT-NUMBERS"
    elif keys(old, "cite") != keys(new, "cite") or keys(old, "ref") != keys(new, "ref"):
        v = "REJECT-KEYS"
    else:
        v = "ACCEPT"
    print(jid, v, len(old), "->", len(new))