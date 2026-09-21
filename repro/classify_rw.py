import json, re
jobs = {x["job"]: x for x in json.load(open("rephrase_rw_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_rw_outs.json", encoding="utf-8"))}
def nums(t):
    return sorted(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t))
def keys(t, cmd):
    return sorted(re.findall(cmd + r"\{([^}]*)\}", t))
BAD = ("here are a few ways", "option 1", "option 2", "**option", "here is the rephrased",
       "here is a rephrased", "to make sense of", "what is actually happening")
for jid, new in outs.items():
    old = jobs[jid]["text"]
    head = jobs[jid]["head"]
    low = new.lower()
    flags = []
    if any(b in low for b in BAD):
        flags.append("FORMAT")
    if nums(old) != nums(new):
        flags.append("NUMBERS")
    if keys(old, "cite") != keys(new, "cite"):
        flags.append("CITES")
    if not new.startswith(head):
        flags.append("HEAD")
    print(jid, "ACCEPT" if not flags else "REJECT-" + "+".join(flags), len(old), "->", len(new))