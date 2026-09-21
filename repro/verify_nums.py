import json, re
jobs = json.load(open("rephrase_jobs.json", encoding="utf-8"))
outs = {x["job"]: x["out"] for x in json.load(open("rephrase_outs.json", encoding="utf-8"))}
def nums(t):
    return sorted(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t.replace("\u2019", "'")))
rep = []
for j in jobs:
    old, new = j["text"], outs[j["job"]].strip()
    a, b = nums(old), nums(new)
    ok = (a == b)
    rep.append((j["job"], ok, len(old), len(new)))
    print(j["job"], "NUMOK" if ok else "NUMFAIL", len(old), "->", len(new))
    if not ok:
        print("  old:", a)
        print("  new:", b)