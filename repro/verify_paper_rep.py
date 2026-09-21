import json, re
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
def nums(t):
    return sorted(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t))
def keys(t, cmd):
    return sorted(re.findall(cmd + r"\{([^}]*)\}", t))
STALE = ("05_method.tex", "03_related_work.tex", "06_experimental_setup.tex")
rep = {}
for jid, j in jobs.items():
    if j["file"].endswith(STALE):
        rep[jid] = ("STALE-SOURCE", 0, 0)
        continue
    old, new = j["text"], outs[jid]
    ok = nums(old) == nums(new)
    ck = keys(old, "cite") == keys(new, "cite")
    rk = keys(old, "ref") == keys(new, "ref")
    rep[jid] = ("OK" if (ok and ck and rk) else "CHECK", len(old), len(new))
    print(jid, rep[jid][0], rep[jid][1], "->", rep[jid][2])
    if rep[jid][0] == "CHECK":
        print("  nums:", nums(old) == nums(new), "cites:", ck, "refs:", rk)
json.dump(rep, open("repro/paper_rep_report.json", "w", encoding="utf-8"), indent=1)
print("OK:", sum(1 for v in rep.values() if v[0] == "OK"))