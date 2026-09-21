import json
jobs = []
for fp, prefix in [("paper/sections/03_related_work.tex", "RW2"), ("paper/appendix/G_negatives.tex", "GNEG")]:
    t = open(fp, encoding="utf-8").read()
    blocks = [b.strip() for b in t.split("\n\n")]
    n = 0
    for b in blocks:
        if not b or b.startswith("%"):
            continue
        if any(k in b for k in ["\\begin", "\\end", "\\input", "\\caption", "\\label", "&", "\\\\"]):
            if not b.startswith("\\paragraph{"):
                continue
            head, _, rest = b.partition("\n")
            b = rest.strip()
        if len(b) < 120:
            continue
        jobs.append({"job": prefix + ":" + str(n), "file": fp, "head": head if "head" in dir() else "", "text": b})
        n += 1
json.dump(jobs, open("rephrase_rw2_jobs.json", "w", encoding="utf-8"), indent=1)
print("rw2-jobs", len(jobs))
for j in jobs:
    print(j["job"], len(j["text"]))