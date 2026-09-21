import json
t = open("paper/sections/03_related_work.tex", encoding="utf-8").read()
blocks = [b.strip() for b in t.split("\n\n")]
jobs = []
n = 0
for b in blocks:
    if not b or b.startswith("%") or b.startswith("\\"):
        continue
    if any(k in b for k in ["\\begin", "\\end", "\\input", "\\caption", "\\label", "&", "\\\\"]):
        continue
    if len(b) < 120:
        continue
    jobs.append({"job": "03rw:" + str(n), "file": "paper/sections/03_related_work.tex", "text": b})
    n += 1
json.dump(jobs, open("rephrase_rw_jobs.json", "w", encoding="utf-8"), indent=1)
print("rw-jobs", len(jobs))