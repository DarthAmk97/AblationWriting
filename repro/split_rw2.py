import json
t = open("paper/sections/03_related_work.tex", encoding="utf-8").read()
lines = t.split("\n")
jobs = []
i = 0
n = 0
while i < len(lines):
    if lines[i].startswith("\\paragraph{"):
        head = lines[i]
        i += 1
        body = []
        while i < len(lines) and lines[i].strip() != "" and not lines[i].startswith("\\paragraph{"):
            body.append(lines[i])
            i += 1
        jobs.append({"job": "RW:" + str(n), "file": "paper/sections/03_related_work.tex",
                     "head": head, "text": head + "\n" + "\n".join(body)})
        n += 1
    else:
        i += 1
json.dump(jobs, open("rephrase_rw_jobs.json", "w", encoding="utf-8"), indent=1)
print("rw-jobs", len(jobs), [len(j["text"]) for j in jobs])