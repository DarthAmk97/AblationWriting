import json, random
random.seed(7)
jobs = []
for m in ["L1", "O1", "Q08", "Q20", "G2"]:
    t = open("hf_cards_v3/" + m + "/README.md", encoding="utf-8").read()
    lines = t.split("\n")
    ai = 0
    ci = 0
    n = 0
    for l in lines:
        s = l.strip()
        if s == "## Abstract":
            ai = n
        if s.startswith("Collection:") and ci == 0:
            ci = n
        n = n + 1
    head = []
    started = False
    for l in lines:
        if l.startswith("# ") and not started:
            started = True
        if not started:
            continue
        if l.strip() == "":
            continue
        head.append(l)
        if l.startswith("Collection:"):
            break
    intro = "\n\n".join(head[0:3])
    abody = []
    k = ai + 1
    while k < ci:
        if lines[k].strip() != "":
            abody.append(lines[k])
        k = k + 1
    nonempties = []
    for l in lines:
        if l.strip() != "":
            nonempties.append(l)
    texts = {"intro": intro, "abstract": abody[0], "fieldnote": nonempties[len(nonempties) - 1]}
    pick = random.sample(["intro", "abstract", "fieldnote"], 2)
    for k in pick:
        jobs.append({"job": m + "-" + k, "model": m, "kind": k, "text": texts[k]})
json.dump(jobs, open("rephrase_jobs.json", "w", encoding="utf-8"), indent=1)
print("jobs", len(jobs))
for j in jobs:
    print(j["job"], len(j["text"]))