import json, glob, os
files = sorted(glob.glob("paper/sections/*.tex") + glob.glob("paper/appendix/A_metrics.tex") +
               glob.glob("paper/appendix/B_prompts.tex") + glob.glob("paper/appendix/C_models_data.tex") +
               glob.glob("paper/appendix/F_preservation.tex") + glob.glob("paper/appendix/G_negatives.tex") +
               glob.glob("paper/appendix/H_jmq.tex"))
jobs = []
for fp in files:
    t = open(fp, encoding="utf-8").read()
    blocks = [b.strip() for b in t.split("\n\n")]
    n = 0
    for b in blocks:
        if not b or b.startswith("%") or b.startswith("\\"):
            continue
        if any(k in b for k in ["\\begin", "\\end", "\\input", "\\caption", "\\label", "&", "\\\\"]):
            continue
        if len(b) < 120:
            continue
        jobs.append({"job": os.path.basename(fp) + ":" + str(n), "file": fp, "text": b})
        n += 1
json.dump(jobs, open("rephrase_paper_jobs.json", "w", encoding="utf-8"), indent=1)
print("para-jobs", len(jobs))
for j in jobs:
    print(j["job"], len(j["text"]))