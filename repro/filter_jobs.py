import json
jobs = json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))
jobs = [j for j in jobs if "/draft_" not in j["file"]]
json.dump(jobs, open("rephrase_paper_jobs.json", "w", encoding="utf-8"), indent=1)
print("kept", len(jobs))