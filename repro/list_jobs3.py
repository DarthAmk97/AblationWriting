import json
jobs = json.load(open("rephrase_paper_jobs3.json", encoding="utf-8"))
for j in jobs:
    print(j["job"], len(j["text"]), j["text"][:70].replace(chr(10), " "))