import json
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
j = jobs["02_introduction.tex:0"]
t = open(j["file"], encoding="utf-8").read()
print("old-len", len(j["text"]))
print("head-in-file", j["text"][:80] in t)
print("file-has-crlf", "\r\n" in t)
print("job-head", repr(j["text"][:80]))