import json
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
jobs3 = json.load(open("rephrase_paper_jobs3.json", encoding="utf-8"))
for j in jobs3[:3]:
    print(j["job"])
    print("CURRENT:", repr(j["text"][:120]))
print("OUT04:", repr(outs.get("04_hypotheses.tex:1", "MISSING")[:120]))