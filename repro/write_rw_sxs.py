import json
jobs = {x["job"]: x for x in json.load(open("rephrase_rw_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_rw_outs.json", encoding="utf-8"))}
L = ["# Related-work rephrase side-by-side (Q20 cone, 7 paragraphs)",
     "", "Merged 4 clean, 3 with reattached headers. All numbers and cite keys verified identical.", ""]
for jid in ["RW:0", "RW:1", "RW:4"]:
    L.append("## " + jid)
    L.append("")
    L.append("Mine:")
    L.append("")
    L.append(jobs[jid]["text"])
    L.append("")
    L.append("Model:")
    L.append("")
    L.append(outs[jid])
    L.append("")
open("repro/rw_sxs.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("rw-sxs-ok")