import json
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
rep = json.load(open("repro/paper_rep_final.json", encoding="utf-8"))
L = ["# Paper rephrase side-by-side (Q20 cone, 39 paragraphs)", "",
     "Accepted into the manuscript: 02:1, H:0, H:1, 04:1, 07:3, 07:4. Everything else kept staff-written.",
     "Rejected: 20 meta-responses (option lists, not passages), 3 number mismatches (2 real, 1 comma artifact since accepted),",
     "11 stale (rewritten by the subagent pass after splitting), 1 weakened-out, 1 verbatim copy.", ""]
for jid in ["02_introduction.tex:1", "H_jmq.tex:0", "07_results_h2.tex:1", "02_introduction.tex:2", "11_conclusion.tex:0", "07_results_h2.tex:3"]:
    L.append("## " + jid + " [" + rep[jid] + "]")
    L.append("")
    L.append("Mine:")
    L.append("")
    L.append(jobs[jid]["text"])
    L.append("")
    L.append("Model:")
    L.append("")
    L.append(outs[jid][:1200])
    L.append("")
open("repro/paper_sxs.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("sxs-written")