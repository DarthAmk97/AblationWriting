import json
L = ["# Paper rephrase side-by-side, all rounds (Q20 cone)",
     "",
     "Round 1 (39 paragraphs): 6 merged, 20 meta-responses out, 11 stale, 2 fouls out, 1 copy skipped.",
     "Round 2 (23 hardened reruns): 23/23 format-clean; 10 merged, 13 stale-skipped (files had moved on).",
     "Round 3 (10 fresh paragraphs): all clean, 0 substantive changes, nothing merged.",
     "Total merged: 16 paragraphs. Rule throughout: numbers, cite keys, and ref keys identical, else keep staff text.",
     ""]
jobs1 = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs1 = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
for jid in ["02_introduction.tex:1", "H_jmq.tex:0"]:
    L += ["## " + jid + " [merged round 1]", "", "Mine:", "", jobs1[jid]["text"], "",
          "Model:", "", outs1[jid][:900], ""]
jobs3 = {x["job"]: x for x in json.load(open("rephrase_paper_jobs3.json", encoding="utf-8"))}
outs3 = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs3.json", encoding="utf-8"))}
for jid in ["07_results_h2.tex:1", "02_introduction.tex:2", "11_conclusion.tex:0", "07_results_h2.tex:3"]:
    if jid in jobs3 and jid in outs3:
        L += ["## " + jid + " [round 3, not merged]", "", "Mine:", "", jobs3[jid]["text"][:700], "",
              "Model:", "", outs3[jid][:700], ""]
open("repro/paper_sxs.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("sxs-regen-ok")