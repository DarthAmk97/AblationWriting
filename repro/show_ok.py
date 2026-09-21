import json
jobs = {x["job"]: x for x in json.load(open("rephrase_paper_jobs.json", encoding="utf-8"))}
outs = {x["job"]: x["out"].strip() for x in json.load(open("rephrase_paper_outs.json", encoding="utf-8"))}
for jid in ["02_introduction.tex:1", "07_results_h2.tex:1", "H_jmq.tex:0", "04_hypotheses.tex:1"]:
    print("=" * 20, jid, "OLD:")
    print(jobs[jid]["text"])
    print("-" * 20, "NEW:")
    print(outs[jid][:900])
    print()