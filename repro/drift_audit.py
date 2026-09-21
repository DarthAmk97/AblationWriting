import json, re
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
rows = [json.loads(l) for l in open("dump_mirror/data/splits/test_jmq.jsonl", encoding="utf-8")]
ax = [r for r in rows if r.get("domain") == "arxiv"]
print("arxiv-prompts", len(ax))
num_re = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
def items(text):
    nums = set(x.replace(",", "").replace(" ", "") for x in num_re.findall(text or ""))
    ents = set(e.text.strip().lower() for e in nlp(text or "").ents if len(e.text.strip()) > 1)
    return nums, ents
out = []
for m in ["L1", "O1", "Q08", "Q20", "G2"]:
    f = pd.read_parquet("artifacts/geometry/H2/" + m + "/controls_generations.parquet")
    g = f[f["split"] == "test_jmq"].set_index(["arm", "prompt_id"])["text"].to_dict()
    for r in ax:
        pid = str(r["prompt_id"])
        hnums, hents = items(r.get("human_text", ""))
        for arm in ["ANCHOR-BASE", "ANCHOR-CONE", "RAND-RANK-DOSE", "LEX-MATCH"]:
            t = g.get((arm, pid), "") or ""
            tl = t.lower()
            nn = sum(1 for x in hnums if x and x in tl.replace(",", "").replace(" ", "")) if hnums else 0
            ne = sum(1 for x in hents if x and x in tl) if hents else 0
            out.append({"model": m, "arm": arm, "pid": pid, "n_nums": len(hnums), "n_ents": len(hents),
                        "keep_nums": nn, "keep_ents": ne})
res = pd.DataFrame(out)
agg = res.groupby(["model", "arm"])[["n_nums", "n_ents", "keep_nums", "keep_ents"]].sum()
agg["num_rate"] = agg["keep_nums"] / agg["n_nums"]
agg["ent_rate"] = agg["keep_ents"] / agg["n_ents"]
print(agg.round(3).to_string())
agg.to_csv("metrics/drift_audit.csv")
print("saved metrics/drift_audit.csv")