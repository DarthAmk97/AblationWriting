import pandas as pd
a = pd.read_csv("metrics/drift_audit.csv")
HUMAN = {"L1": "Llama-3.2-1B", "O1": "OLMo-2 1B", "Q08": "Qwen3.5-0.8B", "Q20": "Qwen3.5-2B", "G2": "Gemma-4-E2B"}
ARMS = {"ANCHOR-BASE": "Base", "ANCHOR-CONE": "Cone", "RAND-RANK-DOSE": "Random", "LEX-MATCH": "Lexical"}
L = ["\\begin{tabular}{lcccc}", "\\toprule",
     "Model & Base & Cone & Random & Lexical \\\\",
     "\\midrule"]
for m in ["G2", "L1", "O1", "Q08", "Q20"]:
    g = a[a["model"] == m].set_index("arm")
    row = "numbers kept"
    L.append("%s & %.3f & %.3f & %.3f & %.3f \\\\" % (
        HUMAN[m] + " num", g.loc["ANCHOR-BASE", "num_rate"], g.loc["ANCHOR-CONE", "num_rate"],
        g.loc["RAND-RANK-DOSE", "num_rate"], g.loc["LEX-MATCH", "num_rate"]))
    L.append("%s & %.3f & %.3f & %.3f & %.3f \\\\" % (
        HUMAN[m] + " ent", g.loc["ANCHOR-BASE", "ent_rate"], g.loc["ANCHOR-CONE", "ent_rate"],
        g.loc["RAND-RANK-DOSE", "ent_rate"], g.loc["LEX-MATCH", "ent_rate"]))
L += ["\\bottomrule", "\\end{tabular}"]
open("paper/tables/T_drift.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("T-drift-ok")