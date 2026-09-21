"""appendix_panels.py - HIP-style preference panels (short cells, small type)."""
import json
from pathlib import Path
import sys
sys.path.insert(0, "src")
from appendix_examples import esc
ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "exports/jmq_case_probe.json", encoding="utf-8"))
NAMES = {"L1": "Llama-3.2-1B", "O1": "OLMo-2 1B", "Q08": "Qwen3.5-0.8B",
         "Q20": "Qwen3.5-2B", "G2": "Gemma-4-E2B"}
PLAIN = {"wikihow": "how-to", "arxiv": "rephrase", "wikipedia": "reference"}
PICKS = [
    (24, "Refusal decides: baseline refuses, ablated complies and wins by default."),
    (16, "Fidelity decides: both severed mid-sentence, baseline keeps the facts."),
    (47, "Format decides: presentation beats content."),
    (144, "Fidelity decides the other way: ablated keeps the findings."),
]
def cut(t, n=200):
    t = (t or "").replace("\n", " ")
    return t[:n] + (" [...]" if len(t) > n else "")
L = ["{\\small", "\\begin{longtable}{p{0.22\\textwidth}p{0.24\\textwidth}p{0.24\\textwidth}p{0.24\\textwidth}}",
     "\\caption{Blind preference up close: what the judge actually rewards. Winner, truncation flags, and texts.}\\label{tab:panels} \\\\ \\hline",
     "Prompt & Baseline & Ablated & Human \\\\ \\hline", "\\endfirsthead",
     "Prompt & Baseline & Ablated & Human \\\\ \\hline", "\\endhead"]
for idx, (i, note) in enumerate(PICKS):
    r = d[i]
    head = "Panel %s (%s, %s): blind wins for %s. Baseline cut: %s. Ablated cut: %s. %s" % (
        "ABCD"[idx], NAMES.get(r["model"], r["model"]), PLAIN.get(r["domain"], r["domain"]),
        r["jmq_direct_winner"], r["baseline_truncated"], r["ablated_truncated"], note)
    L.append("\\multicolumn{4}{p{0.94\\textwidth}}{\\textbf{" + esc(head) + "}} \\\\ \\hline")
    L.append(" & ".join(esc(x) for x in [cut(r["prompt"], 160), cut(r["baseline_output"], 200),
                                         cut(r["ablated_output"], 200), cut(r["human_output"], 200)]) + " \\\\ \\hline")
L.append("\\end{longtable}")
L.append("}")
out = ROOT / "paper" / "tables" / "G_panels.tex"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("WROTE", out, "panels=4", flush=True)