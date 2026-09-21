"""appendix_panels.py - HIP-style JMQ preference panels from the case probe."""
import json
from pathlib import Path
import sys
sys.path.insert(0, "src")
from appendix_examples import esc
ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "exports/jmq_case_probe.json", encoding="utf-8"))
PICKS = [
    (24, "Refusal decides: baseline refuses, ablated complies and wins by default."),
    (16, "Fidelity decides: both severed mid-sentence, baseline keeps the facts."),
    (47, "Format decides: presentation beats content."),
    (144, "Fidelity decides the other way: ablated keeps the findings."),
]
def cut(t, n=550):
    t = (t or "").replace("\n", " ")
    return t[:n] + (" [...]" if len(t) > n else "")
L = ["\\begin{longtable}{p{0.22\\textwidth}p{0.24\\textwidth}p{0.24\\textwidth}p{0.24\\textwidth}}",
     "\\caption{Blind preference up close: what the judge actually rewards. Winner, truncation flags, and texts.}\\label{tab:panels} \\\\ \\hline",
     "Prompt & Baseline & Ablated & Human \\\\ \\hline", "\\endfirsthead",
     "Prompt & Baseline & Ablated & Human \\\\ \\hline", "\\endhead"]
for idx, (i, note) in enumerate(PICKS):
    r = d[i]
    head = "Panel %s (%s, %s): JMQ %s wins. Baseline cut: %s. Ablated cut: %s. %s" % (
        "ABCD"[idx], r["model"], r["domain"], r["jmq_direct_winner"],
        r["baseline_truncated"], r["ablated_truncated"], note)
    L.append("\\multicolumn{4}{l}{\\textbf{" + esc(head) + "}} \\\\ \\hline")
    L.append(" & ".join(esc(x) for x in [cut(r["prompt"], 400), cut(r["baseline_output"]),
                                         cut(r["ablated_output"]), cut(r["human_output"])]) + " \\\\ \\hline")
L.append("\\end{longtable}")
out = ROOT / "paper" / "tables" / "G_panels.tex"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("WROTE", out, "panels=4", flush=True)