"""paper_tables.py - render manuscript .tex tables from frozen parquets."""
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
HUMAN = {"L1": "Llama-3.2-1B", "O1": "OLMo-2 1B", "Q08": "Qwen3.5-0.8B",
         "Q20": "Qwen3.5-2B", "G2": "Gemma-4-E2B"}
def esc(s):
    return str(s).replace("_", "\\_")

def control_main():
    f = pd.read_parquet(ROOT / "metrics/control_metrics.parquet")
    t = f[(f["split"] == "test_jmq") & (f["metric"] == "MMD")].copy()
    order_m = ["G2", "L1", "O1", "Q08", "Q20"]
    order_a = ["ANCHOR-CONE", "RAND-RANK-DOSE", "LEX-MATCH"]
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Model & Cone & Dose-matched random & Lexical \\\\",
         "\\midrule"]
    for m in order_m:
        cells = []
        for a in order_a:
            r = t[(t["model"] == m) & (t["arm"] == a)].iloc[0]
            cells.append("$%+.3f$ [%+.3f, %+.3f]" % (r.recovery, r.recovery_ci_low, r.recovery_ci_high))
        L.append("%s & %s & %s & %s \\\\" % (HUMAN[m], cells[0], cells[1], cells[2]))
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "T_control_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("T-control-ok", flush=True)

def jmq_main():
    f = pd.read_parquet(ROOT / "metrics/jmq_bootstrap.parquet")
    g = f[(f["available"]) & (f["matchup"] == "h2-vs-baseline")].copy()
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Model & Included & Ablated wins & Share [95\\% CI] & Holm $p$ \\\\",
         "\\midrule"]
    for _, r in g.sort_values("model").iterrows():
        L.append("%s & %d & %d & %.3f [%.3f, %.3f] & %.3g \\\\" % (
            HUMAN.get(r.model, r.model), int(r.n_total), int(r.n_a),
            r.estimate, r.ci_low, r.ci_high, r.p_holm))
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "T_jmq_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("T-jmq-ok", flush=True)

def style_table():
    f = pd.read_parquet(ROOT / "metrics/style_bigword.parquet")
    if "model" in f.columns and f["model"].nunique() > 1:
        # pooled cells are the unweighted mean of per-model means; verified
        # against metrics/tables/style_bigword.md (arxiv 5.405/5.564/5.075)
        f = f.groupby(["domain", "arm"], as_index=False)["mean_word_len"].mean()
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Domain & Human & Ablated & Baseline \\\\",
         "\\midrule"]
    for d, grp in f.groupby("domain", sort=True):
        row = {"domain": d}
        for _, r in grp.iterrows():
            row[r.arm] = "%.2f" % r.mean_word_len
        L.append("%s & %s & %s & %s \\\\" % (esc(d), row.get("human", "--"),
                                             row.get("ablated", "--"), row.get("baseline", "--")))
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "T_style.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("T-style-ok", flush=True)

def domain_table():
    f = pd.read_parquet(ROOT / "metrics/jmq_domain_wins.parquet")
    L = ["\\begin{longtable}{llrrrr}", "\\caption{Blind wins by domain and model, refusal-filtered.}\\\\ \\hline",
         "Domain & Model & $n$ & Ablated & Baseline & Share \\\\ \\hline", "\\endfirsthead",
         "Domain & Model & $n$ & Ablated & Baseline & Share \\\\ \\hline", "\\endhead"]
    for _, r in f.sort_values(["domain", "model"]).iterrows():
        L.append("%s & %s & %d & %d & %d & %.3f \\\\ \\hline" % (
            esc(r.domain), HUMAN.get(r.model, r.model), int(r.n), int(r.ablated),
            int(r.baseline), r.ablated_share_decisive))
    L.append("\\end{longtable}")
    (OUT / "T_domain.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("T-domain-ok", flush=True)

def domain_top():
    f = pd.read_parquet(ROOT / "metrics/jmq_domain_wins.parquet")
    f = f.sort_values("ablated_share_decisive")
    sel = pd.concat([f.head(4), f.tail(4)])
    L = ["\\begin{tabular}{llrrr}", "\\toprule",
         "Domain & Model & $n$ & Ablated & Share \\\\",
         "\\midrule"]
    for _, r in sel.iterrows():
        L.append("%s & %s & %d & %d & %.3f \\\\" % (
            esc(r.domain), HUMAN.get(r.model, r.model), int(r.n), int(r.ablated),
            r.ablated_share_decisive))
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "T_domain_top.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("T-domain-top-ok", flush=True)


def main():
    control_main()
    jmq_main()
    style_table()
    domain_table()
    domain_top()
    print("ALL-TABLES-DONE")


if __name__ == "__main__":
    main()