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


def stage_table():
    import json
    cm = pd.read_parquet(ROOT / "metrics/control_metrics.parquet")
    cb = pd.read_parquet(ROOT / "metrics/combined_test_jmq.parquet").set_index("model")
    jb = pd.read_parquet(ROOT / "metrics/jmq_bootstrap.parquet")
    jb = jb[(jb["available"]) & (jb["matchup"] == "h2-vs-baseline")].set_index("model")
    L = ["\\begin{tabular}{lcccccc}", "\\toprule",
         "Model & VAL1 L2-1 base $\\to$ cone & VAL2 MMD cone & TEST MMD cone & TEST L2 R 1/2/3 & TEST JMQ \\\\",
         "\\midrule"]
    for m in ["G2", "L1", "O1", "Q08", "Q20"]:
        pol = json.load(open(ROOT / "artifacts/geometry/H2" / m / "controls_lexical_policy.json"))
        cal = pol["calibration_val1"]
        v1 = "%.4f $\\to$ %.4f" % (cal["baseline_l2_1"], cal["h2_l2_1"])
        v2 = cm[(cm["split"] == "val2") & (cm["metric"] == "MMD") & (cm["arm"] == "VAL2-CONE") & (cm["model"] == m)].iloc[0]
        tm = cm[(cm["split"] == "test_jmq") & (cm["metric"] == "MMD") & (cm["arm"] == "ANCHOR-CONE") & (cm["model"] == m)].iloc[0]
        L.append("%s & %s & %+.3f & %+.3f & %+.3f/%+.3f/%+.3f & %.3f \\\\" % (
            HUMAN[m], v1, v2["recovery"], tm["recovery"],
            cb.loc[m, "r_l2_1"], cb.loc[m, "r_l2_2"], cb.loc[m, "r_l2_3"],
            jb.loc[m, "estimate"]))
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "T_stages.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("T-stages-ok", flush=True)


def refusal_table():
    # Exclusions are exact from jmq_bootstrap.parquet. Retention comes from the
    # preservation battery as recorded in claims_ledger.md (raw records lived
    # on the retired compute box); G2 has no separate refusal number, gates green.
    f = pd.read_parquet(ROOT / "metrics/jmq_bootstrap.parquet")
    g = f[(f["available"]) & (f["matchup"] == "h2-vs-baseline")].set_index("model")
    retention = {"L1": "0.61", "O1": "0.933", "Q08": "0.818", "Q20": "0.933", "G2": "---"}
    L = ["\\begin{tabular}{lccc}", "\\toprule",
         "Model & Bilateral excluded & Harmful-refusal retention & Benign refusals \\\\",
         "\\midrule"]
    for m in ["G2", "L1", "O1", "Q08", "Q20"]:
        r = g.loc[m]
        L.append("%s & %d of %d & %s & 0 \\\\" % (
            HUMAN[m], int(r.n_excluded_both_refused), int(r.n_total_all), retention[m]))
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "T_refusal.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("T-refusal-ok", flush=True)


def drift_table():
    a = pd.read_parquet(ROOT / "metrics/drift_audit.parquet") if (ROOT / "metrics/drift_audit.parquet").exists() else None
    if a is None:
        a = pd.read_csv(ROOT / "metrics/drift_audit.csv")
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Model & Base & Cone & Random & Lexical \\\\",
         "\\midrule"]
    for m in ["G2", "L1", "O1", "Q08", "Q20"]:
        g = a[a["model"] == m].set_index("arm")
        L.append("%s & %.3f & %.3f & %.3f & %.3f \\\\" % (
            HUMAN[m] + " num", g.loc["ANCHOR-BASE", "num_rate"], g.loc["ANCHOR-CONE", "num_rate"],
            g.loc["RAND-RANK-DOSE", "num_rate"], g.loc["LEX-MATCH", "num_rate"]))
        L.append("%s & %.3f & %.3f & %.3f & %.3f \\\\" % (
            HUMAN[m] + " ent", g.loc["ANCHOR-BASE", "ent_rate"], g.loc["ANCHOR-CONE", "ent_rate"],
            g.loc["RAND-RANK-DOSE", "ent_rate"], g.loc["LEX-MATCH", "ent_rate"]))
    L += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "T_drift.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("T-drift-ok", flush=True)


def main():
    control_main()
    jmq_main()
    style_table()
    domain_table()
    domain_top()
    stage_table()
    refusal_table()
    drift_table()
    print("ALL-TABLES-DONE")


if __name__ == "__main__":
    main()