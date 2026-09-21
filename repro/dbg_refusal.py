import pandas as pd
f = pd.read_parquet("metrics/jmq_bootstrap.parquet")
g = f[(f["available"]) & (f["matchup"] == "h2-vs-baseline")]
for _, r in g.sort_values("model").iterrows():
    print(r["model"], int(r["n_total"]), int(r["n_excluded_both_refused"]), int(r["n_total_all"]))