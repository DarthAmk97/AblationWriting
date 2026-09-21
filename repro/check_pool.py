import pandas as pd
f = pd.read_parquet("metrics/style_bigword.parquet")
g = f.groupby(["domain", "arm"], as_index=False)["mean_word_len"].mean()
print(g[g["domain"] == "arxiv"].to_dict("records"))
print(g[(g["domain"] == "reddit") & (g["arm"] == "human")].to_dict("records"))