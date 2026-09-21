import pandas as pd
f = pd.read_parquet("dump_mirror/judging/inputs/test_jmq_pairs.parquet")
print(list(f.columns))
print(f["domain"].value_counts().to_dict() if "domain" in f.columns else "no-domain-col")