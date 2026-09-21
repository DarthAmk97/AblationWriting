import pandas as pd
c = pd.read_parquet("metrics/combined_test_jmq.parquet")
print(list(c.columns)[:20])
j = pd.read_parquet("metrics/jmq_domain_wins.parquet")
print(j[["model", "domain"]].drop_duplicates().groupby("domain").size().to_dict())