import pandas as pd, re, json
frames = {}
for m in ["L1", "O1", "Q08", "Q20", "G2"]:
    f = pd.read_parquet("artifacts/geometry/H2/" + m + "/controls_generations.parquet")
    frames[m] = f[(f["split"] == "test_jmq")].copy()
print({m: sorted(frames[m]["arm"].unique().tolist()) for m in frames})