import pandas as pd
r = pd.read_json("dump_mirror/runs.jsonl", lines=True)
s = r[r["stage"].isin(["s1", "s2"])].copy()
print(s["R_L2_1"].describe().to_dict())
d = s[s["model"] == "L1"]
print(d[d["window"] == "W-A"][["rank", "R_L2_1"]].head(8).to_dict("records"))
print("nan-frac", float(s["R_L2_1"].isna().mean()))