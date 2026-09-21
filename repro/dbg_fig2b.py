import pandas as pd
r = pd.read_json("dump_mirror/runs.jsonl", lines=True)
s = r[r["stage"].isin(["s1", "s2"])].copy()
for c in ["R_L2", "R_L2_1", "R_L2_2", "R_L2_3", "L2_base", "L2_abl", "L2_1_base"]:
    print(c, float(s[c].notna().mean()), s[c].max() if s[c].notna().any() else None)