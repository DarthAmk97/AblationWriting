import pandas as pd
r = pd.read_json("dump_mirror/runs.jsonl", lines=True)
s = r[(r["stage"] == "s2") & (r["model"].isin(["L1", "O1", "Q08", "Q20", "G2"]))].copy()
fw = {"L1": "W-BEST", "O1": "W-BEST", "Q08": "W-A", "Q20": "W-B", "G2": "W-A"}
for m, w in fw.items():
    d = s[(s["model"] == m) & (s["window"] == w)].sort_values("rank")
    print(m, [(int(x[0]), round(float(x[1]), 3)) for x in zip(d["rank"], d["R_L2"])])