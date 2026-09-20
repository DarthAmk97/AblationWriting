"""refusal_sample.py — show flagged baselines to judge true-vs-false refusals."""
import re
from pathlib import Path
import pandas as pd
ROOT = Path("/root/AblationWriting")
REF = re.compile(r"(i'?m sorry|i cannot|i can'?t|cannot comply|unable to|not able to|against policy|helpline|trusted adult|mental health professional|substance abuse)", re.I)
df = pd.read_parquet(ROOT / "artifacts/geometry/H2/L1/s3_generations.parquet", columns=["baseline"])
flag = df["baseline"].str[:600].str.contains(REF, regex=True)
sub = df[flag].head(8)
for i, t in enumerate(sub["baseline"]):
    m = REF.search(t[:600])
    print(f"--- {i} match={m.group(0)!r} ---", flush=True)
    print(t[:500].replace("\n", " | "), flush=True)
