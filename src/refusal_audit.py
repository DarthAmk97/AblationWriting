"""refusal_audit.py — CPU-only. Refusal prevalence in baseline generations per split/stage (out-of-space contamination check)."""
import re
from pathlib import Path
import pandas as pd
ROOT = Path("/root/AblationWriting")
REF = re.compile(r"(i'?m sorry|i cannot|i can'?t|cannot comply|unable to|not able to|against policy|helpline|trusted adult|mental health professional|substance abuse)", re.I)
for pq in sorted((ROOT / "artifacts" / "geometry").glob("H2/*/s*_generations.parquet")):
    try:
        df = pd.read_parquet(pq, columns=["baseline"])
    except Exception as e:
        print(pq.parent.name, pq.name, "ERR", e, flush=True)
        continue
    n = len(df)
    nb = int(df["baseline"].str[:600].str.contains(REF, regex=True).sum())
    print(f"{pq.parent.name} {pq.name}: {nb}/{n} refusal-borderline baselines ({100*nb/max(n,1):.1f}%)", flush=True)
