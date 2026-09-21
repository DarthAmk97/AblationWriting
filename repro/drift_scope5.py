import pandas as pd
f = pd.read_parquet("dump_mirror/judging/inputs/test_jmq_pairs.parquet")
print(f["pair_id"].iloc[:3].tolist())
import glob
print(glob.glob("dump_mirror/data/splits/*.jsonl"))
print(glob.glob("dump_mirror/**/*.jsonl")[:8])