import pandas as pd
p = pd.read_parquet("exports/jmq_case_probe.parquet") if __import__("os").path.exists("exports/jmq_case_probe.parquet") else None
import glob, os
print(glob.glob("exports/jmq_case*"))
print(os.path.exists("exports/jmq_case_probe.md"))