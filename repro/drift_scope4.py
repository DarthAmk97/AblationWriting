import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
f = pd.read_parquet("dump_mirror/judging/inputs/test_jmq_pairs.parquet")
px = f[["prompt"]].drop_duplicates()
print(len(px))
ax = px[px["prompt"].str.startswith("Rephrase the abstract")]
print("arxiv-like-prompts", len(ax))