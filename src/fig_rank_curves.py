"""fig_rank_curves.py - median screen recovery vs rank per frozen window."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
HUMAN = {"L1": "Llama-3.2-1B", "O1": "OLMo-2 1B", "Q08": "Qwen3.5-0.8B",
         "Q20": "Qwen3.5-2B", "G2": "Gemma-4-E2B"}
FROZEN = {"L1": 4, "O1": 2, "Q08": 8, "Q20": 32, "G2": 8}
FW = {"L1": "W-BEST", "O1": "W-BEST", "Q08": "W-A", "Q20": "W-B", "G2": "W-A"}
r = pd.read_json(ROOT / "dump_mirror/runs.jsonl", lines=True)
s = r[r["stage"] == "s2"].copy()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for m in ["L1", "O1", "Q08", "Q20", "G2"]:
    d = s[(s["model"] == m) & (s["window"] == FW[m])]
    ranks = sorted(d["rank"].dropna().unique().tolist())
    med = [float(d[d["rank"] == rk]["R_L2"].median()) for rk in ranks]
    ax.plot(ranks, med, marker="o", markersize=4, label=HUMAN[m])
    ax.axvline(FROZEN[m], linestyle=":", linewidth=0.8)
ax.set_xscale("log")
ax.set_xticks([2, 4, 8, 16, 32], [2, 4, 8, 16, 32])
ax.tick_params(which="minor", labelbottom=False)
ax.set_ylabel("median screen R (unigram)")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend(fontsize=7)
ax.set_xlabel("rank (dotted: frozen rank)")
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/fig_rank_curves.png", dpi=300)
fig.savefig(ROOT / "paper/figures/fig_rank_curves.pdf")
print("fig-rank-ok")