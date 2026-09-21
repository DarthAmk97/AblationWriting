"""paper_figures.py - build all manuscript figures from frozen artifacts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "figures" / "data"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
HUMAN = {"L1": "Llama-3.2-1B", "O1": "OLMo-2 1B", "Q08": "Qwen3.5-0.8B",
         "Q20": "Qwen3.5-2B", "G2": "Gemma-4-E2B"}
MODELS = ["L1", "O1", "Q08", "Q20", "G2"]

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + ".png"), dpi=300)
    fig.savefig(FIG / (name + ".pdf"))
    plt.close(fig)
    print("fig-ok", name, flush=True)

# Fig 2: sweep heatmap, best R_L2_1 by window x rank
runs = pd.read_json(ROOT / "dump_mirror/runs.jsonl", lines=True)
s = runs[runs["stage"].isin(["s1", "s2"])].copy()
s = s[s["model"].isin(MODELS)]
fig, axes = plt.subplots(1, 5, figsize=(9, 2.6), sharey=True)
ranks = [2, 4, 8, 16, 32]
for ax, m in zip(axes, MODELS):
    d = s[s["model"] == m]
    wins = sorted(d["window"].unique().tolist())
    mat = np.full((len(wins), len(ranks)), np.nan)
    for i, w in enumerate(wins):
        for j, r in enumerate(ranks):
            v = d[(d["window"] == w) & (d["rank"] == r)]["R_L2_1"]
            if len(v):
                mat[i, j] = v.max()
    im = ax.imshow(mat, aspect="auto", cmap="Greens", vmin=-0.1, vmax=0.5)
    ax.set_xticks(range(len(ranks)), ranks)
    ax.set_yticks(range(len(wins)), wins, fontsize=7)
    ax.set_title(HUMAN[m], fontsize=8)
    ax.set_xlabel("rank")
fig.suptitle("Figure 2: sweep recovery (best R L2-1 by window and rank)", fontsize=9)
fig.colorbar(im, ax=axes, label="R L2-1", shrink=0.8)
savefig(fig, "fig2_sweep_heatmap")

# Fig 3: selectivity frontier, refusal damage vs cone MMD
cm = pd.read_parquet(ROOT / "metrics/control_metrics.parquet")
t = cm[(cm["split"] == "test_jmq") & (cm["metric"] == "MMD")]
cone = {r.model: float(r.recovery) for r in t[t["arm"] == "ANCHOR-CONE"].itertuples()}
damage = {"L1": 1 - 0.61, "O1": 1 - 0.933, "Q08": 1 - 0.818, "Q20": 1 - 0.933}
fig, ax = plt.subplots(figsize=(4.5, 3.5))
for m, d in damage.items():
    ax.scatter([d], [cone[m]], s=60)
    ax.annotate(HUMAN[m], (d, cone[m]), textcoords="offset points", xytext=(5, 5), fontsize=7)
ax.set_xlabel("refusal damage (1 - retention)")
ax.set_ylabel("cone TEST MMD recovery")
ax.set_title("Figure 3: selectivity frontier (Gemma gate green, refusal number n/a)", fontsize=9)
savefig(fig, "fig3_selectivity_frontier")

# Fig 4b: control bars with CIs
arms = ["ANCHOR-CONE", "RAND-RANK-DOSE", "LEX-MATCH"]
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(MODELS))
w = 0.22
for i, arm in enumerate(arms):
    vals, los, his = [], [], []
    for m in MODELS:
        r = t[(t["model"] == m) & (t["arm"] == arm)].iloc[0]
        vals.append(float(r.recovery))
        los.append(float(r.recovery) - float(r.recovery_ci_low))
        his.append(float(r.recovery_ci_high) - float(r.recovery))
    ax.bar(x + (i - 1) * w, vals, w, yerr=[los, his], capsize=2, label=arm)
ax.set_xticks(x, [HUMAN[m] for m in MODELS], fontsize=7)
ax.set_ylabel("TEST MMD recovery")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend(fontsize=7)
ax.set_title("Figure 4: falsification bars, cone vs dose-matched random vs lexical", fontsize=9)
savefig(fig, "fig4_control_bars")
t[t["arm"].isin(arms + ["ANCHOR-BASE"])].to_parquet(DATA / "fig4_control_bars.parquet", index=False)

# Fig 5: spectra, cumulative variance + frozen rank markers
ranks_frozen = {"L1": 4, "O1": 2, "Q08": 8, "Q20": 32, "G2": 8}
fig, axes = plt.subplots(1, 5, figsize=(9, 2.6), sharey=True)
spec_rows = []
for ax, m in zip(axes, MODELS):
    curves = []
    d = Path("fetch_spectra") / m / "basis"
    for f in sorted(d.glob("full_layer*.npz")):
        z = np.load(ROOT / f)
        s = np.asarray(z["S"], dtype=float)
        v = np.cumsum(s ** 2)
        v = v / v[-1]
        curves.append(v[:256])
        spec_rows.append({"model": m, "layer_file": f.name})
    curves = np.stack(curves)
    xx = np.arange(1, curves.shape[1] + 1)
    ax.plot(xx, curves.T, linewidth=0.8)
    ax.axvline(ranks_frozen[m], color="black", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_title(HUMAN[m], fontsize=8)
    ax.set_xlabel("components")
axes[0].set_ylabel("cumulative variance share")
fig.suptitle("Figure 5: discovery spectra with frozen ranks (dashed)", fontsize=9)
savefig(fig, "fig5_spectra")
print("ALL-FIGS-DONE")