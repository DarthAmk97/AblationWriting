"""fig1_method_schematic.py - Figure 1: paired displacements, whitening, cone,
protection scrub, generated-token hook, distribution plus preservation eval."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
fig, ax = plt.subplots(figsize=(7.5, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")
stages = [
    (0.2, "Paired\nmodel-human\nactivations"),
    (2.2, "Whiten +\nSVD cone"),
    (4.2, "Protection\nscrub"),
    (6.2, "Generated-token\nhook"),
    (8.2, "Distribution +\npreservation eval"),
]
for i, (x, label) in enumerate(stages):
    box = patches.FancyBboxPatch((x, 0.7), 1.5, 1.6, boxstyle="round,pad=0.05",
                                 facecolor="white", edgecolor="black")
    ax.add_patch(box)
    ax.text(x + 0.75, 1.5, label, ha="center", va="center", fontsize=8)
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + 1.55, 1.5), xytext=(x + 2.15, 1.5),
                    arrowprops={"arrowstyle": "->", "color": "black"})
ax.set_title("Figure 1: method schematic (data flow, left to right)", fontsize=9, pad=8)
fig.tight_layout()
fig.savefig("paper/figures/fig1_method.png", dpi=300)
fig.savefig("paper/figures/fig1_method.pdf")
print("fig1-ok")