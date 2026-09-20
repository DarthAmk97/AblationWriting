"""Build the joint TEST distribution, preservation, and refusal-filtered JMQ view."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "L1": ("meta-llama/Llama-3.2-1B-Instruct", "Llama-1B"),
    "O1": ("allenai/OLMo-2-0425-1B-Instruct", "OLMo-1B"),
    "Q08": ("Qwen/Qwen3.5-0.8B", "Qwen-0.8B"),
    "Q20": ("Qwen/Qwen3.5-2B", "Qwen-2B"),
    "G2": ("google/gemma-4-E2B-it", "Gemma-2B"),
}


def reduction(base, ablated):
    return 100.0 * (base - ablated) / base


def recovery(base, ablated, floor):
    denominator = base - floor
    if denominator <= 0:
        raise ValueError(f"invalid recovery denominator: baseline={base}, floor={floor}")
    return (base - ablated) / denominator


def load_rows(root):
    jmq = pd.read_parquet(root / "metrics/jmq_bootstrap.parquet")
    jmq = jmq[(jmq["available"]) & (jmq["matchup"] == "h2-vs-baseline")].set_index("model")
    floors = json.loads((root / "dump_mirror/artifacts/metrics/val2_floor.json").read_text())
    rows = []
    for model, (full_name, short_name) in MODELS.items():
        geom = root / "dump_mirror/artifacts/geometry/H2" / model
        test = json.loads((geom / "test_metrics.json").read_text())
        preservation = json.loads((geom / "preservation_full.json").read_text())
        mmd = json.loads((root / "artifacts/geometry/H2" / model / "mmd_metrics.json").read_text())
        if mmd.get("model") != model or set(mmd.get("splits", {})) != {"val2", "test"}:
            raise ValueError(f"{model}: incomplete or mismatched MMD metrics")
        judge = jmq.loc[model]
        row = {
            "model": model, "model_name": full_name, "short_name": short_name,
            "test_n": test["n"], "kappa": test["kappa"],
            "jmq_n": int(judge.n_total), "jmq_excluded_both_refused": int(judge.n_excluded_both_refused),
            "jmq_ablated": judge.estimate, "jmq_baseline": judge.baseline_score,
            "jmq_difference_pp": 100 * judge.score_difference,
            "jmq_difference_ci_low_pp": 100 * (2 * judge.ci_low - 1),
            "jmq_difference_ci_high_pp": 100 * (2 * judge.ci_high - 1),
            "jmq_holm_p": judge.p_holm,
            "ifeval_retention_pct": 100 * preservation["gates"]["ifeval_rel"],
            "capability_retention_pct": 100 * preservation["gates"]["capability_rel"],
            "harm_refusal_retention_pct": 100 * preservation["gates"]["refusal_rel"],
        }
        for order in (1, 2, 3):
            base, ablated = test[f"L2_{order}_base"], test[f"L2_{order}_abl"]
            floor = floors[f"L2_{order}gram_hh"]
            row.update({
                f"l2_{order}_base": base,
                f"l2_{order}_ablated": ablated,
                f"l2_{order}_floor": floor,
                f"l2_{order}_reduction_pct": reduction(base, ablated),
                f"r_l2_{order}": recovery(base, ablated, floor),
            })
        for split in ("val2", "test"):
            value = mmd["splits"][split]
            base = value["baseline_vs_human"]["mmd"]
            ablated = value["ablated_vs_human"]["mmd"]
            floor = value["human_vs_human_floor"]["mmd"]
            if value.get("n") != {"val2": 512, "test": 2000}[split]:
                raise ValueError(f"{model}: unexpected MMD {split} row count")
            if not np.isclose(value["recovery_mmd"], recovery(base, ablated, floor)):
                raise ValueError(f"{model}: inconsistent MMD {split} recovery")
            row.update({
                f"mmd_{split}_base": base,
                f"mmd_{split}_ablated": ablated,
                f"mmd_{split}_floor": floor,
                f"mmd_{split}_recovery": value["recovery_mmd"],
                f"mmd_{split}_reduction_pct": reduction(base, ablated),
            })
        row.update({
            "jsd_base": test["JSD_base"], "jsd_ablated": test["JSD_abl"],
            "jsd_reduction_pct": reduction(test["JSD_base"], test["JSD_abl"]),
            "mmd_bandwidth": mmd["bandwidth"], "mmd_revision": mmd["revision"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def pair(row, name, digits=5):
    base, ablated = row[f"{name}_base"], row[f"{name}_ablated"]
    return f"{base:.{digits}f}→{ablated:.{digits}f} ({row[f'{name}_reduction_pct']:+.1f}%)"


def write_table(data, path):
    lines = [
        "# Joint TEST distribution and JMQ results",
        "",
        "Positive distance reduction means the ablated distribution moved toward the human corpus. "
        "R is human-gap recovery using order-matched frozen VAL2 human-human floors. MMD uses an RBF "
        "bandwidth selected from VAL2 human+baseline embeddings only and reused unchanged on TEST.",
        "",
        "## Distributional movement",
        "",
        "| Model | R L2-1 | L2-1 baseline→ablated | R L2-2 | L2-2 baseline→ablated | R L2-3 | L2-3 baseline→ablated | R MMD | MMD baseline→ablated | JSD baseline→ablated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data.itertuples():
        series = pd.Series(row._asdict())
        lines.append(
            f"| {row.model_name} | {row.r_l2_1:+.3f} | {pair(series, 'l2_1')} | "
            f"{row.r_l2_2:+.3f} | {pair(series, 'l2_2')} | {row.r_l2_3:+.3f} | "
            f"{pair(series, 'l2_3')} | {row.mmd_test_recovery:+.3f} | "
            f"{pair(series, 'mmd_test', 4)} | "
            f"{pair(series, 'jsd', 4)} |"
        )
    lines += [
        "",
        "## Refusal-filtered quality and preservation",
        "",
        "| Model | JMQ included/excluded | Ablated score | Baseline score | A−B | Holm p | IFEval retention | Capability retention | Harm-refusal retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data.itertuples():
        lines.append(
            f"| {row.model_name} | {row.jmq_n}/{row.jmq_excluded_both_refused} | "
            f"{row.jmq_ablated:.4f} | {row.jmq_baseline:.4f} | {row.jmq_difference_pp:+.1f}pp | "
            f"{row.jmq_holm_p:.3g} | {row.ifeval_retention_pct:.1f}% | "
            f"{row.capability_retention_pct:.1f}% | {row.harm_refusal_retention_pct:.1f}% |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bold_pair(baseline, h2, digits, lower_is_better):
    baseline_better = baseline < h2 if lower_is_better else baseline > h2
    h2_better = h2 < baseline if lower_is_better else h2 > baseline
    baseline_text, h2_text = f"{baseline:.{digits}f}", f"{h2:.{digits}f}"
    return (
        f"**{baseline_text}**" if baseline_better else baseline_text,
        f"**{h2_text}**" if h2_better else h2_text,
    )


def write_paired_table(data, path):
    lines = [
        "# H2 vs. baseline on held-out TEST",
        "",
        "| Model / condition | MMD ↓ | Direct JMQ ↑ | Token L2-1 ↓ |",
        "|---|---:|---:|---:|",
    ]
    for row in data.itertuples():
        mmd = _bold_pair(row.mmd_test_base, row.mmd_test_ablated, 4, True)
        jmq = _bold_pair(row.jmq_baseline, row.jmq_ablated, 3, False)
        l2 = _bold_pair(row.l2_1_base, row.l2_1_ablated, 5, True)
        lines.extend([
            f"| {row.short_name} · Baseline | {mmd[0]} | {jmq[0]} | {l2[0]} |",
            f"| {row.short_name} · H2 | {mmd[1]} | {jmq[1]} | {l2[1]} |",
        ])
    lines += [
        "",
        "MMD and Token L2-1 are distances to the human TEST distribution, so lower is closer. "
        "Direct JMQ is the refusal-filtered H2-vs-baseline preference score, so higher is preferred; "
        "the two condition scores are complementary within each model and are not cross-model ratings. "
        "Bold marks the numerically better value in each same-model pair. Holm-significant JMQ losses "
        "occur for Llama-1B, OLMo-1B, and Qwen-2B. MMD recovery percentages remain in the companion table.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paired_figure(data, path):
    background, ink, secondary = "#F8F9FB", "#14171C", "#59616B"
    h2_fill, mmd_accent, rule = "#EEE7EC", "#E3A127", "#252A31"
    fig, ax = plt.subplots(figsize=(15, 8.8), facecolor=background)
    ax.set_facecolor(background)
    ax.axis("off")
    left, right = 0.035, 0.965
    columns = [left, 0.405, 0.625, 0.815]
    ax.text(left, 0.965, "TABLE 4  ·  H2 VS. BASELINE  ·  HELD-OUT TEST  ·  FIVE FROZEN MODELS",
            transform=ax.transAxes, color=secondary, fontsize=10, fontfamily="DejaVu Sans Mono",
            fontweight="bold", va="top")
    ax.plot([left, right], [0.925, 0.925], color=rule, linewidth=1, transform=ax.transAxes)
    headers = ("MODEL / CONDITION", "MMD ↓", "DIRECT JMQ ↑", "TOKEN L2-1 ↓")
    for index, (x, label) in enumerate(zip(columns, headers)):
        if index == 1:
            ax.text(x, 0.885, label, transform=ax.transAxes, fontsize=10.5, color=ink,
                    fontfamily="DejaVu Sans Mono", fontweight="bold", va="center",
                    bbox={"facecolor": mmd_accent, "edgecolor": "none", "pad": 1.5})
        else:
            ax.text(x, 0.885, label, transform=ax.transAxes, fontsize=10.5, color=secondary,
                    fontfamily="DejaVu Sans Mono", fontweight="bold", va="center")
    ax.plot([left, right], [0.85, 0.85], color=rule, linewidth=0.9, transform=ax.transAxes)

    top, row_height = 0.825, 0.066
    for model_index, row in enumerate(data.itertuples()):
        values = [
            ("BASELINE", row.mmd_test_base, row.jmq_baseline, row.l2_1_base),
            ("H2", row.mmd_test_ablated, row.jmq_ablated, row.l2_1_ablated),
        ]
        better = {
            "mmd": 0 if row.mmd_test_base < row.mmd_test_ablated else 1,
            "jmq": 0 if row.jmq_baseline > row.jmq_ablated else 1,
            "l2": 0 if row.l2_1_base < row.l2_1_ablated else 1,
        }
        for condition_index, (condition, mmd, jmq, l2) in enumerate(values):
            row_index = model_index * 2 + condition_index
            y_top = top - row_index * row_height
            y_center = y_top - row_height / 2
            if condition == "H2":
                ax.add_patch(plt.Rectangle(
                    (left, y_top - row_height), right - left, row_height,
                    transform=ax.transAxes, facecolor=h2_fill, edgecolor="none", zorder=0,
                ))
            label = f"{row.short_name.upper()}  {condition}"
            ax.text(columns[0], y_center, label, transform=ax.transAxes, va="center", ha="left",
                    fontsize=10.5, color=secondary, fontfamily="DejaVu Sans Mono")
            numeric = ((mmd, 4, better["mmd"]), (jmq, 3, better["jmq"]), (l2, 5, better["l2"]))
            for metric_index, (value, digits, winner) in enumerate(numeric, start=1):
                ax.text(columns[metric_index], y_center, f"{value:.{digits}f}",
                        transform=ax.transAxes, va="center", ha="left", fontsize=12, color=ink,
                        fontweight="bold" if condition_index == winner else "normal")
            if condition == "H2" and model_index < len(data) - 1:
                ax.plot([left, right], [y_top - row_height, y_top - row_height], color=rule,
                        linewidth=0.75, transform=ax.transAxes)

    foot_y = top - len(data) * 2 * row_height - 0.035
    ax.text(left, foot_y,
            "MMD AND TOKEN L2-1: DISTANCE TO HUMAN TEST TEXT, LOWER IS CLOSER.  "
            "DIRECT JMQ: WITHIN-MODEL PAIRWISE PREFERENCE, HIGHER IS PREFERRED.",
            transform=ax.transAxes, color=secondary, fontsize=9.2, fontfamily="DejaVu Sans Mono", va="top")
    ax.text(left, foot_y - 0.045,
            "BOLD MARKS THE NUMERICAL PAIR WINNER. JMQ LOSSES ARE HOLM-SIGNIFICANT FOR LLAMA-1B, OLMO-1B, AND QWEN-2B.",
            transform=ax.transAxes, color=secondary, fontsize=9.2, fontfamily="DejaVu Sans Mono", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=background)
    plt.close(fig)


def _signed_percent(value):
    digits = 2 if 0 < abs(value) < 0.1 else 1
    return f"{value:+.{digits}f}%"


def _shift(base, h2, digits):
    change = 100 * (h2 - base) / base
    direction = "closer" if h2 < base else "farther" if h2 > base else "unchanged"
    return f"{base:.{digits}f} → {h2:.{digits}f}<br>{_signed_percent(change)} · {direction}"


def write_grouped_distribution_table(data, path):
    lines = [
        "# H2 distributional shift on held-out TEST",
        "",
        "| Model | MMD ↓ | R MMD ↑ | L2-1 ↓ | R L2-1 ↑ | L2-2 ↓ | R L2-2 ↑ | L2-3 ↓ | R L2-3 ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data.itertuples():
        lines.append(
            f"| {row.short_name} | {_shift(row.mmd_test_base, row.mmd_test_ablated, 4)} | "
            f"{_signed_percent(100 * row.mmd_test_recovery)} | "
            f"{_shift(row.l2_1_base, row.l2_1_ablated, 5)} | {_signed_percent(100 * row.r_l2_1)} | "
            f"{_shift(row.l2_2_base, row.l2_2_ablated, 5)} | {_signed_percent(100 * row.r_l2_2)} | "
            f"{_shift(row.l2_3_base, row.l2_3_ablated, 5)} | {_signed_percent(100 * row.r_l2_3)} |"
        )
    lines += [
        "",
        "All four metrics are distances to the human TEST distribution, so lower is closer. "
        "The signed raw change is `100 × (H2 − baseline) / baseline`: **negative is better** because the distance fell. "
        "Human-gap recovery is `R = (baseline − H2) / (baseline − human floor)`: **positive is better** because H2 closed part of the baseline-to-human gap. "
        "Raw change and R can differ in magnitude because R accounts for the nonzero human-human floor.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grouped_distribution_figure(data, path):
    background, ink, secondary = "#F8F9FB", "#14171C", "#59616B"
    closer, farther, row_fill, rule = "#176B87", "#C65D21", "#EEEFF2", "#252A31"
    fig, ax = plt.subplots(figsize=(20, 8.6), facecolor=background)
    ax.set_facecolor(background)
    ax.axis("off")
    left, right = 0.025, 0.98
    shift_x = [0.15, 0.365, 0.58, 0.795]
    recovery_x = [0.31, 0.525, 0.74, 0.955]
    ax.text(left, 0.965, "TABLE 5  ·  DISTRIBUTIONAL SHIFT  ·  HELD-OUT TEST  ·  BASELINE → H2",
            transform=ax.transAxes, color=secondary, fontsize=10, fontfamily="DejaVu Sans Mono",
            fontweight="bold", va="top")
    ax.plot([left, right], [0.925, 0.925], color=rule, linewidth=1, transform=ax.transAxes)
    ax.text(left, 0.875, "MODEL", transform=ax.transAxes, fontsize=10.5, color=secondary,
            fontfamily="DejaVu Sans Mono", fontweight="bold", va="center")
    for index, metric in enumerate(("MMD ↓", "L2-1 ↓", "L2-2 ↓", "L2-3 ↓")):
        ax.text(shift_x[index], 0.885, metric, transform=ax.transAxes, fontsize=11, color=ink,
                fontfamily="DejaVu Sans Mono", fontweight="bold", va="center")
        ax.text(shift_x[index], 0.852, "BASE → H2 · ΔDIST", transform=ax.transAxes, fontsize=8.5,
                color=secondary, fontfamily="DejaVu Sans Mono", va="center")
        ax.text(recovery_x[index], 0.875, "R ↑", transform=ax.transAxes, fontsize=10.5,
                color=secondary, fontfamily="DejaVu Sans Mono", fontweight="bold", va="center")
    ax.plot([left, right], [0.82, 0.82], color=rule, linewidth=0.9, transform=ax.transAxes)

    top, row_height = 0.80, 0.125
    specs = (
        ("mmd_test_base", "mmd_test_ablated", "mmd_test_recovery", 4),
        ("l2_1_base", "l2_1_ablated", "r_l2_1", 5),
        ("l2_2_base", "l2_2_ablated", "r_l2_2", 5),
        ("l2_3_base", "l2_3_ablated", "r_l2_3", 5),
    )
    for row_index, row in enumerate(data.itertuples()):
        y_top = top - row_index * row_height
        y_center = y_top - row_height / 2
        if row_index % 2:
            ax.add_patch(plt.Rectangle(
                (left, y_top - row_height), right - left, row_height,
                transform=ax.transAxes, facecolor=row_fill, edgecolor="none", zorder=0,
            ))
        ax.text(left, y_center, row.short_name.upper(), transform=ax.transAxes, va="center", ha="left",
                fontsize=11, color=secondary, fontfamily="DejaVu Sans Mono", fontweight="bold")
        for metric_index, (base_name, h2_name, recovery_name, digits) in enumerate(specs):
            base, h2, recovery_value = getattr(row, base_name), getattr(row, h2_name), getattr(row, recovery_name)
            change = 100 * (h2 - base) / base
            direction = "CLOSER" if h2 < base else "FARTHER" if h2 > base else "UNCHANGED"
            color = closer if h2 < base else farther if h2 > base else secondary
            ax.text(shift_x[metric_index], y_center + 0.020, f"{base:.{digits}f} → {h2:.{digits}f}",
                    transform=ax.transAxes, va="center", ha="left", fontsize=10.5, color=ink)
            ax.text(shift_x[metric_index], y_center - 0.027, f"{_signed_percent(change)} · {direction}",
                    transform=ax.transAxes, va="center", ha="left", fontsize=8.8, color=color,
                    fontfamily="DejaVu Sans Mono", fontweight="bold")
            recovery_color = closer if recovery_value > 0 else farther if recovery_value < 0 else secondary
            ax.text(recovery_x[metric_index], y_center, _signed_percent(100 * recovery_value),
                    transform=ax.transAxes, va="center", ha="left", fontsize=10.5,
                    color=recovery_color, fontweight="bold")
        ax.plot([left, right], [y_top - row_height, y_top - row_height], color="#C8CDD3",
                linewidth=0.6, transform=ax.transAxes)

    foot_y = top - len(data) * row_height - 0.035
    ax.text(left, foot_y,
            "ΔDIST = 100 × (H2 − BASELINE) / BASELINE: NEGATIVE = CLOSER / BETTER.  "
            "R = (BASELINE − H2) / (BASELINE − HUMAN FLOOR): POSITIVE = GAP CLOSED / BETTER.",
            transform=ax.transAxes, color=secondary, fontsize=9.1, fontfamily="DejaVu Sans Mono", va="top")
    ax.text(left, foot_y - 0.045,
            "RAW CHANGE AND R DIFFER BECAUSE R NORMALIZES BY THE NONZERO HUMAN-HUMAN FLOOR.",
            transform=ax.transAxes, color=secondary, fontsize=9.1, fontfamily="DejaVu Sans Mono", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=background)
    plt.close(fig)


def annotate_heatmap(ax, matrix, xlabels, ylabels, title, norm, cmap, suffix="%", signed=True):
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(length=0, labelsize=10)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            number = f"{value:+.1f}" if signed else f"{value:.1f}"
            ax.text(column, row, f"{number}{suffix}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color="#111111" if luminance > 0.58 else "white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def write_figure(data, png_path, svg_path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#D0D4D8"})
    fig = plt.figure(figsize=(16, 10), facecolor="white", layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.2, 1])
    names = data["short_name"].tolist()
    cmap = LinearSegmentedColormap.from_list("movement", ["#C65D21", "#F6F3EE", "#176B87"])

    movement_columns = [
        ("r_l2_1", "R L2-1"), ("r_l2_2", "R L2-2"),
        ("r_l2_3", "R L2-3"), ("mmd_test_recovery", "R MMD"),
        ("l2_1_reduction_pct", "Δ L2-1"), ("l2_2_reduction_pct", "Δ L2-2"),
        ("l2_3_reduction_pct", "Δ L2-3"), ("jsd_reduction_pct", "Δ JSD"),
    ]
    movement = np.column_stack([
        data[column].to_numpy() * (100 if column.startswith("r_") or column.endswith("_recovery") else 1)
        for column, _ in movement_columns
    ])
    limit = max(25, float(np.ceil(np.abs(movement).max() / 5) * 5))
    ax0 = fig.add_subplot(grid[0, :])
    annotate_heatmap(
        ax0, movement, [label for _, label in movement_columns], names,
        "A. Distributional movement on held-out TEST", TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), cmap,
    )
    ax0.text(0, -0.18, "Positive = closer to the human corpus. R uses frozen VAL2 floors; MMD bandwidth is selected on VAL2 only.",
             transform=ax0.transAxes, fontsize=9.5, color="#555B61")

    ax1 = fig.add_subplot(grid[1, 0])
    y = np.arange(len(data))
    difference = data["jmq_difference_pp"].to_numpy()
    low = data["jmq_difference_ci_low_pp"].to_numpy()
    high = data["jmq_difference_ci_high_pp"].to_numpy()
    colors = np.where(difference >= 0, "#176B87", "#C65D21")
    ax1.barh(y, difference, color=colors, height=0.58)
    ax1.errorbar(difference, y, xerr=np.vstack([difference - low, high - difference]),
                 fmt="none", ecolor="#222222", capsize=3, linewidth=1.2)
    ax1.axvline(0, color="#70757A", linewidth=1)
    ax1.set_yticks(y, names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Ablated minus baseline preference (percentage points)")
    ax1.set_title("B. Refusal-filtered JMQ", loc="left", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(axis="x", color="#E4E7EA", linewidth=0.8)
    ax1.set_axisbelow(True)
    for index, value in enumerate(difference):
        ax1.text(value - 0.8 if value < 0 else value + 0.8, index, f"{value:+.1f}",
                  va="center", ha="right" if value < 0 else "left", fontsize=9, fontweight="bold")
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    ax2 = fig.add_subplot(grid[1, 1])
    preservation = data[["ifeval_retention_pct", "capability_retention_pct", "harm_refusal_retention_pct"]].to_numpy()
    retention_cmap = LinearSegmentedColormap.from_list("retention", ["#C65D21", "#F6F3EE", "#176B87"])
    annotate_heatmap(
        ax2, preservation, ["IFEval", "Capability", "Harm refusal"], names,
        "C. Preservation (% of baseline)", TwoSlopeNorm(vmin=60, vcenter=100, vmax=110), retention_cmap,
        signed=False,
    )
    ax2.text(0, -0.18, "Refusal retention is reported but exempt from the frozen pass rule.",
             transform=ax2.transAxes, fontsize=9.5, color="#555B61")

    fig.suptitle("H2 moves distributions but does not improve judged quality", fontsize=18,
                  fontweight="bold", x=0.01, y=0.985, ha="left")
    fig.text(0.01, 0.945, "Five frozen models · TEST n=2,000/model · JMQ direct n≤200/model · raw base→ablated scores in companion table · controls pending",
             fontsize=10.5, color="#555B61", ha="left")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    data = load_rows(args.root)
    metrics_path = args.root / "metrics/combined_test_jmq.parquet"
    figure_data_path = args.root / "figures/data/combined_test_jmq.parquet"
    table_path = args.root / "metrics/tables/combined_test_jmq.md"
    paired_table_path = args.root / "metrics/tables/paired_mmd_jmq_l2.md"
    grouped_table_path = args.root / "metrics/tables/grouped_distribution_shift.md"
    png_path = args.root / "figures/combined_test_jmq.png"
    svg_path = args.root / "figures/combined_test_jmq.svg"
    paired_png_path = args.root / "figures/paired_mmd_jmq_l2.png"
    grouped_png_path = args.root / "figures/grouped_distribution_shift.png"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    figure_data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(metrics_path, index=False)
    data.to_parquet(figure_data_path, index=False)
    write_table(data, table_path)
    write_paired_table(data, paired_table_path)
    write_grouped_distribution_table(data, grouped_table_path)
    write_figure(data, png_path, svg_path)
    write_paired_figure(data, paired_png_path)
    write_grouped_distribution_figure(data, grouped_png_path)
    print(f"WROTE {len(data)} combined rows -> {metrics_path}, {table_path}, {paired_table_path}, "
          f"{grouped_table_path}, {png_path}, {svg_path}, {paired_png_path}, {grouped_png_path}")


if __name__ == "__main__":
    main()
