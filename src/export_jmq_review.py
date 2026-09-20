"""Export one reviewable row per direct H2-vs-baseline JMQ pair."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "L1": "meta-llama/Llama-3.2-1B-Instruct",
    "O1": "allenai/OLMo-2-0425-1B-Instruct",
    "Q08": "Qwen/Qwen3.5-0.8B",
    "Q20": "Qwen/Qwen3.5-2B",
    "G2": "google/gemma-4-E2B-it",
}
MODEL_ORDER = {model: index for index, model in enumerate(MODELS)}


def reduction(base, ablated):
    return 100.0 * (base - ablated) / base


def recovery(base, ablated, floor):
    denominator = base - floor
    if denominator <= 0:
        raise ValueError(f"invalid recovery denominator: baseline={base}, floor={floor}")
    return (base - ablated) / denominator


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def load_frozen_runs(root):
    runs = [json.loads(line) for line in (root / "dump_mirror/runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    output = {}
    for model in MODELS:
        geom = root / "dump_mirror/artifacts/geometry/H2" / model
        frozen = json.loads((geom / "frozen.json").read_text())["frozen"]
        output[model] = next(row for row in runs if row.get("run_id") == frozen)
    return output


def maybe_mmd(root, model):
    candidates = [
        root / "artifacts/geometry/H2" / model / "mmd_metrics.json",
        root / "mmd_recovery" / model / "mmd_metrics.json",
        root / "dump_mirror/artifacts/geometry/H2" / model / "mmd_metrics.json",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    return json.loads(path.read_text()) if path else {"status": "pending", "note": "MMD embedding/scoring still running on VAST"}


def model_statistics(root):
    frozen_runs = load_frozen_runs(root)
    floors_path = root / "dump_mirror/artifacts/metrics/val2_floor.json"
    floors = json.loads(floors_path.read_text())
    jmq = pd.read_parquet(root / "metrics/jmq_bootstrap.parquet")
    jmq = jmq[(jmq["available"]) & (jmq["matchup"] == "h2-vs-baseline")].set_index("model")
    output = {}
    for model, model_name in MODELS.items():
        geom = root / "dump_mirror/artifacts/geometry/H2" / model
        val2 = frozen_runs[model]
        test = json.loads((geom / "test_metrics.json").read_text())
        preservation = json.loads((geom / "preservation_full.json").read_text())
        judge = jmq.loc[model]
        mmd = maybe_mmd(root, model)
        if set(mmd.get("splits", {})) != {"val2", "test"}:
            raise ValueError(f"{model}: complete local MMD metrics are required for the review export")
        output[model] = {
            "model": model_name,
            "frozen_config": {
                "run_id": val2["run_id"], "layers": val2["layers"], "rank": val2["rank"],
                "alpha": val2["alpha"], "rho": val2["rho"], "cone": val2["cone"],
            },
            "val2": {
                "n": 512, "r_l2_1": val2["R_L2"],
                "l2_1_base": val2["L2_base"], "l2_1_ablated": val2["L2_abl"],
                "l2_1_reduction_pct": reduction(val2["L2_base"], val2["L2_abl"]),
                "l2_2_base": val2["L2_2_base"], "l2_2_ablated": val2["L2_2_abl"],
                "l2_2_reduction_pct": reduction(val2["L2_2_base"], val2["L2_2_abl"]),
                "l2_3_base": val2["L2_3_base"], "l2_3_ablated": val2["L2_3_abl"],
                "l2_3_reduction_pct": reduction(val2["L2_3_base"], val2["L2_3_abl"]),
                "jsd_base": val2["JSD_base"], "jsd_ablated": val2["JSD_abl"],
                "jsd_reduction_pct": reduction(val2["JSD_base"], val2["JSD_abl"]),
                "length_base": val2["len_base"], "length_ablated": val2["len_abl"],
                "length_human": val2["len_human"], "kappa": val2["kappa"],
            },
            "test": {
                "n": test["n"],
                "recovery_floor_source": str(floors_path.relative_to(root)),
                **{
                    f"r_l2_{order}": recovery(
                        test[f"L2_{order}_base"], test[f"L2_{order}_abl"], floors[f"L2_{order}gram_hh"]
                    )
                    for order in (1, 2, 3)
                },
                **{f"l2_{order}_floor": floors[f"L2_{order}gram_hh"] for order in (1, 2, 3)},
                **{
                    f"l2_{order}_{label}": test[f"L2_{order}_{source}"]
                    for order in (1, 2, 3)
                    for label, source in (("base", "base"), ("ablated", "abl"))
                },
                **{
                    f"l2_{order}_reduction_pct": reduction(test[f"L2_{order}_base"], test[f"L2_{order}_abl"])
                    for order in (1, 2, 3)
                },
                "jsd_base": test["JSD_base"], "jsd_ablated": test["JSD_abl"],
                "jsd_reduction_pct": reduction(test["JSD_base"], test["JSD_abl"]),
                "kappa": test["kappa"], "pathology_base": test["path_base"],
                "pathology_ablated": test["path_abl"],
            },
            "jmq_direct": {
                "included": int(judge.n_total),
                "excluded_both_refused": int(judge.n_excluded_both_refused),
                "ablated_wins": int(judge.n_a), "baseline_wins": int(judge.n_b),
                "ties": int(judge.n_tie), "ablated_score": judge.estimate,
                "baseline_score": judge.baseline_score, "difference_pp": 100 * judge.score_difference,
                "ci_low": judge.ci_low, "ci_high": judge.ci_high, "holm_p": judge.p_holm,
            },
            "preservation": preservation,
            "mmd": mmd,
        }
    return json_ready(output)


def mapped_winner(value, a_label, b_label):
    return {"a": a_label, "b": b_label, "tie": "tie"}[str(value).lower()]


def build_export(root):
    feed = pd.read_parquet(root / "dump_mirror/judging/inputs/test_jmq_pairs.parquet")
    judgments = pd.read_parquet(root / "judgments/jmq_overall.parquet")
    audit = pd.read_parquet(root / "judgments/jmq_bilateral_refusal_audit.parquet").set_index("pair_id")
    stats = model_statistics(root)
    split = {
        str(row["prompt_id"]): row
        for row in (
            json.loads(line)
            for line in (root / "dump_mirror/data/splits/test_jmq.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    feed = feed.copy()
    feed["prompt_id"] = feed["pair_id"].astype(str).str.rsplit("-", n=1).str[-1]
    human = feed[feed["matchup"] == "h2-vs-human"].set_index(["model", "prompt_id"])["b"]
    judge = judgments.set_index("pair_id")
    direct = feed[feed["matchup"] == "h2-vs-baseline"].copy()

    rows = []
    for row in direct.itertuples(index=False):
        judgment = judge.loc[row.pair_id]
        h2_human_id = f"{row.model}-HvH-{row.prompt_id}"
        base_human_id = f"{row.model}-BvH-{row.prompt_id}"
        h2_human = judge.loc[h2_human_id]
        base_human = judge.loc[base_human_id]
        refusal = audit.loc[row.pair_id]
        model_stats = stats[row.model]
        source = split[row.prompt_id]
        winner = mapped_winner(judgment.winner, "ablated", "baseline")
        rows.append({
            "schema_version": "jmq-review-v2",
            "model_code": row.model,
            "model_name": MODELS[row.model],
            "prompt_id": row.prompt_id,
            "pair_id": row.pair_id,
            "domain": source.get("domain", ""),
            "dataset": source.get("dataset", ""),
            "source_hash": source.get("hash", ""),
            "duplicate_cluster_id": source.get("duplicate_cluster_id", ""),
            "prompt": row.prompt,
            "human_output": human.loc[(row.model, row.prompt_id)],
            "baseline_output": row.b,
            "ablated_output": row.a,
            "both_refused_excluded": bool(refusal.exclude_both_refused),
            "refusal_audit_override": refusal.audit_override,
            "jmq_direct_winner": winner,
            "jmq_direct_ablated_score": None if refusal.exclude_both_refused else {"a": 1.0, "b": 0.0, "tie": 0.5}[judgment.winner],
            "jmq_direct_raw": judgment.raw,
            "jmq_direct_flip": bool(judgment.flip),
            "jmq_direct_winner_presented": judgment.winner_presented,
            "jmq_request_id": judgment.request_id,
            "jmq_h2_vs_human_winner": mapped_winner(h2_human.winner, "ablated", "human"),
            "jmq_h2_vs_human_raw": h2_human.raw,
            "jmq_baseline_vs_human_winner": mapped_winner(base_human.winner, "baseline", "human"),
            "jmq_baseline_vs_human_raw": base_human.raw,
            "val2_r_l2_1": model_stats["val2"]["r_l2_1"],
            "test_r_l2_1": model_stats["test"]["r_l2_1"],
            "test_r_l2_2": model_stats["test"]["r_l2_2"],
            "test_r_l2_3": model_stats["test"]["r_l2_3"],
            "test_l2_1_base": model_stats["test"]["l2_1_base"],
            "test_l2_1_ablated": model_stats["test"]["l2_1_ablated"],
            "test_l2_2_base": model_stats["test"]["l2_2_base"],
            "test_l2_2_ablated": model_stats["test"]["l2_2_ablated"],
            "test_l2_3_base": model_stats["test"]["l2_3_base"],
            "test_l2_3_ablated": model_stats["test"]["l2_3_ablated"],
            "test_jsd_base": model_stats["test"]["jsd_base"],
            "test_jsd_ablated": model_stats["test"]["jsd_ablated"],
            "test_mmd_recovery": model_stats["mmd"]["splits"]["test"]["recovery_mmd"],
            "test_mmd_base": model_stats["mmd"]["splits"]["test"]["baseline_vs_human"]["mmd"],
            "test_mmd_ablated": model_stats["mmd"]["splits"]["test"]["ablated_vs_human"]["mmd"],
            "model_stats_json": json.dumps(model_stats, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        })
    output = pd.DataFrame(rows)
    output["_model_order"] = output["model_code"].map(MODEL_ORDER)
    output["_prompt_order"] = pd.to_numeric(output["prompt_id"], errors="coerce")
    return output.sort_values(["_model_order", "_prompt_order", "prompt_id"]).drop(columns=["_model_order", "_prompt_order"]).reset_index(drop=True)


def write_excel(data, summary, path):
    from openpyxl.styles import Alignment, Font, PatternFill

    excel_data = data.copy()
    text_columns = [
        "prompt", "human_output", "baseline_output", "ablated_output", "jmq_direct_raw",
        "jmq_h2_vs_human_raw", "jmq_baseline_vs_human_raw", "model_stats_json",
    ]
    truncated = pd.Series([[] for _ in range(len(excel_data))], index=excel_data.index, dtype=object)
    for column in text_columns:
        values = excel_data[column].fillna("").astype(str)
        mask = values.str.len() > 32_000
        for index in excel_data.index[mask]:
            truncated.at[index].append(column)
        excel_data[column] = values.map(
            lambda value: value if len(value) <= 32_000 else value[:32_000] + "\n[TRUNCATED IN XLSX — use Parquet/CSV/JSONL for full text]"
        )
    excel_data["xlsx_truncated_columns"] = truncated.map(",".join)

    readme = pd.DataFrame({
        "Field": ["Purpose", "Rows", "Winner mapping", "Refusal filter", "Recovery fields", "Nested statistics", "Full-fidelity format"],
        "Description": [
            "Manual review of direct H2-vs-baseline JMQ pairs with text and model-level metrics.",
            "1,000 rows: 200 prompts for each of five frozen models.",
            "jmq_direct_winner is ablated, baseline, or tie after reversing candidate presentation flips.",
            "Rows marked both_refused_excluded are retained for audit but omitted from aggregate direct JMQ statistics.",
            "TEST L2 recovery uses order-matched frozen VAL2 human-human floors; TEST MMD fields come from verified local artifacts.",
            "model_stats_json repeats the complete VAL2, TEST, JMQ, preservation, and current MMD record for that model.",
            "Use Parquet or JSONL if Excel truncates unusually long cells.",
        ],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        excel_data.to_excel(writer, sheet_name="prompt_reviews", index=False)
        summary.to_excel(writer, sheet_name="model_summary", index=False)
        readme.to_excel(writer, sheet_name="README", index=False)
        workbook = writer.book
        for name in ("prompt_reviews", "model_summary", "README"):
            sheet = workbook[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="176B87")
                cell.alignment = Alignment(vertical="center")
        sheet = workbook["prompt_reviews"]
        widths = {
            "schema_version": 16, "model_code": 12, "model_name": 38, "prompt_id": 14,
            "pair_id": 22, "domain": 18, "dataset": 22, "source_hash": 18,
            "duplicate_cluster_id": 22, "prompt": 60, "human_output": 70,
            "baseline_output": 70, "ablated_output": 70, "model_stats_json": 60,
        }
        wrap = {
            "prompt", "human_output", "baseline_output", "ablated_output", "jmq_direct_raw",
            "jmq_h2_vs_human_raw", "jmq_baseline_vs_human_raw", "model_stats_json",
        }
        for cell in sheet[1]:
            sheet.column_dimensions[cell.column_letter].width = widths.get(cell.value, 24)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                header = sheet.cell(1, cell.column).value
                cell.alignment = Alignment(vertical="top", wrap_text=header in wrap)
        workbook["model_summary"].column_dimensions["B"].width = 38
        workbook["README"].column_dimensions["A"].width = 24
        workbook["README"].column_dimensions["B"].width = 110
        for row in workbook["README"].iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)


def write_notebook(path):
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"colab": {"name": "AblationWriting_JMQ_Review.ipynb"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# AblationWriting JMQ prompt review\n", "Upload `jmq_prompt_review.parquet` when prompted. Parquet preserves multiline text and types better than CSV."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "from google.colab import files\n", "uploaded = files.upload()\n", "path = next(iter(uploaded))\n",
                "!pip -q install pyarrow\n", "import pandas as pd, json\n",
                "df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)\n",
                "print(df.shape)\n", "display(df[['model_name','prompt_id','jmq_direct_winner','both_refused_excluded','test_r_l2_1','test_mmd_recovery']].head())\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "from IPython.display import display, Markdown\n",
                "def show_row(index):\n",
                "    r = df.iloc[index]\n",
                "    display(Markdown(f\"## {r.model_name} · prompt {r.prompt_id}\\n**JMQ winner:** {r.jmq_direct_winner}  \\n**Both refused/excluded:** {r.both_refused_excluded}\"))\n",
                "    for label, column in [('Prompt','prompt'),('Human','human_output'),('Baseline','baseline_output'),('Ablated','ablated_output'),('Judge rationale','jmq_direct_raw')]:\n",
                "        display(Markdown(f\"### {label}\\n```text\\n{r[column]}\\n```\"))\n",
                "    display(json.loads(r.model_stats_json))\n",
                "show_row(0)\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "# Examples: filter to significant direct losses, or inspect only non-refusal pairs.\n",
                "clean = df[~df.both_refused_excluded]\n",
                "display(clean.groupby(['model_name','jmq_direct_winner']).size().unstack(fill_value=0))\n",
            ]},
        ],
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_readme(path):
    path.write_text(
        "# JMQ prompt-level review export\n\n"
        "- `jmq_prompt_review.xlsx`: easiest option for Excel; includes prompt reviews, model summary, and README sheets.\n"
        "- `jmq_prompt_review.parquet`: recommended for Colab/Pandas; complete multiline text with compact storage.\n"
        "- `jmq_prompt_review.csv`: Excel-compatible UTF-8 CSV. Multiline text is quoted correctly.\n"
        "- `jmq_prompt_review.jsonl`: one complete JSON object per prompt/model pair.\n"
        "- `AblationWriting_JMQ_Review.ipynb`: upload this notebook to Colab, then upload the Parquet file when prompted.\n\n"
        "The export has 1,000 rows: 200 direct H2-vs-baseline pairs for each of five frozen models. "
        "Rows where both candidates refused remain visible but have `both_refused_excluded=true` and no per-row ablated score. "
        "Schema v2 reports order-corrected TEST L2 recovery and verified TEST MMD fields. "
        "`model_stats_json` contains the complete repeated VAL2, TEST, JMQ, preservation, and MMD record.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.root / "exports"
    output.mkdir(parents=True, exist_ok=True)
    data = build_export(args.root)
    summary = pd.read_parquet(args.root / "metrics/combined_test_jmq.parquet")
    data.to_csv(output / "jmq_prompt_review.csv", index=False, encoding="utf-8-sig")
    data.to_parquet(output / "jmq_prompt_review.parquet", index=False)
    data.to_json(output / "jmq_prompt_review.jsonl", orient="records", lines=True, force_ascii=False)
    write_excel(data, summary, output / "jmq_prompt_review.xlsx")
    write_notebook(output / "AblationWriting_JMQ_Review.ipynb")
    write_readme(output / "README.md")
    print(f"WROTE {len(data)} prompt-review rows to {output}")


if __name__ == "__main__":
    main()
