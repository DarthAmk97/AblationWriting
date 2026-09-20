"""Compute frozen JMQ estimates, bootstrap CIs, and Holm-corrected tests."""
import argparse
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
SEED = 999
N_BOOTSTRAP = 10_000
VALID_WINNERS = {"a", "b", "tie"}
MATCHUPS = {
    "h2-vs-human": ("JMQ_H", 2.0, 1.0, 400),
    "baseline-vs-human": ("JMQ_H_baseline", 2.0, 1.0, 400),
    "h2-vs-baseline": ("JMQ_delta", 1.0, 0.5, 200),
}
MODEL_FAMILY = {
    "L1": "meta-llama/Llama-3.2-1B-Instruct",
    "O1": "allenai/OLMo-2-0425-1B-Instruct",
    "Q08": "Qwen/Qwen3.5-0.8B",
    "Q20": "Qwen/Qwen3.5-2B",
    "G2": "google/gemma-4-E2B-it",
    "G4": "google/gemma-4-E4B-it",
    "LF12": "LiquidAI/LFM2-1.2B",
    "S3": "HuggingFaceTB/SmolLM3-3B",
}
# High-recall refusal opening, manually audited without looking at judge winners.
# Two openings continued into substantive answers and are retained explicitly.
REFUSAL_OPENING = re.compile(
    r"^\s*(?:[#>*_\-]+\s*)*(?:i(?:['’]?m| am) sorry|sorry\b|i cannot\b|"
    r"i can['’]?t\b|i couldn['’]?t\b|i won['’]?t\b|i am unable\b|as an ai\b|unable to\b|cannot comply\b)",
    re.I,
)
REFUSAL_AUDIT_OVERRIDES = {
    "L1-HvB-1571": "opening disclaimer followed by a substantive historical answer",
    "Q08-HvB-1696": "opening disclaimer followed by a substantive biology answer",
}


def bootstrap_ci(scores, scale, rng, n_bootstrap=N_BOOTSTRAP):
    draws = scores[rng.integers(0, len(scores), size=(n_bootstrap, len(scores)))].mean(axis=1) * scale
    return np.quantile(draws, [0.025, 0.975])


def paired_permutation_p(n_a, n_b):
    """Exact label-swap test; ties are unchanged by permutation."""
    return binomtest(n_a, n_a + n_b, 0.5, alternative="two-sided").pvalue if n_a + n_b else 1.0


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def validate(data):
    required = {"pair_id", "dim", "model", "matchup", "winner", "ok"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing JMQ columns: {sorted(missing)}")
    if data["pair_id"].astype(str).duplicated().any():
        raise ValueError("duplicate pair_id in canonical JMQ input")
    winners = data["winner"].astype(str).str.lower()
    invalid = ~data["ok"].fillna(False) | ~winners.isin(VALID_WINNERS)
    if invalid.any():
        raise ValueError(f"JMQ contains {int(invalid.sum())} unsuccessful or unparsed judgments")
    if set(data["dim"].astype(str)) != {"overall"}:
        raise ValueError("canonical JMQ statistics require only dim=overall")
    unknown = set(data["model"].astype(str)) - set(MODEL_FAMILY)
    if unknown:
        raise ValueError(f"models outside frozen eight-model family: {sorted(unknown)}")
    for (model, matchup), group in data.groupby(["model", "matchup"]):
        if matchup not in MATCHUPS:
            raise ValueError(f"unknown matchup: {matchup}")
        expected = MATCHUPS[matchup][3]
        if len(group) != expected:
            raise ValueError(f"{model}/{matchup}: expected {expected}, found {len(group)}")
    for model, group in data.groupby("model"):
        if set(group["matchup"]) != set(MATCHUPS):
            raise ValueError(f"{model}: missing a required matchup")


def build_refusal_audit(data, feed):
    required = {"pair_id", "model", "matchup", "a", "b"}
    missing = required - set(feed.columns)
    if missing:
        raise ValueError(f"missing JMQ feed columns: {sorted(missing)}")
    if feed["pair_id"].astype(str).duplicated().any():
        raise ValueError("duplicate pair_id in JMQ feed")
    if set(feed["pair_id"].astype(str)) != set(data["pair_id"].astype(str)):
        raise ValueError("JMQ feed pair_ids do not exactly match canonical judgments")
    audit = feed[feed["matchup"] == "h2-vs-baseline"][
        ["pair_id", "model", "matchup", "a", "b"]
    ].copy()
    audit["a_refusal_opening"] = audit["a"].map(lambda text: bool(REFUSAL_OPENING.search(str(text)[:300])))
    audit["b_refusal_opening"] = audit["b"].map(lambda text: bool(REFUSAL_OPENING.search(str(text)[:300])))
    audit["audit_override"] = audit["pair_id"].map(REFUSAL_AUDIT_OVERRIDES).fillna("")
    audit["exclude_both_refused"] = (
        audit["a_refusal_opening"] & audit["b_refusal_opening"] & audit["audit_override"].eq("")
    )
    audit["rule"] = "both candidates open with a refusal; two substantive-answer false positives retained"
    return audit.drop(columns=["a", "b"])


def analyze(data, feed, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    validate(data)
    audit = build_refusal_audit(data, feed)
    excluded = set(audit.loc[audit["exclude_both_refused"], "pair_id"].astype(str))
    rng = np.random.default_rng(seed)
    rows = []
    for matchup, (estimand, scale, null, _) in MATCHUPS.items():
        groups = {model: group for model, group in data[data["matchup"] == matchup].groupby("model")}
        family_p = []
        family_rows = []
        for model, full_name in MODEL_FAMILY.items():
            group = groups.get(model)
            if group is None:
                family_p.append(1.0)
                family_rows.append({
                    "model": model, "model_name": full_name, "dim": "overall", "matchup": matchup,
                    "estimand": estimand, "available": False, "estimate": math.nan,
                    "ci_low": math.nan, "ci_high": math.nan, "null": null,
                    "alternative": "two-sided", "p_raw": 1.0, "n_total": 0, "n_valid": 0,
                    "n_a": 0, "n_b": 0, "n_tie": 0, "strict_a_rate": math.nan,
                    "n_total_all": 0, "n_excluded_both_refused": 0,
                    "estimate_all": math.nan, "baseline_score": math.nan,
                    "score_difference": math.nan,
                })
                continue
            all_winner = group["winner"].astype(str).str.lower()
            all_scores = all_winner.map({"a": 1.0, "b": 0.0, "tie": 0.5}).to_numpy(dtype=float)
            n_total_all = len(group)
            if matchup == "h2-vs-baseline":
                group = group[~group["pair_id"].astype(str).isin(excluded)]
            winner = group["winner"].astype(str).str.lower()
            n_a, n_b, n_tie = (int((winner == value).sum()) for value in ("a", "b", "tie"))
            scores = winner.map({"a": 1.0, "b": 0.0, "tie": 0.5}).to_numpy(dtype=float)
            low, high = bootstrap_ci(scores, scale, rng, n_bootstrap)
            p_raw = paired_permutation_p(n_a, n_b)
            estimate = float(scores.mean() * scale)
            baseline_score = 1.0 - estimate if matchup == "h2-vs-baseline" else math.nan
            family_p.append(p_raw)
            family_rows.append({
                "model": model, "model_name": full_name, "dim": "overall", "matchup": matchup,
                "estimand": estimand, "available": True, "estimate": estimate,
                "ci_low": float(low), "ci_high": float(high), "null": null,
                "alternative": "two-sided", "p_raw": float(p_raw), "n_total": len(group),
                "n_valid": len(group), "n_a": n_a, "n_b": n_b, "n_tie": n_tie,
                "strict_a_rate": n_a / len(group),
                "n_total_all": n_total_all,
                "n_excluded_both_refused": n_total_all - len(group),
                "estimate_all": float(all_scores.mean() * scale),
                "baseline_score": baseline_score,
                "score_difference": estimate - baseline_score if matchup == "h2-vs-baseline" else math.nan,
            })
        adjusted = holm(np.asarray(family_p))
        for row, p_holm in zip(family_rows, adjusted):
            row.update({
                "p_holm": float(p_holm), "reject_holm_0_05": bool(p_holm < 0.05),
                "n_bootstrap": n_bootstrap, "bootstrap_seed": seed,
                "tie_value": 0.5, "test": "exact paired label-swap",
                "correction": "Holm", "correction_family": f"overall/{matchup}"
                + ("/exclude-bilateral-refusals" if matchup == "h2-vs-baseline" else ""),
                "correction_m": len(MODEL_FAMILY),
            })
            rows.append(row)
    return pd.DataFrame(rows), audit


def atomic_parquet(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_outputs(stats, audit, root):
    metrics_path = root / "metrics/jmq_bootstrap.parquet"
    figure_path = root / "figures/data/jmq_overall.parquet"
    table_path = root / "metrics/tables/jmq.md"
    audit_path = root / "judgments/jmq_bilateral_refusal_audit.parquet"
    atomic_parquet(stats, metrics_path)
    atomic_parquet(stats[stats["available"]].copy(), figure_path)
    atomic_parquet(audit, audit_path)

    available = stats[stats["available"]].set_index(["model", "matchup"])
    lines = [
        f"<!-- generated {datetime.now(timezone.utc).isoformat()} by src/jmq_stats.py -->",
        "## Ablated versus baseline",
        "",
        "Pairs where both candidates refused are excluded before estimation and inference. Score gives ties half credit; difference is ablated minus baseline.",
        "",
        "| Model | Included | Both refused | Ablated wins | Baseline wins | Ties | Ablated score (95% CI) | Baseline score | Difference | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, full_name in MODEL_FAMILY.items():
        if (model, "h2-vs-baseline") not in available.index:
            continue
        row = available.loc[(model, "h2-vs-baseline")]
        lines.append(
            f"| {full_name} | {int(row.n_total)} | {int(row.n_excluded_both_refused)} | "
            f"{int(row.n_a)} | {int(row.n_b)} | {int(row.n_tie)} | "
            f"{row.estimate:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}] | "
            f"{row.baseline_score:.4f} | {row.score_difference * 100:+.1f}pp | {row.p_holm:.3g} |"
        )
    lines += [
        "",
        "## Human-reference audit",
        "",
        "| Model | H2 vs human JMQ_H (95% CI) | Baseline vs human JMQ_H (95% CI) |",
        "|---|---:|---:|",
    ]
    for model, full_name in MODEL_FAMILY.items():
        if (model, "h2-vs-human") not in available.index:
            continue
        h2 = available.loc[(model, "h2-vs-human")]
        base = available.loc[(model, "baseline-vs-human")]
        lines.append(
            f"| {full_name} | {h2.estimate:.4f} [{h2.ci_low:.4f}, {h2.ci_high:.4f}] | "
            f"{base.estimate:.4f} [{base.ci_low:.4f}, {base.ci_high:.4f}] |"
        )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_path, figure_path, table_path, audit_path


def self_test():
    scores = np.array([1.0, 0.0, 0.5, 1.0])
    first = bootstrap_ci(scores, 2.0, np.random.default_rng(7), 1000)
    second = bootstrap_ci(scores, 2.0, np.random.default_rng(7), 1000)
    assert np.array_equal(first, second)
    assert paired_permutation_p(4, 0) == 0.125
    adjusted = holm(np.array([0.01, 0.03, 0.04, 1.0]))
    assert np.allclose(adjusted, [0.04, 0.09, 0.09, 1.0])
    assert REFUSAL_OPENING.search("I can't assist with that request.")
    assert REFUSAL_OPENING.search("I couldn't find enough information to fulfill that request.")
    assert not REFUSAL_OPENING.search("A citizen may be unable to work due to illness.")
    duplicate = pd.DataFrame({
        "pair_id": ["x", "x"], "dim": ["overall"] * 2, "model": ["L1"] * 2,
        "matchup": ["h2-vs-human"] * 2, "winner": ["a", "b"], "ok": [True, True],
    })
    try:
        validate(duplicate)
        raise AssertionError("duplicate pair_id was accepted")
    except ValueError as error:
        assert "duplicate" in str(error)
    print("JMQ statistics self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "judgments/jmq_overall.parquet")
    parser.add_argument("--feed", type=Path, default=ROOT / "dump_mirror/judging/inputs/test_jmq_pairs.parquet")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    stats, audit = analyze(pd.read_parquet(args.input), pd.read_parquet(args.feed), args.bootstrap, args.seed)
    paths = write_outputs(stats, audit, args.root)
    print(f"WROTE {len(stats)} JMQ rows ({int(stats.available.sum())} available) -> {', '.join(map(str, paths))}")


if __name__ == "__main__":
    main()
