"""Score the frozen H2-MRSC-400 controls and seal completion hashes.

Computes corpus L2-1/2/3, JSD, pinned-embedding RBF MMD, human-gap recovery,
10,000 paired prompt-attribution bootstraps, and Holm corrections across the
five frozen models.  Core MMD artifacts are read-only and hash-checked.

Usage:
  python src/control_stats.py
  python src/control_stats.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from control_embed import EMBED_SCHEMA, EMBEDDER, MAX_LENGTH, POOLING, REVISION
from h2_controls import (
    ARCHIVED_STRATUM,
    CURRENT_STRATUM,
    EPOCH_FINGERPRINT,
    EPOCH_ID,
    EVAL_MODEL,
    EVAL_REVISION,
    FROZEN_RUNS,
    MODELS,
    SUITE,
    atomic_json,
    atomic_parquet,
    canonical_json,
    eval_cache_dir,
    network_allowed,
    read_json,
    sha256_bytes,
    sha256_file,
)

STATS_SCHEMA = "h2-mrsc-400.stats.v5"
BOOTSTRAP_METHOD = "paired prompt-attribution bootstrap; point estimator exactly recovered by mean attribution"
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 999
METRICS = ("L2-1", "L2-2", "L2-3", "JSD", "MMD")
SPLIT_ARMS = {
    "test_jmq": ("ANCHOR-BASE", "ANCHOR-CONE", "RAND-RANK-DOSE", "LEX-MATCH"),
    "val2": ("VAL2-BASE", "VAL2-CONE", "FULL-SYM"),
}
BASELINE_ARM = {"test_jmq": "ANCHOR-BASE", "val2": "VAL2-BASE"}
CONE_ARM = {"test_jmq": "ANCHOR-CONE", "val2": "VAL2-CONE"}


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def _ngram_counter(ids: list[int], order: int) -> Counter:
    if len(ids) < order:
        return Counter()
    return Counter(tuple(ids[index:index + order]) for index in range(len(ids) - order + 1))


def _tokenize(tokenizer, texts: list[str]) -> list[list[int]]:
    return [list(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]


def l2_attributions(left_ids: list[list[int]], right_ids: list[list[int]], order: int) -> tuple[float, np.ndarray]:
    """Return corpus L2 and paired prompt attributions whose mean is L2 squared."""
    if len(left_ids) != len(right_ids) or not left_ids:
        raise ValueError("L2 requires equal non-empty paired corpora")
    left = [_ngram_counter(ids, order) for ids in left_ids]
    right = [_ngram_counter(ids, order) for ids in right_ids]
    left_total = sum(sum(counter.values()) for counter in left)
    right_total = sum(sum(counter.values()) for counter in right)
    if left_total <= 0 or right_total <= 0:
        raise ValueError(f"L2-{order} received an empty n-gram corpus")
    pooled_left, pooled_right = sum(left, Counter()), sum(right, Counter())
    keys = pooled_left.keys() | pooled_right.keys()
    difference = {key: pooled_left.get(key, 0) / left_total - pooled_right.get(key, 0) / right_total
                  for key in keys}
    squared = sum(value * value for value in difference.values())
    n = len(left)
    contributions = np.empty(n, dtype=np.float64)
    for index, (left_doc, right_doc) in enumerate(zip(left, right)):
        contributions[index] = sum(
            difference[key] * (n * left_doc.get(key, 0) / left_total
                               - n * right_doc.get(key, 0) / right_total)
            for key in left_doc.keys() | right_doc.keys()
        )
    if not np.isclose(contributions.mean(), squared, atol=1e-12, rtol=1e-10):
        raise AssertionError("L2 attribution does not recover the point estimator")
    return math.sqrt(max(squared, 0.0)), contributions


def jsd_attributions(left_ids: list[list[int]], right_ids: list[list[int]]) -> tuple[float, np.ndarray]:
    """Return corpus JSD and non-negative paired prompt attributions."""
    if len(left_ids) != len(right_ids) or not left_ids:
        raise ValueError("JSD requires equal non-empty paired corpora")
    left = [Counter(ids) for ids in left_ids]
    right = [Counter(ids) for ids in right_ids]
    left_total = sum(sum(counter.values()) for counter in left)
    right_total = sum(sum(counter.values()) for counter in right)
    if left_total <= 0 or right_total <= 0:
        raise ValueError("JSD received an empty token corpus")
    pooled_left, pooled_right = sum(left, Counter()), sum(right, Counter())
    terms = {}
    for key in pooled_left.keys() | pooled_right.keys():
        p, q = pooled_left.get(key, 0) / left_total, pooled_right.get(key, 0) / right_total
        midpoint = 0.5 * (p + q)
        value = 0.0
        if p:
            value += 0.5 * p * math.log((p + 1e-8) / (midpoint + 1e-8))
        if q:
            value += 0.5 * q * math.log((q + 1e-8) / (midpoint + 1e-8))
        terms[key] = value
    point = sum(terms.values())
    n = len(left)
    contributions = np.zeros(n, dtype=np.float64)
    for index, (left_doc, right_doc) in enumerate(zip(left, right)):
        for key in left_doc.keys() | right_doc.keys():
            p, q = pooled_left.get(key, 0) / left_total, pooled_right.get(key, 0) / right_total
            available = int(p > 0) + int(q > 0)
            allocation = 0.0
            if p:
                allocation += n * left_doc.get(key, 0) / left_total / p
            if q:
                allocation += n * right_doc.get(key, 0) / right_total / q
            contributions[index] += terms[key] * allocation / available
    if not np.isclose(contributions.mean(), point, atol=1e-12, rtol=1e-10):
        raise AssertionError("JSD attribution does not recover the point estimator")
    return float(point), contributions


def _rbf(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    left2 = np.square(left).sum(axis=1, dtype=np.float64)[:, None]
    right2 = np.square(right).sum(axis=1, dtype=np.float64)[None, :]
    squared = np.maximum(left2 + right2 - 2.0 * (left @ right.T), 0.0)
    return np.exp(-squared / (2.0 * bandwidth * bandwidth))


def mmd_attributions(left: np.ndarray, right: np.ndarray, bandwidth: float) -> tuple[float, np.ndarray, float]:
    """Return biased RBF MMD, exact row attributions, and unbiased MMD2."""
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if (left.ndim != 2 or right.shape != left.shape or len(left) < 2
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("MMD requires equal finite matrices and a positive bandwidth")
    left_kernel = _rbf(left, left, bandwidth)
    right_kernel = _rbf(right, right, bandwidth)
    cross_kernel = _rbf(left, right, bandwidth)
    contributions = (left_kernel.mean(axis=1) + right_kernel.mean(axis=1)
                     - cross_kernel.mean(axis=1) - cross_kernel.mean(axis=0))
    biased2 = float(contributions.mean())
    if biased2 < -1e-10:
        raise ValueError(f"biased MMD2 is numerically invalid: {biased2}")
    n = len(left)
    unbiased2 = ((left_kernel.sum() - n) / (n * (n - 1))
                 + (right_kernel.sum() - n) / (n * (n - 1))
                 - 2.0 * cross_kernel.mean())
    return math.sqrt(max(biased2, 0.0)), contributions.astype(np.float64), float(unbiased2)


def recovery(baseline: float, arm: float, floor: float) -> float:
    denominator = baseline - floor
    if not all(math.isfinite(value) for value in (baseline, arm, floor)) or denominator <= 1e-12:
        raise ValueError(f"invalid recovery inputs baseline={baseline}, arm={arm}, floor={floor}")
    return float((baseline - arm) / denominator)


def _distance_bootstrap(counts: np.ndarray, contributions: np.ndarray, square_root: bool) -> np.ndarray:
    values = counts @ np.asarray(contributions, dtype=np.float64) / counts.shape[1]
    return np.sqrt(np.maximum(values, 0.0)) if square_root else values


def _p_two_sided(samples: np.ndarray) -> float:
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return float("nan")
    tail = min(np.count_nonzero(samples <= 0), np.count_nonzero(samples >= 0))
    return min(1.0, float((1 + 2 * tail) / (len(samples) + 1)))


def _summary(samples: np.ndarray) -> tuple[float, float, float]:
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        raise ValueError("bootstrap produced no finite estimates")
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high), _p_two_sided(samples)


def holm(values: list[float]) -> list[float]:
    output = [float("nan")] * len(values)
    finite = [(index, value) for index, value in enumerate(values) if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        output[index] = running
    return output


def load_eval_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        EVAL_MODEL, revision=EVAL_REVISION, trust_remote_code=True,
        local_files_only=not network_allowed(), token=os.environ.get("HF_TOKEN") or None,
        cache_dir=eval_cache_dir(),
    )


def load_model_inputs(root: Path, model: str) -> dict:
    geom = root / "artifacts" / "geometry" / "H2" / model
    generation_path, embedding_path = geom / "controls_generations.parquet", geom / "control_embeddings.parquet"
    generation_done = read_json(geom / "control_generation_done.json")
    embed_done = read_json(geom / "control_embed_done.json")
    plan = read_json(geom / "controls_plan.json")
    embed_plan = read_json(geom / "control_embed_plan.json")
    claimed_plan_hash = plan.get("plan_sha256")
    unhashed_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if claimed_plan_hash != sha256_bytes(canonical_json(unhashed_plan)):
        raise ValueError(f"{model}: control plan self-hash mismatch")
    claimed_embed_hash = embed_plan.get("embed_plan_sha256")
    unhashed_embed_plan = {key: value for key, value in embed_plan.items() if key != "embed_plan_sha256"}
    if claimed_embed_hash != sha256_bytes(canonical_json(unhashed_embed_plan)):
        raise ValueError(f"{model}: embed plan self-hash mismatch")
    calibration_path = geom / "controls_calibration.json"
    reconstruction_path = geom / "controls_reconstruction_preflight.json"
    if (generation_done.get("generations_sha256") != sha256_file(generation_path)
            or not calibration_path.is_file()
            or generation_done.get("calibration_sha256") != sha256_file(calibration_path)
            or not reconstruction_path.is_file()
            or generation_done.get("reconstruction_preflight_sha256") != sha256_file(reconstruction_path)
            or embed_done.get("embeddings_sha256") != sha256_file(embedding_path)
            or embed_done.get("generation_sha256") != generation_done["generations_sha256"]
            or embed_done.get("plan_sha256") != plan.get("plan_sha256")
            or embed_done.get("source_sha256") != plan.get("source_sha256")
            or embed_done.get("embed_plan_sha256") != claimed_embed_hash):
        raise ValueError(f"{model}: generation/embed plan or completion hash mismatch")
    generation, embeddings = pd.read_parquet(generation_path), pd.read_parquet(embedding_path)
    _require_columns(generation, {"suite", "epoch_id", "epoch_fingerprint", "runtime_stratum", "split", "arm", "prompt_id", "position", "text", "text_sha256"}, f"{model} generations")
    if (set(generation["suite"].astype(str)) != {SUITE}
            or set(generation["epoch_id"].astype(str)) != {EPOCH_ID}
            or set(generation["epoch_fingerprint"].astype(str)) != {EPOCH_FINGERPRINT}):
        raise ValueError(f"{model}: generation epoch mismatch")
    if (set(generation.loc[generation["split"] == "test_jmq", "runtime_stratum"].astype(str)) != {CURRENT_STRATUM}
            or set(generation.loc[generation["split"] == "val2", "runtime_stratum"].astype(str)) != {ARCHIVED_STRATUM}):
        raise ValueError(f"{model}: generation runtime strata mismatch")
    _require_columns(embeddings, {
        "schema_version", "split", "arm", "prompt_id", "position", "text_sha256", "embedding",
        "embedder", "revision", "max_length", "pooling", "embed_plan_sha256",
    }, f"{model} embeddings")
    generation = generation.assign(prompt_id=generation["prompt_id"].astype(str))
    embeddings = embeddings.assign(prompt_id=embeddings["prompt_id"].astype(str))
    generation_keys = {(row.split, row.arm, row.prompt_id): row.text_sha256 for row in generation.itertuples()}
    embedding_keys = {(row.split, row.arm, row.prompt_id): row.text_sha256 for row in embeddings.itertuples()}
    if generation_keys != embedding_keys or len(generation_keys) != len(generation) or len(embedding_keys) != len(embeddings):
        raise ValueError(f"{model}: generation/embedding keys or text hashes differ")
    if (set(embeddings["schema_version"].astype(str)) != {EMBED_SCHEMA}
            or set(embeddings["embedder"].astype(str)) != {EMBEDDER}
            or set(embeddings["revision"].astype(str)) != {REVISION}
            or set(embeddings["max_length"].astype(int)) != {MAX_LENGTH}
            or set(embeddings["pooling"].astype(str)) != {POOLING}
            or set(embeddings["embed_plan_sha256"].astype(str)) != {embed_done["embed_plan_sha256"]}):
        raise ValueError(f"{model}: embedding metadata mismatch")

    mmd_metrics_path = geom / "mmd_metrics.json"
    mmd_done_path = geom / "mmd_done.json"
    mmd_metrics, mmd_done = read_json(mmd_metrics_path), read_json(mmd_done_path)
    if (mmd_done.get("metrics_sha256") != sha256_file(mmd_metrics_path)
            or mmd_metrics.get("model") != model or mmd_metrics.get("frozen") != FROZEN_RUNS[model]
            or mmd_metrics.get("embedder") != EMBEDDER or mmd_metrics.get("revision") != REVISION
            or set(mmd_metrics.get("splits", {})) != {"val2", "test"}):
        raise ValueError(f"{model}: frozen MMD metrics are incomplete or stale")
    core_hashes = embed_done.get("core_hashes")
    if not isinstance(core_hashes, dict):
        raise TypeError(f"{model}: embed completion marker lacks core hashes")
    for relative, digest in core_hashes.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{model}: frozen core file changed: {relative}")
    return {
        "geom": geom, "generation": generation, "embeddings": embeddings,
        "generation_path": generation_path, "embedding_path": embedding_path,
        "generation_done": generation_done, "embed_done": embed_done, "plan": plan,
        "mmd_metrics": mmd_metrics, "core_hashes": core_hashes,
    }


def _ordered(frame: pd.DataFrame, split: str, arm: str) -> pd.DataFrame:
    selected = frame[(frame["split"] == split) & (frame["arm"] == arm)].copy()
    if selected.empty or selected["prompt_id"].duplicated().any():
        raise ValueError(f"missing or duplicate rows for {split}/{arm}")
    return selected.sort_values("position").reset_index(drop=True)


def _vectors(frame: pd.DataFrame, split: str, arm: str) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float32) for value in _ordered(frame, split, arm)["embedding"]])


def _human_floor_order(ids: list[str], prefix: str) -> list[int]:
    return sorted(range(len(ids)), key=lambda index: hashlib.sha256((prefix + ids[index]).encode("utf-8")).digest())


def _floor_metrics(root: Path, inputs: dict, split: str, human_ids: list[list[int]],
                   human_vectors: np.ndarray, ids: list[str], tokenizer) -> dict[str, float]:
    if split == "val2":
        path = root / "artifacts" / "metrics" / "val2_floor.json"
        floors = read_json(path)
        if floors.get("n") != 512:
            raise ValueError("frozen VAL2 token floor is missing or stale")
        mmd_floor = inputs["mmd_metrics"]["splits"]["val2"]["human_vs_human_floor"]["mmd"]
        return {
            "L2-1": float(floors["L2_1gram_hh"]), "L2-2": float(floors["L2_2gram_hh"]),
            "L2-3": float(floors["L2_3gram_hh"]), "JSD": float(floors["JSD_hh"]),
            "MMD": float(mmd_floor),
        }
    order = _human_floor_order(ids, "H2-MRSC-human-floor:")
    midpoint = len(order) // 2
    left_ids = [human_ids[index] for index in order[:midpoint]]
    right_ids = [human_ids[index] for index in order[midpoint:]]
    floors = {f"L2-{n}": l2_attributions(left_ids, right_ids, n)[0] for n in (1, 2, 3)}
    floors["JSD"] = jsd_attributions(left_ids, right_ids)[0]
    mmd_order = _human_floor_order(ids, "mmd-human-floor:")
    left = human_vectors[mmd_order[:midpoint]]
    right = human_vectors[mmd_order[midpoint:]]
    floors["MMD"] = mmd_attributions(left, right, inputs["mmd_metrics"]["bandwidth"])[0]
    return floors


def _bootstrap_seed(model: str, split: str) -> int:
    value = int(hashlib.sha256(f"H2-MRSC|bootstrap|{model}|{split}".encode()).hexdigest()[:8], 16)
    return (BOOTSTRAP_SEED + value) % (2**31 - 1)


def score_model(root: Path, model: str, tokenizer, bootstraps: int) -> tuple[list[dict], dict]:
    inputs = load_model_inputs(root, model)
    generation, embedding = inputs["generation"], inputs["embeddings"]
    all_rows = []
    for split, arms in SPLIT_ARMS.items():
        human = _ordered(generation, split, "HUMAN")
        ids = human["prompt_id"].tolist()
        texts = {arm: _ordered(generation, split, arm)["text"].tolist() for arm in arms}
        for arm in arms:
            selected_ids = _ordered(generation, split, arm)["prompt_id"].tolist()
            if selected_ids != ids:
                raise ValueError(f"{model} {split}/{arm}: prompt order differs from HUMAN")
        human_ids = _tokenize(tokenizer, human["text"].tolist())
        token_ids = {arm: _tokenize(tokenizer, values) for arm, values in texts.items()}
        human_vectors = _vectors(embedding, split, "HUMAN")
        vectors = {arm: _vectors(embedding, split, arm) for arm in arms}
        bandwidth = float(inputs["mmd_metrics"]["bandwidth"])
        floors = _floor_metrics(root, inputs, split, human_ids, human_vectors, ids, tokenizer)

        points: dict[str, dict[str, float]] = {arm: {} for arm in arms}
        contributions: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in arms}
        unbiased_mmd = {}
        for arm in arms:
            for order in (1, 2, 3):
                point, attribution = l2_attributions(token_ids[arm], human_ids, order)
                points[arm][f"L2-{order}"], contributions[arm][f"L2-{order}"] = point, attribution
            point, attribution = jsd_attributions(token_ids[arm], human_ids)
            points[arm]["JSD"], contributions[arm]["JSD"] = point, attribution
            point, attribution, unbiased = mmd_attributions(vectors[arm], human_vectors, bandwidth)
            points[arm]["MMD"], contributions[arm]["MMD"] = point, attribution
            unbiased_mmd[arm] = unbiased

        n = len(ids)
        rng = np.random.default_rng(_bootstrap_seed(model, split))
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=bootstraps).astype(np.float64)
        samples: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in arms}
        for arm in arms:
            for metric in METRICS:
                samples[arm][metric] = _distance_bootstrap(
                    counts, contributions[arm][metric], square_root=metric.startswith("L2") or metric == "MMD"
                )
        baseline_arm, cone_arm = BASELINE_ARM[split], CONE_ARM[split]
        for arm in arms:
            for metric in METRICS:
                baseline_point, arm_point, floor = points[baseline_arm][metric], points[arm][metric], floors[metric]
                try:
                    estimate = recovery(baseline_point, arm_point, floor)
                except ValueError:
                    estimate = float("nan")
                try:
                    cone_point_estimate = recovery(baseline_point, points[cone_arm][metric], floor)
                except ValueError:
                    cone_point_estimate = float("nan")
                denominator = samples[baseline_arm][metric] - floor
                valid = denominator > 1e-12
                recovery_samples = np.full(bootstraps, np.nan)
                recovery_samples[valid] = ((samples[baseline_arm][metric][valid] - samples[arm][metric][valid])
                                           / denominator[valid])
                finite_frac = float(np.count_nonzero(np.isfinite(recovery_samples))) / bootstraps
                if finite_frac < 0.99 or not math.isfinite(estimate):
                    # v5-patch1: baseline already at the human floor, so the
                    # recovery ratio is ill-conditioned (division by ~zero).
                    # Record flagged NaNs instead of aborting the suite; raw
                    # distances stay in the row and the block remains visible.
                    print(
                        f"[{model}] {split}/{arm}/{metric}: degenerate recovery denominator "
                        f"(point denom={baseline_point - floor:.3g}, "
                        f"finite bootstrap frac={finite_frac:.4f}); recording flagged NaNs",
                        flush=True,
                    )
                    all_rows.append({
                        "schema_version": STATS_SCHEMA, "suite": SUITE, "model": model,
                        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
                        "runtime_stratum": CURRENT_STRATUM if split == "test_jmq" else ARCHIVED_STRATUM,
                        "split": split, "n": n, "arm": arm, "metric": metric,
                        "distance": arm_point, "baseline_distance": baseline_point, "human_floor": floor,
                        "recovery": float("nan"), "recovery_ci_low": float("nan"),
                        "recovery_ci_high": float("nan"),
                        "p_recovery": float("nan"), "delta_vs_cone": float("nan"),
                        "delta_vs_cone_ci_low": float("nan"), "delta_vs_cone_ci_high": float("nan"),
                        "p_vs_cone": float("nan"),
                        "recovery_undefined_reason": "degenerate_denominator",
                        "mmd2_unbiased": unbiased_mmd[arm] if metric == "MMD" else np.nan,
                        "mmd_bandwidth": bandwidth if metric == "MMD" else np.nan,
                        "bootstrap_replicates": bootstraps, "bootstrap_seed": _bootstrap_seed(model, split),
                        "bootstrap_method": BOOTSTRAP_METHOD,
                        "source_sha256": inputs["plan"]["source_sha256"],
                        "plan_sha256": inputs["plan"]["plan_sha256"],
                        "generation_sha256": inputs["generation_done"]["generations_sha256"],
                        "embeddings_sha256": inputs["embed_done"]["embeddings_sha256"],
                    })
                    continue
                ci_low, ci_high, p_recovery = _summary(recovery_samples)
                cone_denominator = samples[baseline_arm][metric] - floor
                cone_recovery = np.full(bootstraps, np.nan)
                cone_recovery[valid] = ((samples[baseline_arm][metric][valid] - samples[cone_arm][metric][valid])
                                        / cone_denominator[valid])
                delta_samples = recovery_samples - cone_recovery
                delta = estimate - cone_point_estimate
                delta_low, delta_high, p_delta = _summary(delta_samples)
                all_rows.append({
                    "schema_version": STATS_SCHEMA, "suite": SUITE, "model": model,
                    "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
                    "runtime_stratum": CURRENT_STRATUM if split == "test_jmq" else ARCHIVED_STRATUM,
                    "split": split, "n": n, "arm": arm, "metric": metric,
                    "distance": arm_point, "baseline_distance": baseline_point, "human_floor": floor,
                    "recovery": estimate, "recovery_ci_low": ci_low, "recovery_ci_high": ci_high,
                    "p_recovery": p_recovery, "delta_vs_cone": delta,
                    "delta_vs_cone_ci_low": delta_low, "delta_vs_cone_ci_high": delta_high,
                    "p_vs_cone": p_delta, "recovery_undefined_reason": "",
                    "mmd2_unbiased": unbiased_mmd[arm] if metric == "MMD" else np.nan,
                    "mmd_bandwidth": bandwidth if metric == "MMD" else np.nan,
                    "bootstrap_replicates": bootstraps, "bootstrap_seed": _bootstrap_seed(model, split),
                    "bootstrap_method": BOOTSTRAP_METHOD,
                    "source_sha256": inputs["plan"]["source_sha256"],
                    "plan_sha256": inputs["plan"]["plan_sha256"],
                    "generation_sha256": inputs["generation_done"]["generations_sha256"],
                    "embeddings_sha256": inputs["embed_done"]["embeddings_sha256"],
                })
    model_record = {
        "schema_version": STATS_SCHEMA, "suite": SUITE, "model": model,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
        "frozen": FROZEN_RUNS[model], "eval_tokenizer": EVAL_MODEL, "eval_revision": EVAL_REVISION,
        "embedder": EMBEDDER, "embed_revision": REVISION,
        "bootstrap_replicates": bootstraps, "bootstrap_method": BOOTSTRAP_METHOD,
        "source_sha256": inputs["plan"]["source_sha256"], "plan_sha256": inputs["plan"]["plan_sha256"],
        "generation_sha256": inputs["generation_done"]["generations_sha256"],
        "embeddings_sha256": inputs["embed_done"]["embeddings_sha256"],
        "core_hashes": inputs["core_hashes"], "rows": all_rows,
    }
    return all_rows, model_record


def apply_holm(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["p_recovery_holm"] = np.nan
    frame["p_vs_cone_holm"] = np.nan
    for column, output in (("p_recovery", "p_recovery_holm"), ("p_vs_cone", "p_vs_cone_holm")):
        for indices in frame.groupby(["split", "arm", "metric"], sort=True).groups.values():
            positions = list(indices)
            frame.loc[positions, output] = holm(frame.loc[positions, column].astype(float).tolist())
    frame["holm_family"] = frame.apply(
        lambda row: f"{row['split']}/{row['arm']}/{row['metric']}/across-{len(MODELS)}-models", axis=1
    )
    return frame


def _fmt_recovery(value: float) -> str:
    return f"{float(value):+.3f}" if math.isfinite(float(value)) else "n/a*"


def _fmt_p(value: float) -> str:
    return f"{float(value):.3g}" if math.isfinite(float(value)) else "n/a"


def write_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "# H2-MRSC-400-v5 controls (RAND-only; contemporaneous TEST anchors; archived VAL2 secondary)", "",
        ("TEST comparisons use same-runtime contemporaneous ANCHOR-BASE/ANCHOR-CONE with a total-dose matched "
         "RAND arm (PC1 dropped as dose-infeasible; see protocol). VAL2 FULL-SYM remains "
         "archived-original-runtime and is never pooled with v5 TEST. Distances use the pinned Qwen tokenizer "
         "and NVIDIA embedder. Recovery is relative to the split's human-human floor; positive values move "
         "toward human. CIs use 10,000 paired prompt-attribution bootstraps. Holm families are arm x metric "
         "across the five frozen models. *n/a = degenerate recovery denominator (split baseline already at "
         "the human floor, v5-patch1); raw distances are still reported and the block stays visible."), "",
        "| Model | Split | Arm | R L2-1 | R L2-2 | R L2-3 | R JSD | R MMD | Holm p vs cone (MMD) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, split, arm), rows in frame.groupby(["model", "split", "arm"], sort=False):
        values = rows.set_index("metric")
        lines.append(
            f"| {model} | {split.upper()} | {arm} | {_fmt_recovery(values.loc['L2-1', 'recovery'])} | "
            f"{_fmt_recovery(values.loc['L2-2', 'recovery'])} | {_fmt_recovery(values.loc['L2-3', 'recovery'])} | "
            f"{_fmt_recovery(values.loc['JSD', 'recovery'])} | {_fmt_recovery(values.loc['MMD', 'recovery'])} | "
            f"{_fmt_p(values.loc['MMD', 'p_vs_cone_holm'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def run(root: Path, bootstraps: int) -> None:
    if bootstraps != BOOTSTRAPS:
        raise ValueError(f"frozen suite requires exactly {BOOTSTRAPS} bootstrap replicates")
    snapshots = {model: load_model_inputs(root, model)["core_hashes"] for model in MODELS}
    tokenizer = load_eval_tokenizer()
    rows, model_records = [], {}
    for model in MODELS:
        model_rows, record = score_model(root, model, tokenizer, bootstraps)
        rows.extend(model_rows)
        model_records[model] = record
        print(f"[{model}] scored {len(model_rows)} control metric rows", flush=True)
    frame = apply_holm(pd.DataFrame(rows))
    frame = frame.sort_values(["split", "model", "arm", "metric"]).reset_index(drop=True)

    metrics_dir = root / "metrics"
    metrics_path = metrics_dir / "control_metrics.parquet"
    bootstrap_path = metrics_dir / "control_bootstrap.parquet"
    table_path = metrics_dir / "tables" / "controls.md"
    atomic_parquet(frame, metrics_path)
    bootstrap_columns = [
        "model", "split", "arm", "metric", "recovery", "recovery_ci_low", "recovery_ci_high",
        "p_recovery", "p_recovery_holm", "delta_vs_cone", "delta_vs_cone_ci_low",
        "delta_vs_cone_ci_high", "p_vs_cone", "p_vs_cone_holm", "holm_family",
        "recovery_undefined_reason",
        "bootstrap_replicates", "bootstrap_seed", "bootstrap_method", "source_sha256", "plan_sha256",
    ]
    atomic_parquet(frame[bootstrap_columns], bootstrap_path)
    write_table(frame, table_path)

    for model in MODELS:
        current = load_model_inputs(root, model)
        if current["core_hashes"] != snapshots[model]:
            raise ValueError(f"{model}: core MMD files changed during control statistics")
        model_frame = frame[frame["model"] == model]
        record = dict(model_records[model], rows=_json_records(model_frame))
        atomic_json(record, current["geom"] / "control_metrics.json")

    hashes = {
        "metrics/control_metrics.parquet": sha256_file(metrics_path),
        "metrics/control_bootstrap.parquet": sha256_file(bootstrap_path),
        "metrics/tables/controls.md": sha256_file(table_path),
    }
    model_hashes = {}
    for model in MODELS:
        path = root / "artifacts" / "geometry" / "H2" / model / "control_metrics.json"
        model_hashes[model] = sha256_file(path)
    done = {
        "schema_version": "h2-mrsc-400.stats-done.v5", "suite": SUITE,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
        "models": list(MODELS), "bootstrap_replicates": bootstraps,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "eval_tokenizer": EVAL_MODEL, "eval_revision": EVAL_REVISION,
        "embedder": EMBEDDER, "embed_revision": REVISION,
        "rows": len(frame), "hashes": hashes, "model_metrics_sha256": model_hashes,
        "runtime_strata": {"test_jmq": CURRENT_STRATUM, "val2": ARCHIVED_STRATUM},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    stats_done_path = metrics_dir / "control_stats_done.json"
    atomic_json(done, stats_done_path)
    stats_done_hash = sha256_file(stats_done_path)
    for model in MODELS:
        inputs = load_model_inputs(root, model)
        metrics_json_path = inputs["geom"] / "control_metrics.json"
        atomic_json({
            "schema_version": "h2-mrsc-400.done.v5", "suite": SUITE, "model": model,
            "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
            "frozen": FROZEN_RUNS[model], "source_sha256": inputs["plan"]["source_sha256"],
            "plan_sha256": inputs["plan"]["plan_sha256"],
            "generation_sha256": inputs["generation_done"]["generations_sha256"],
            "embeddings_sha256": inputs["embed_done"]["embeddings_sha256"],
            "control_metrics_sha256": sha256_file(metrics_json_path),
            "control_stats_done_sha256": stats_done_hash, "output_hashes": hashes,
            "runtime_strata": {"test_jmq": CURRENT_STRATUM, "val2": ARCHIVED_STRATUM},
            "core_hashes": snapshots[model], "created_at_utc": done["created_at_utc"],
        }, inputs["geom"] / "controls_done.json")
    print(f"Control statistics complete: {len(frame)} rows; stats marker sha256={stats_done_hash}", flush=True)


class _TinyTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
            ids.append(self.vocab[word])
        return {"input_ids": ids}


def self_test() -> None:
    tokenizer = _TinyTokenizer()
    left = _tokenize(tokenizer, ["a b c", "a b", "c c", "a c"])
    right = _tokenize(tokenizer, ["a c", "b c", "a a", "b b"])
    for order in (1, 2):
        point, values = l2_attributions(left, right, order)
        assert np.isclose(values.mean(), point * point)
    jsd, values = jsd_attributions(left, right)
    assert np.isclose(values.mean(), jsd) and jsd >= 0

    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, 4)).astype(np.float32)
    y = rng.normal(size=(6, 4)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    mmd, values, _ = mmd_attributions(x, y, 0.8)
    assert np.isclose(values.mean(), mmd * mmd)
    counts = rng.multinomial(6, np.full(6, 1 / 6), size=100)
    boot = _distance_bootstrap(counts, values, square_root=True)
    assert len(boot) == 100 and np.isfinite(boot).all()
    assert holm([0.01, 0.03, 0.02]) == [0.03, 0.04, 0.04]
    assert recovery(0.8, 0.5, 0.2) == 0.5
    try:
        recovery(0.2, 0.1, 0.2)
        raise AssertionError("invalid recovery denominator was accepted")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory(prefix="control-stats-self-test-") as directory:
        path = Path(directory) / "table.md"
        sample_rows = []
        for metric in METRICS:
            sample_rows.append({
                "model": "X", "split": "test_jmq", "arm": "A", "metric": metric,
                "recovery": 0.1, "p_vs_cone_holm": 0.5,
            })
        write_table(pd.DataFrame(sample_rows), path)
        assert path.is_file() and "H2-MRSC-400" in path.read_text(encoding="utf-8")
    print("Control stats synthetic self-test passed (CPU only)")


def diag(root: Path) -> None:
    """Temporary diagnostic: report TEST JSD baseline-vs-floor bootstrap denominators."""
    tokenizer = load_eval_tokenizer()
    for model in MODELS:
        path = root / "artifacts" / "geometry" / "H2" / model / "controls_generations.parquet"
        if not path.is_file():
            print(f"[{model}] no local generations, skipped", flush=True)
            continue
        generation = pd.read_parquet(path)
        split = "test_jmq"
        base_arm = BASELINE_ARM[split]
        human = _ordered(generation, split, "HUMAN")
        ids = human["prompt_id"].tolist()
        human_ids = _tokenize(tokenizer, human["text"].tolist())
        base_ids = _tokenize(tokenizer, _ordered(generation, split, base_arm)["text"].tolist())
        point, contrib = jsd_attributions(base_ids, human_ids)
        order = _human_floor_order(ids, "H2-MRSC-human-floor:")
        midpoint = len(order) // 2
        floor = jsd_attributions(
            [human_ids[index] for index in order[:midpoint]],
            [human_ids[index] for index in order[midpoint:]],
        )[0]
        n = len(ids)
        rng = np.random.default_rng(_bootstrap_seed(model, split))
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=BOOTSTRAPS).astype(np.float64)
        samples = _distance_bootstrap(counts, np.asarray(contrib, dtype=np.float64), False)
        denom = samples - floor
        valid = denom > 1e-12
        low, median = float(np.quantile(denom, 0.0)), float(np.quantile(denom, 0.5))
        print(
            f"[{model}] test JSD point={point:.6g} floor={floor:.6g} "
            f"pointdenom={point - floor:.3g} validfrac={float(valid.mean()):.4f} "
            f"min={low:.3g} median={median:.3g} n={n}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diag", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.diag:
        diag(args.root)
        return
    run(args.root, args.bootstrap)


if __name__ == "__main__":
    main()
