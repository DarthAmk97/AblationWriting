"""Score frozen H2 embeddings with an RBF maximum mean discrepancy.

Bandwidth is the median off-diagonal Euclidean distance among combined VAL2
human and baseline embeddings. A saved VAL2 bandwidth is immutable for TEST.

Usage: python src/mmd_score.py --models O1 L1 --splits val2 test
       python src/mmd_score.py --self-test
"""
import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mmd_embed import (
    EMBEDDER, EXPECTED_COUNTS, POOLING, REVISION, SOURCES, atomic_write_json,
    atomic_write_parquet, collect_documents, validate_embedding_artifact,
)


def _kernel_sum(x, y, bandwidth, block_size):
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("RBF bandwidth must be finite and positive")
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("kernel inputs must be finite matrices with equal dimensions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("kernel inputs must be finite")
    same = x is y
    total = 0.0
    scale = 2.0 * bandwidth * bandwidth
    for i in range(0, len(x), block_size):
        xb = x[i:i + block_size]
        x2 = np.square(xb).sum(axis=1, dtype=np.float64)[:, None]
        for j in range(i if same else 0, len(y), block_size):
            yb = y[j:j + block_size]
            y2 = np.square(yb).sum(axis=1, dtype=np.float64)[None, :]
            squared = np.maximum(x2 + y2 - 2.0 * (xb @ yb.T), 0.0)
            if same and i == j:
                np.fill_diagonal(squared, 0.0)
            block_sum = np.exp(-squared / scale).sum(dtype=np.float64)
            total += block_sum if not same or i == j else 2.0 * block_sum
    return total


def _mmd2_from_sums(n_x, n_y, sum_xx, sum_yy, sum_xy):
    if n_x < 2 or n_y < 2:
        raise ValueError("unbiased MMD2 requires at least two samples in each group")
    biased = sum_xx / (n_x * n_x) + sum_yy / (n_y * n_y) - 2.0 * sum_xy / (n_x * n_y)
    unbiased = ((sum_xx - n_x) / (n_x * (n_x - 1))
                + (sum_yy - n_y) / (n_y * (n_y - 1))
                - 2.0 * sum_xy / (n_x * n_y))
    if biased < -1e-10:
        raise ValueError(f"biased MMD2 is numerically invalid: {biased}")
    return {"mmd2_biased": float(max(biased, 0.0)), "mmd2_unbiased": float(unbiased)}


def blocked_mmd2(x, y, bandwidth, block_size=128, sum_xx=None, sum_yy=None):
    """Return biased and unbiased MMD2 using bounded kernel blocks."""
    x, y = np.asarray(x), np.asarray(y)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    sum_xx = _kernel_sum(x, x, bandwidth, block_size) if sum_xx is None else sum_xx
    sum_yy = _kernel_sum(y, y, bandwidth, block_size) if sum_yy is None else sum_yy
    sum_xy = _kernel_sum(x, y, bandwidth, block_size)
    return _mmd2_from_sums(len(x), len(y), sum_xx, sum_yy, sum_xy)


def median_off_diagonal_euclidean(vectors, block_size=128):
    """Exact median over each unique off-diagonal pair, computed in blocks."""
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(vectors) < 2 or not np.isfinite(vectors).all():
        raise ValueError("bandwidth calibration needs at least two finite vectors")
    distances = []
    for i in range(0, len(vectors), block_size):
        left = vectors[i:i + block_size]
        left2 = np.square(left).sum(axis=1, dtype=np.float64)[:, None]
        for j in range(i, len(vectors), block_size):
            right = vectors[j:j + block_size]
            right2 = np.square(right).sum(axis=1, dtype=np.float64)[None, :]
            squared = np.maximum(left2 + right2 - 2.0 * (left @ right.T), 0.0)
            if i == j:
                squared = squared[np.triu_indices(len(left), k=1)]
            else:
                squared = squared.ravel()
            distances.append(np.sqrt(squared))
    bandwidth = float(np.median(np.concatenate(distances)))
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("VAL2 median bandwidth is zero or non-finite")
    return bandwidth


def recovery(baseline, ablated, floor):
    values = (baseline, ablated, floor)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("MMD recovery inputs must be finite")
    denominator = baseline - floor
    if denominator <= 1e-12:
        raise ValueError("MMD recovery denominator must be positive")
    return float((baseline - ablated) / denominator)


def _source_digest(documents, allowed_sources=SOURCES):
    allowed_sources = set(allowed_sources)
    payload = [
        [doc["prompt_id"], doc["source"], doc["text_sha256"]]
        for doc in documents if doc["source"] in allowed_sources
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _embedding_digest(ids, arrays, sources=SOURCES):
    digest = hashlib.sha256()
    for source in sources:
        for prompt_id, vector in zip(ids, arrays[source]):
            digest.update(prompt_id.encode("utf-8") + b"\0" + source.encode("ascii") + b"\0")
            digest.update(np.asarray(vector, dtype="<f4").tobytes())
    return digest.hexdigest()


def _load_split(root, model, split):
    documents, frozen = collect_documents(root, model, split)
    path = root / "artifacts" / "geometry" / "H2" / model / f"mmd_embeddings_{split}.parquet"
    frame = validate_embedding_artifact(path, documents, model, split, frozen)
    lookup = {
        (str(row.prompt_id), str(row.source)): np.asarray(row.embedding, dtype=np.float32)
        for row in frame.itertuples()
    }
    ids = [doc["prompt_id"] for doc in documents if doc["source"] == "human"]
    arrays = {
        source: np.stack([lookup[(prompt_id, source)] for prompt_id in ids])
        for source in SOURCES
    }
    return documents, frozen, ids, arrays, int(frame["max_length"].iloc[0])


def _load_or_calibrate_bandwidth(root, model, val2, block_size, allow_create):
    documents, frozen, ids, arrays, max_length = val2
    geom = root / "artifacts" / "geometry" / "H2" / model
    path = geom / "mmd_bandwidth.json"
    source_digest = _source_digest(documents, ("human", "baseline"))
    embedding_digest = _embedding_digest(ids, arrays, ("human", "baseline"))
    expected = {
        "model": model,
        "embedder": EMBEDDER,
        "revision": REVISION,
        "frozen": frozen,
        "calibration_split": "val2",
        "calibration_sources": ["human", "baseline"],
        "n_per_source": EXPECTED_COUNTS["val2"],
        "method": "median off-diagonal Euclidean",
        "rbf": "exp(-squared_euclidean / (2 * bandwidth^2))",
        "source_digest": source_digest,
        "embedding_digest": embedding_digest,
        "max_length": max_length,
        "pooling": POOLING,
    }
    if path.is_file():
        with open(path, encoding="utf-8") as stream:
            saved = json.load(stream)
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"{model}: frozen bandwidth metadata mismatch at {key}; refusing recalibration")
        bandwidth = saved.get("bandwidth")
        if not isinstance(bandwidth, (int, float)) or not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError(f"{model}: invalid saved bandwidth")
        return float(bandwidth), saved
    if not allow_create:
        raise ValueError(f"{model}: TEST requires an existing bandwidth calibrated by an explicit VAL2 run")
    bandwidth = median_off_diagonal_euclidean(
        np.concatenate([arrays["human"], arrays["baseline"]]), block_size
    )
    saved = dict(expected, bandwidth=bandwidth)
    atomic_write_json(saved, path)
    return bandwidth, saved


def _with_headline(values):
    return dict(values, mmd=math.sqrt(values["mmd2_biased"]))


def _score_split(ids, arrays, bandwidth, block_size):
    human_sum = _kernel_sum(arrays["human"], arrays["human"], bandwidth, block_size)
    base = _with_headline(blocked_mmd2(
        arrays["baseline"], arrays["human"], bandwidth, block_size, sum_yy=human_sum
    ))
    ablated = _with_headline(blocked_mmd2(
        arrays["ablated"], arrays["human"], bandwidth, block_size, sum_yy=human_sum
    ))
    order = sorted(range(len(ids)), key=lambda index: hashlib.sha256(
        ("mmd-human-floor:" + ids[index]).encode("utf-8")
    ).digest())
    midpoint = len(order) // 2
    left, right = arrays["human"][order[:midpoint]], arrays["human"][order[midpoint:]]
    floor = _with_headline(blocked_mmd2(left, right, bandwidth, block_size))
    return {
        "n": len(ids),
        "baseline_vs_human": base,
        "ablated_vs_human": ablated,
        "human_vs_human_floor": dict(
            floor, n_left=len(left), n_right=len(right),
            assignment="sha256('mmd-human-floor:' + prompt_id), sorted then halved",
        ),
        "recovery_mmd": recovery(base["mmd"], ablated["mmd"], floor["mmd"]),
    }


def _write_table(root):
    rows = []
    geometry = root / "artifacts" / "geometry" / "H2"
    for path in sorted(geometry.glob("*/mmd_metrics.json")) if geometry.exists() else []:
        with open(path, encoding="utf-8") as stream:
            metric = json.load(stream)
        if metric.get("embedder") != EMBEDDER or metric.get("revision") != REVISION:
            continue
        for split in ("val2", "test"):
            value = metric.get("splits", {}).get(split)
            if not value:
                continue
            base = value["baseline_vs_human"]
            ablated = value["ablated_vs_human"]
            floor = value["human_vs_human_floor"]
            rows.append(
                f"| {metric['model']} | {split.upper()} | {metric['bandwidth']:.6f} | "
                f"{base['mmd']:.6f} | {ablated['mmd']:.6f} | {floor['mmd']:.6f} | "
                f"{value['recovery_mmd']:+.3f} | {base['mmd2_unbiased']:.6f} | "
                f"{ablated['mmd2_unbiased']:.6f} |"
            )
    lines = [
        "<!-- RBF bandwidth: VAL2 human+baseline median off-diagonal Euclidean; TEST reuses it. -->",
        "| Model | Split | Bandwidth | Baseline MMD | H2 MMD | Human floor | Recovery | Baseline MMD2u | H2 MMD2u |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
    ]
    path = root / "metrics" / "tables" / "mmd.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    tmp = Path(handle.name)
    try:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        handle.close()
        tmp.replace(path)
    finally:
        if not handle.closed:
            handle.close()
        tmp.unlink(missing_ok=True)


def score_model(root, model, splits, block_size):
    # Always validate VAL2: it is the sole legal source of a TEST bandwidth.
    val2 = _load_split(root, model, "val2")
    bandwidth, bandwidth_record = _load_or_calibrate_bandwidth(
        root, model, val2, block_size, allow_create="val2" in splits
    )
    geom = root / "artifacts" / "geometry" / "H2" / model
    metrics_path = geom / "mmd_metrics.json"
    metrics = {}
    if metrics_path.is_file():
        try:
            with open(metrics_path, encoding="utf-8") as stream:
                candidate = json.load(stream)
            if (candidate.get("embedder") == EMBEDDER and candidate.get("revision") == REVISION
                    and candidate.get("frozen") == val2[1]
                    and candidate.get("bandwidth") == bandwidth
                    and candidate.get("block_size") == block_size):
                for split, result in candidate.get("splits", {}).items():
                    if split not in EXPECTED_COUNTS:
                        continue
                    loaded = val2 if split == "val2" else _load_split(root, model, split)
                    documents, frozen, ids, arrays, max_length = loaded
                    if (frozen == val2[1]
                            and max_length == val2[4]
                            and result.get("n") == EXPECTED_COUNTS[split]
                            and result.get("source_digest") == _source_digest(documents)
                            and result.get("embedding_digest") == _embedding_digest(ids, arrays)):
                        metrics[split] = result
        except (OSError, ValueError, TypeError, AttributeError):
            metrics = {}

    for split in splits:
        loaded = val2 if split == "val2" else _load_split(root, model, split)
        documents, frozen, ids, arrays, max_length = loaded
        if frozen != val2[1] or max_length != val2[4]:
            raise ValueError(f"{model}: {split} embedding configuration differs from VAL2")
        result = _score_split(ids, arrays, bandwidth, block_size)
        result["source_digest"] = _source_digest(documents)
        result["embedding_digest"] = _embedding_digest(ids, arrays)
        metrics[split] = result
        print(
            f"[{model}] {split.upper()} MMD {result['baseline_vs_human']['mmd']:.6f} -> "
            f"{result['ablated_vs_human']['mmd']:.6f}; R={result['recovery_mmd']:+.3f}",
            flush=True,
        )

    output = {
        "model": model,
        "embedder": EMBEDDER,
        "revision": REVISION,
        "frozen": val2[1],
        "bandwidth": bandwidth,
        "bandwidth_source": "VAL2 human+baseline only",
        "kernel": bandwidth_record["rbf"],
        "block_size": block_size,
        "headline": "sqrt(max(biased MMD2, 0))",
        "embedding_max_length": val2[4],
        "pooling": POOLING,
        "splits": metrics,
    }
    atomic_write_json(output, metrics_path)

    done_path = geom / "mmd_done.json"
    complete = all(
        metrics.get(split, {}).get("n") == count for split, count in EXPECTED_COUNTS.items()
    )
    if complete:
        metrics_sha256 = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        atomic_write_json({
            "model": model,
            "embedder": EMBEDDER,
            "revision": REVISION,
            "frozen": val2[1],
            "splits": ["val2", "test"],
            "n": EXPECTED_COUNTS,
            "bandwidth": bandwidth,
            "metrics_sha256": metrics_sha256,
        }, done_path)
    else:
        done_path.unlink(missing_ok=True)


def self_test():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 3))
    y = rng.normal(size=(4, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    bandwidth = 0.7
    scale = 2 * bandwidth * bandwidth
    kxx = np.exp(-np.maximum(
        np.square(x).sum(1)[:, None] + np.square(x).sum(1)[None, :] - 2 * x @ x.T, 0
    ) / scale)
    kyy = np.exp(-np.maximum(
        np.square(y).sum(1)[:, None] + np.square(y).sum(1)[None, :] - 2 * y @ y.T, 0
    ) / scale)
    kxy = np.exp(-np.maximum(
        np.square(x).sum(1)[:, None] + np.square(y).sum(1)[None, :] - 2 * x @ y.T, 0
    ) / scale)
    dense = _mmd2_from_sums(len(x), len(y), kxx.sum(), kyy.sum(), kxy.sum())
    blocked = blocked_mmd2(x, y, bandwidth, block_size=2)
    assert all(abs(dense[key] - blocked[key]) < 1e-12 for key in dense)
    combined = np.concatenate([x, y])
    dense_distances = np.sqrt(np.maximum(
        np.square(combined).sum(1)[:, None] + np.square(combined).sum(1)[None, :]
        - 2 * combined @ combined.T, 0
    ))[np.triu_indices(len(combined), 1)]
    assert abs(median_off_diagonal_euclidean(combined, 2) - np.median(dense_distances)) < 1e-12
    assert recovery(0.8, 0.5, 0.2) == 0.5
    try:
        recovery(0.2, 0.1, 0.2)
        raise AssertionError("invalid recovery denominator was accepted")
    except ValueError:
        pass

    ids = [f"p{i}" for i in range(6)]
    human = rng.normal(size=(6, 3))
    human /= np.linalg.norm(human, axis=1, keepdims=True)
    arrays = {
        "human": human,
        "baseline": np.tile(np.array([[1.0, 0.0, 0.0]]), (6, 1)),
        "ablated": human,
    }
    first = _score_split(ids, arrays, bandwidth, 2)["human_vs_human_floor"]
    permutation = [4, 1, 5, 0, 3, 2]
    shuffled = {source: values[permutation] for source, values in arrays.items()}
    second = _score_split([ids[index] for index in permutation], shuffled, bandwidth, 2)["human_vs_human_floor"]
    assert first == second

    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix=".mmd-self-test-", dir=project_root) as directory:
        root = Path(directory)
        frozen = "H2-full-s3-X-config-cone"
        bandwidth_documents = [
            {
                "prompt_id": prompt_id, "source": source,
                "text_sha256": hashlib.sha256(f"{prompt_id}-{source}".encode()).hexdigest(),
            }
            for prompt_id in ids for source in SOURCES
        ]
        val2 = (bandwidth_documents, frozen, ids, arrays, 32)
        try:
            _load_or_calibrate_bandwidth(root, "X", val2, 2, allow_create=False)
            raise AssertionError("TEST created a bandwidth without an explicit VAL2 run")
        except ValueError:
            pass
        first_bandwidth, _ = _load_or_calibrate_bandwidth(root, "X", val2, 2, allow_create=True)
        reused_bandwidth, _ = _load_or_calibrate_bandwidth(root, "X", val2, 2, allow_create=False)
        assert first_bandwidth == reused_bandwidth
        changed = dict(arrays, baseline=-arrays["baseline"])
        try:
            _load_or_calibrate_bandwidth(
                root, "X", (bandwidth_documents, frozen, ids, changed, 32), 2, allow_create=True
            )
            raise AssertionError("changed VAL2 embeddings recalibrated a frozen bandwidth")
        except ValueError:
            pass

        path = root / "artifact.parquet"
        documents = []
        vectors = []
        for index, (prompt_id, source) in enumerate((pid, src) for pid in ids[:2] for src in SOURCES):
            text = f"{prompt_id}-{source}"
            documents.append({
                "prompt_id": prompt_id, "source": source, "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            })
            vector = np.array([index + 1.0, 1.0], dtype=np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        rows = [
            {
                "prompt_id": doc["prompt_id"], "source": doc["source"],
                "text_sha256": doc["text_sha256"], "n_tokens": 2, "truncated": False,
                "embedding": vector, "embedder": EMBEDDER, "revision": REVISION,
                "split": "val2", "frozen": "H2-full-s3-X-config-cone",
                "max_length": 32, "pooling": POOLING,
            }
            for doc, vector in zip(documents, vectors)
        ]
        atomic_write_parquet(pd.DataFrame(rows), path)
        validate_embedding_artifact(
            path, documents, "X", "val2", "H2-full-s3-X-config-cone"
        )
        try:
            validate_embedding_artifact(
                path, documents, "X", "val2", "H2-full-s3-X-config-cone", max_length=64
            )
            raise AssertionError("stale max_length metadata was accepted")
        except ValueError:
            pass
        interrupted = path.with_name(f".{path.name}.interrupted.tmp")
        interrupted.write_bytes(b"partial")
        validate_embedding_artifact(
            path, documents, "X", "val2", "H2-full-s3-X-config-cone"
        )
        atomic_write_parquet(pd.DataFrame(rows[:-1]), path)
        try:
            validate_embedding_artifact(
                path, documents, "X", "val2", "H2-full-s3-X-config-cone"
            )
            raise AssertionError("partial artifact was accepted")
        except ValueError:
            pass
        atomic_write_parquet(pd.DataFrame(rows), path)
        validate_embedding_artifact(
            path, documents, "X", "val2", "H2-full-s3-X-config-cone"
        )
    print("MMD synthetic self-test passed (CPU only)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--splits", nargs="+", type=str.lower, choices=sorted(EXPECTED_COUNTS))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.models or not args.splits:
        parser.error("--models and --splits are required unless --self-test is used")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if len(set(args.models)) != len(args.models) or len(set(args.splits)) != len(args.splits):
        parser.error("models and splits must not contain duplicates")
    for model in args.models:
        score_model(args.root, model, args.splits, args.block_size)
    _write_table(args.root)


if __name__ == "__main__":
    main()
