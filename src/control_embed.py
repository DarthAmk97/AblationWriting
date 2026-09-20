"""Embed H2-MRSC-400-v5 controls without modifying frozen core MMD files.

Only TEST HUMAN reuses a core TEST vector.  All contemporaneous machine arms
are newly embedded.  Archived VAL2 human/base/cone vectors remain reusable in
their separate original-runtime stratum; FULL-SYM is newly embedded.

Usage:
  python src/control_embed.py --models L1 O1 Q08 Q20 G2
  python src/control_embed.py --self-test
"""
from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from h2_controls import (
    ARCHIVED_STRATUM,
    CONTROL_COUNT,
    CURRENT_STRATUM,
    EMBEDDER_PIN,
    EPOCH_FINGERPRINT,
    EPOCH_ID,
    FROZEN_RUNS,
    MODEL_PINS,
    MODELS,
    SCHEMA,
    SUITE,
    TEST_ARMS,
    VAL2_ARMS,
    VAL2_COUNT,
    atomic_json,
    atomic_parquet,
    canonical_json,
    network_allowed,
    read_json,
    sha256_bytes,
    sha256_file,
    text_sha256,
)

EMBED_SCHEMA = "h2-mrsc-400.embeddings.v5"
EMBED_DONE_SCHEMA = "h2-mrsc-400.embed-done.v5"
EMBEDDER, REVISION = EMBEDDER_PIN
POOLING = "attention-mask mean in FP32, then L2 normalize"
MAX_LENGTH = 4096
CHECKPOINT_DOCUMENTS = 16
CORE_MAP = {
    ("test_jmq", "HUMAN"): ("test", "human"),
    ("val2", "HUMAN"): ("val2", "human"),
    ("val2", "VAL2-BASE"): ("val2", "baseline"),
    ("val2", "VAL2-CONE"): ("val2", "ablated"),
}
NEW_ARMS = {"ANCHOR-BASE", "ANCHOR-CONE", "RAND-RANK-DOSE", "LEX-MATCH", "FULL-SYM"}
EMBED_COLUMNS = {
    "schema_version", "suite", "epoch_id", "epoch_fingerprint", "model",
    "runtime_stratum", "runtime_fingerprint_sha256", "split", "arm", "prompt_id", "position",
    "text_sha256", "embedding", "n_tokens", "truncated", "embedder", "revision",
    "max_length", "pooling", "embedding_origin", "source_sha256", "plan_sha256",
    "embed_plan_sha256",
}


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def load_generation(root: Path, model: str) -> tuple[pd.DataFrame, dict, dict, Path]:
    geom = root / "artifacts" / "geometry" / "H2" / model
    path = geom / "controls_generations.parquet"
    marker_path = geom / "control_generation_done.json"
    plan_path = geom / "controls_plan.json"
    if not path.is_file() or not marker_path.is_file() or not plan_path.is_file():
        raise ValueError(f"{model}: completed control generations and plan are required")
    frame = pd.read_parquet(path)
    _require_columns(frame, {
        "schema_version", "suite", "epoch_id", "epoch_fingerprint", "model",
        "runtime_stratum", "runtime_fingerprint_sha256", "split", "arm", "prompt_id", "position",
        "text", "text_sha256", "source_sha256", "plan_sha256",
    }, f"{model} controls")
    marker, plan = read_json(marker_path), read_json(plan_path)
    claimed_plan_hash = plan.get("plan_sha256")
    unhashed_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if claimed_plan_hash != sha256_bytes(canonical_json(unhashed_plan)):
        raise ValueError(f"{model}: control plan self-hash mismatch")
    if (marker.get("schema_version") != "h2-mrsc-400.generation-done.v5"
            or marker.get("epoch_fingerprint") != EPOCH_FINGERPRINT
            or marker.get("equivalence_status") != "pass"
            or marker.get("generations_sha256") != sha256_file(path)
            or marker.get("rows") != len(frame)
            or marker.get("plan_sha256") != plan.get("plan_sha256")
            or marker.get("source_sha256") != plan.get("source_sha256")):
        raise ValueError(f"{model}: generation marker/plan/hash mismatch")
    calibration_path = geom / "controls_calibration.json"
    reconstruction_path = geom / "controls_reconstruction_preflight.json"
    if (not calibration_path.is_file()
            or marker.get("calibration_sha256") != sha256_file(calibration_path)
            or not reconstruction_path.is_file()
            or marker.get("reconstruction_preflight_sha256") != sha256_file(reconstruction_path)):
        raise ValueError(f"{model}: generation calibration/reconstruction hash mismatch")
    if (set(frame["schema_version"].astype(str)) != {SCHEMA}
            or set(frame["suite"].astype(str)) != {SUITE}
            or set(frame["epoch_id"].astype(str)) != {EPOCH_ID}
            or set(frame["epoch_fingerprint"].astype(str)) != {EPOCH_FINGERPRINT}
            or set(frame["model"].astype(str)) != {model}
            or set(frame["plan_sha256"].astype(str)) != {plan["plan_sha256"]}
            or set(frame["source_sha256"].astype(str)) != {plan["source_sha256"]}):
        raise ValueError(f"{model}: generation row metadata mismatch")
    frame = frame.copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    if frame.duplicated(["split", "arm", "prompt_id"]).any():
        raise ValueError(f"{model}: duplicate generation rows")
    expected_counts = {("test_jmq", arm): CONTROL_COUNT for arm in TEST_ARMS}
    expected_counts.update({("val2", arm): VAL2_COUNT for arm in VAL2_ARMS})
    actual_counts = frame.groupby(["split", "arm"]).size().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"{model}: generation arm counts differ from the frozen suite")
    if any(not isinstance(row.text, str) or text_sha256(row.text) != row.text_sha256
           for row in frame.itertuples()):
        raise ValueError(f"{model}: generation text/hash mismatch")
    if (set(frame.loc[frame["split"] == "test_jmq", "runtime_stratum"]) != {CURRENT_STRATUM}
            or set(frame.loc[frame["split"] == "val2", "runtime_stratum"]) != {ARCHIVED_STRATUM}):
        raise ValueError(f"{model}: generation runtime strata mismatch")
    return frame, marker, plan, path


def load_core(root: Path, model: str) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    geom = root / "artifacts" / "geometry" / "H2" / model
    done_path, metrics_path = geom / "mmd_done.json", geom / "mmd_metrics.json"
    done, metrics = read_json(done_path), read_json(metrics_path)
    if (done.get("model") != model or done.get("embedder") != EMBEDDER
            or done.get("revision") != REVISION or done.get("frozen") != FROZEN_RUNS[model]
            or done.get("metrics_sha256") != sha256_file(metrics_path)):
        raise ValueError(f"{model}: frozen MMD completion marker is invalid")
    if (metrics.get("model") != model or metrics.get("embedder") != EMBEDDER
            or metrics.get("revision") != REVISION or metrics.get("frozen") != FROZEN_RUNS[model]
            or metrics.get("embedding_max_length") != MAX_LENGTH or metrics.get("pooling") != POOLING):
        raise ValueError(f"{model}: frozen MMD metadata differs from the control embedding contract")
    frames, hashes = {}, {
        str(done_path.relative_to(root)).replace("\\", "/"): sha256_file(done_path),
        str(metrics_path.relative_to(root)).replace("\\", "/"): sha256_file(metrics_path),
    }
    for split, count in (("test", 2000), ("val2", 512)):
        path = geom / f"mmd_embeddings_{split}.parquet"
        if not path.is_file():
            raise ValueError(f"{model}: missing frozen core embeddings {path}")
        frame = pd.read_parquet(path)
        _require_columns(frame, {
            "prompt_id", "source", "text_sha256", "n_tokens", "truncated", "embedding",
            "embedder", "revision", "split", "frozen", "max_length", "pooling",
        }, f"{model} core {split}")
        if len(frame) != count * 3 or frame.duplicated(["prompt_id", "source"]).any():
            raise ValueError(f"{model}: core {split} embeddings are partial or duplicated")
        expected = {
            "embedder": EMBEDDER, "revision": REVISION, "split": split,
            "frozen": FROZEN_RUNS[model], "max_length": MAX_LENGTH, "pooling": POOLING,
        }
        for column, value in expected.items():
            if set(frame[column].astype(str)) != {str(value)}:
                raise ValueError(f"{model}: core {split} metadata mismatch at {column}")
        frames[split] = frame.assign(prompt_id=frame["prompt_id"].astype(str))
        hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    return frames, hashes


def make_embed_plan(model: str, generation_path: Path, generation_marker: dict,
                    control_plan: dict, core_hashes: dict[str, str]) -> dict:
    value = {
        "schema_version": "h2-mrsc-400.embed-plan.v5", "suite": SUITE,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT, "model": model,
        "model_revision": MODEL_PINS[model][1], "embedder": EMBEDDER, "revision": REVISION,
        "max_length": MAX_LENGTH, "pooling": POOLING, "batch_size": 1, "dtype": "bfloat16",
        "checkpoint_documents": CHECKPOINT_DOCUMENTS,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "generation_sha256": sha256_file(generation_path),
        "source_sha256": generation_marker["source_sha256"],
        "plan_sha256": control_plan["plan_sha256"], "core_hashes": core_hashes,
    }
    value["embed_plan_sha256"] = sha256_bytes(canonical_json(value))
    return value


def copy_core_rows(generation: pd.DataFrame, core: dict[str, pd.DataFrame], embed_plan: dict) -> list[dict]:
    lookups = {
        split: {(row.prompt_id, str(row.source)): row for row in frame.itertuples()}
        for split, frame in core.items()
    }
    rows = []
    for source_row in generation.itertuples():
        mapping = CORE_MAP.get((source_row.split, source_row.arm))
        if mapping is None:
            continue
        core_split, core_source = mapping
        key = (str(source_row.prompt_id), core_source)
        if key not in lookups[core_split]:
            raise ValueError(f"missing core vector for {source_row.split}/{source_row.arm}/{source_row.prompt_id}")
        archived = lookups[core_split][key]
        if archived.text_sha256 != source_row.text_sha256:
            raise ValueError(f"core text hash differs for {source_row.split}/{source_row.arm}/{source_row.prompt_id}")
        rows.append({
            "schema_version": EMBED_SCHEMA, "suite": SUITE,
            "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
            "model": source_row.model, "runtime_stratum": source_row.runtime_stratum,
            "runtime_fingerprint_sha256": source_row.runtime_fingerprint_sha256,
            "split": source_row.split, "arm": source_row.arm, "prompt_id": str(source_row.prompt_id),
            "position": int(source_row.position), "text_sha256": source_row.text_sha256,
            "embedding": np.asarray(archived.embedding, dtype=np.float32),
            "n_tokens": int(archived.n_tokens), "truncated": bool(archived.truncated),
            "embedder": EMBEDDER, "revision": REVISION, "max_length": MAX_LENGTH,
            "pooling": POOLING, "embedding_origin": f"core-copy:{core_split}/{core_source}",
            "source_sha256": source_row.source_sha256, "plan_sha256": source_row.plan_sha256,
            "embed_plan_sha256": embed_plan["embed_plan_sha256"],
        })
    return rows


def _validate_embeddings(frame: pd.DataFrame, generation: pd.DataFrame, embed_plan: dict,
                         complete: bool) -> pd.DataFrame:
    _require_columns(frame, EMBED_COLUMNS, "control embeddings")
    frame = frame.copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    metadata = {
        "schema_version": EMBED_SCHEMA, "suite": SUITE, "epoch_id": EPOCH_ID,
        "epoch_fingerprint": EPOCH_FINGERPRINT, "model": embed_plan["model"],
        "embedder": EMBEDDER, "revision": REVISION, "max_length": MAX_LENGTH,
        "pooling": POOLING, "source_sha256": embed_plan["source_sha256"],
        "plan_sha256": embed_plan["plan_sha256"], "embed_plan_sha256": embed_plan["embed_plan_sha256"],
    }
    for column, value in metadata.items():
        if set(frame[column].astype(str)) != {str(value)}:
            raise ValueError(f"control embeddings: metadata mismatch at {column}")
    if frame.duplicated(["split", "arm", "prompt_id"]).any():
        raise ValueError("control embeddings contain duplicate rows")
    if not pd.api.types.is_bool_dtype(frame["truncated"]):
        raise ValueError("control embeddings: truncated must be boolean")
    if (frame["n_tokens"].astype(int) <= 0).any():
        raise ValueError("control embeddings: token counts must be positive")
    vectors = [np.asarray(value, dtype=np.float32) for value in frame["embedding"]]
    shapes = {value.shape for value in vectors}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 1 or next(iter(shapes))[0] == 0:
        raise ValueError("control embeddings must have one non-empty vector shape")
    matrix = np.stack(vectors)
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2e-4, rtol=0):
        raise ValueError("control embeddings must be finite and L2-normalized")
    expected_hashes = {
        (str(row.split), str(row.arm), str(row.prompt_id)): (
            str(row.text_sha256), str(row.runtime_stratum), str(row.runtime_fingerprint_sha256)
        )
        for row in generation.itertuples()
    }
    actual_hashes = {
        (str(row.split), str(row.arm), str(row.prompt_id)): (
            str(row.text_sha256), str(row.runtime_stratum), str(row.runtime_fingerprint_sha256)
        )
        for row in frame.itertuples()
    }
    if not set(actual_hashes) <= set(expected_hashes) or any(
            expected_hashes[key] != value for key, value in actual_hashes.items()):
        raise ValueError("control embeddings do not match generation text hashes")
    if complete and actual_hashes != expected_hashes:
        raise ValueError("control embedding artifact is partial")
    return frame


def load_embedder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("control embedding requires CUDA")
    local_only = not network_allowed()
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDER, revision=REVISION, trust_remote_code=True, local_files_only=local_only, token=token,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("embedder tokenizer has neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        EMBEDDER, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        trust_remote_code=True, local_files_only=local_only, token=token,
    ).eval()
    floating = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if floating != {torch.bfloat16}:
        raise ValueError(f"embedder is not entirely BF16 (floating dtypes={floating})")
    commit = getattr(model.config, "_commit_hash", None)
    if commit not in (None, REVISION):
        raise ValueError("embedder cache commit does not match the pinned revision")
    return tokenizer, model


def embed_one(text: str, tokenizer, model) -> tuple[np.ndarray, int, bool]:
    import torch

    device = next(model.parameters()).device
    untruncated = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    encoded = tokenizer(
        text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH,
        padding=False, return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in encoded.items()
              if key in {"input_ids", "attention_mask"}}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state.float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        denominator = mask.sum(dim=1)
        if (denominator <= 0).any():
            raise ValueError("embedder produced an empty attention mask")
        pooled = (hidden * mask).sum(dim=1) / denominator
        norm = pooled.norm(p=2, dim=1, keepdim=True)
        if (norm <= 0).any() or not torch.isfinite(pooled).all():
            raise ValueError("embedder produced an invalid vector")
        vector = (pooled / norm)[0].cpu().numpy().astype(np.float32)
    used = int(inputs["attention_mask"].sum().item())
    return vector, len(untruncated), len(untruncated) > used


def _new_row(source_row, vector, n_tokens: int, truncated: bool, embed_plan: dict) -> dict:
    return {
        "schema_version": EMBED_SCHEMA, "suite": SUITE,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
        "model": source_row.model, "runtime_stratum": source_row.runtime_stratum,
        "runtime_fingerprint_sha256": source_row.runtime_fingerprint_sha256,
        "split": source_row.split, "arm": source_row.arm, "prompt_id": str(source_row.prompt_id),
        "position": int(source_row.position), "text_sha256": source_row.text_sha256,
        "embedding": vector, "n_tokens": int(n_tokens), "truncated": bool(truncated),
        "embedder": EMBEDDER, "revision": REVISION, "max_length": MAX_LENGTH,
        "pooling": POOLING, "embedding_origin": "new-control",
        "source_sha256": source_row.source_sha256, "plan_sha256": source_row.plan_sha256,
        "embed_plan_sha256": embed_plan["embed_plan_sha256"],
    }


def run_model(root: Path, model: str, preflight: bool = False) -> None:
    generation, generation_marker, control_plan, generation_path = load_generation(root, model)
    core, core_hashes = load_core(root, model)
    embed_plan = make_embed_plan(model, generation_path, generation_marker, control_plan, core_hashes)
    geom = root / "artifacts" / "geometry" / "H2" / model
    plan_path = geom / "control_embed_plan.json"
    if plan_path.exists() and read_json(plan_path) != embed_plan:
        raise ValueError(f"{model}: existing embed plan differs; refusing resume")
    if not plan_path.exists() and not preflight:
        atomic_json(embed_plan, plan_path)
    core_rows = copy_core_rows(generation, core, embed_plan)
    if len(core_rows) != 400 + 3 * 512:
        raise ValueError(f"{model}: unexpected number of reusable core vectors ({len(core_rows)})")
    print(f"[{model}] embedding preflight passed; reuse={len(core_rows)} new={len(generation) - len(core_rows)}", flush=True)
    if preflight:
        return

    output_path = geom / "control_embeddings.parquet"
    done_path = geom / "control_embed_done.json"
    if output_path.exists() and done_path.exists():
        output = _validate_embeddings(pd.read_parquet(output_path), generation, embed_plan, complete=True)
        done = read_json(done_path)
        if (done.get("embeddings_sha256") != sha256_file(output_path)
                or done.get("embed_plan_sha256") != embed_plan["embed_plan_sha256"]
                or done.get("rows") != len(output)):
            raise ValueError(f"{model}: control embedding completion marker is stale")
        if core_hashes != load_core(root, model)[1]:
            raise ValueError(f"{model}: core MMD files changed while validating controls")
        print(f"[{model}] reuse complete control embeddings ({len(output)} rows)", flush=True)
        return
    if output_path.exists() != done_path.exists():
        raise ValueError(f"{model}: final control embedding artifact/marker is incomplete")

    progress_path = geom / "control_embeddings.progress.parquet"
    progress = pd.read_parquet(progress_path) if progress_path.exists() else pd.DataFrame()
    if len(progress):
        progress = _validate_embeddings(progress, generation, embed_plan, complete=False)
        if not set(progress["arm"]) <= NEW_ARMS:
            raise ValueError(f"{model}: progress contains copied core rows")
    completed = set(zip(progress.get("split", []), progress.get("arm", []), progress.get("prompt_id", [])))
    pending = [row for row in generation.itertuples()
               if row.arm in NEW_ARMS and (row.split, row.arm, str(row.prompt_id)) not in completed]
    tokenizer = embedder = None
    try:
        if pending:
            tokenizer, embedder = load_embedder()
        for start in range(0, len(pending), CHECKPOINT_DOCUMENTS):
            rows = []
            for source_row in pending[start:start + CHECKPOINT_DOCUMENTS]:
                vector, n_tokens, truncated = embed_one(source_row.text, tokenizer, embedder)
                rows.append(_new_row(source_row, vector, n_tokens, truncated, embed_plan))
            progress = pd.concat([progress, pd.DataFrame(rows)], ignore_index=True)
            progress = _validate_embeddings(progress, generation, embed_plan, complete=False)
            atomic_parquet(progress.sort_values(["split", "arm", "position"]), progress_path)
            print(f"[{model}] embedded {start + len(rows)}/{len(pending)} pending documents", flush=True)
    finally:
        if embedder is not None:
            del embedder
            import torch
            torch.cuda.empty_cache()

    output = pd.concat([pd.DataFrame(core_rows), progress], ignore_index=True)
    output = output.sort_values(["split", "arm", "position"]).reset_index(drop=True)
    output = _validate_embeddings(output, generation, embed_plan, complete=True)
    atomic_parquet(output, output_path)
    _validate_embeddings(pd.read_parquet(output_path), generation, embed_plan, complete=True)
    after_hashes = load_core(root, model)[1]
    if after_hashes != core_hashes:
        raise ValueError(f"{model}: core MMD files changed during control embedding")
    output_hash = sha256_file(output_path)
    atomic_json({
        "schema_version": EMBED_DONE_SCHEMA, "suite": SUITE, "model": model,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
        "embedder": EMBEDDER, "revision": REVISION, "max_length": MAX_LENGTH,
        "pooling": POOLING, "batch_size": 1, "dtype": "bfloat16",
        "source_sha256": embed_plan["source_sha256"], "plan_sha256": embed_plan["plan_sha256"],
        "embed_plan_sha256": embed_plan["embed_plan_sha256"],
        "generation_sha256": embed_plan["generation_sha256"],
        "equivalence_status": generation_marker["equivalence_status"],
        "runtime_fingerprint_sha256": generation_marker["runtime_fingerprint_sha256"],
        "val1_anchors_sha256": generation_marker["val1_anchors_sha256"],
        "embeddings_sha256": output_hash, "core_hashes": core_hashes, "rows": len(output),
        "copied_core_rows": len(core_rows), "new_rows": len(output) - len(core_rows),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }, done_path)
    progress_path.unlink(missing_ok=True)
    print(f"[{model}] control embeddings complete sha256={output_hash}", flush=True)


def self_test() -> None:
    generation = pd.DataFrame([
        {"model": "L1", "split": "test_jmq", "arm": arm, "prompt_id": "p1", "position": 0,
         "text": text, "text_sha256": text_sha256(text), "source_sha256": "a" * 64,
         "plan_sha256": "b" * 64, "runtime_stratum": CURRENT_STRATUM,
         "runtime_fingerprint_sha256": "d" * 64}
        for arm, text in (("HUMAN", "human"), ("ANCHOR-BASE", "base"), ("ANCHOR-CONE", "cone"))
    ])
    vectors = {
        "human": np.array([1.0, 0.0], dtype=np.float32),
        "baseline": np.array([0.0, 1.0], dtype=np.float32),
        "ablated": np.array([2**-0.5, 2**-0.5], dtype=np.float32),
    }
    core = pd.DataFrame([
        {"prompt_id": "p1", "source": source, "text_sha256": text_sha256(text),
         "embedding": vectors[source], "n_tokens": 1, "truncated": False}
        for source, text in (("human", "human"), ("baseline", "base"), ("ablated", "cone"))
    ])
    plan = {"embed_plan_sha256": "c" * 64}
    rows = copy_core_rows(generation, {"test": core, "val2": core.iloc[0:0]}, plan)
    assert len(rows) == 1 and all(row["embedding_origin"].startswith("core-copy") for row in rows)
    frame = pd.DataFrame(rows)
    full_plan = {
        "model": "L1", "source_sha256": "a" * 64, "plan_sha256": "b" * 64,
        "embed_plan_sha256": "c" * 64,
    }
    _validate_embeddings(frame, generation, full_plan, complete=False)
    human_generation = generation[generation["arm"] == "HUMAN"].reset_index(drop=True)
    _validate_embeddings(frame, human_generation, full_plan, complete=True)
    with tempfile.TemporaryDirectory(prefix="control-embed-self-test-") as directory:
        core_path = Path(directory) / "core.parquet"
        atomic_parquet(core, core_path)
        before = sha256_file(core_path)
        output_path = Path(directory) / "controls.parquet"
        atomic_parquet(frame, output_path)
        assert sha256_file(core_path) == before
        damaged = frame.copy()
        damaged.at[0, "text_sha256"] = "0" * 64
        try:
            _validate_embeddings(damaged, generation, full_plan, complete=False)
            raise AssertionError("stale embedding text hash was accepted")
        except ValueError:
            pass
    print("Control embed synthetic self-test passed (CPU only)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.models:
        parser.error("--models is required unless --self-test is used")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    for model in args.models:
        run_model(args.root, model, args.preflight)


if __name__ == "__main__":
    main()
