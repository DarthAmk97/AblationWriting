"""Generate the frozen H2-MRSC-400-v5 RAND-only control epoch.

TEST baseline, frozen cone, PC1, and random-rank arms are generated together
in one target-model runtime.  Archived TEST is diagnostic-only; archived VAL2
FULL-SYM remains a separate secondary runtime stratum.

Usage:
  python src/h2_controls.py --models L1 O1 Q08 Q20 G2
  python src/h2_controls.py --models L1 --preflight
  python src/h2_controls.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SUITE = "H2-MRSC-400-v5"
SCHEMA = "h2-mrsc-400.generations.v5"
PLAN_SCHEMA = "h2-mrsc-400.plan.v5"
VAL1_SCHEMA = "h2-mrsc-400.val1-anchors.v5"
EPOCH_ID = "H2-MRSC-400-v5-rand-only"
CURRENT_STRATUM = "contemporaneous-v5-runtime"
PRIOR_SUITE = "H2-MRSC-400-v4"
ARCHIVED_STRATUM = "archived-original-runtime"
MODELS = ("L1", "O1", "Q08", "Q20", "G2")
MODEL_PINS = {
    "L1": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
    "O1": ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e"),
    "Q08": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "Q20": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    "G2": ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
}
FROZEN_RUNS = {
    "L1": "H2-full-s3-L1-W-BEST-r4-a0.75-rho0.5-cone",
    "O1": "H2-full-s3-O1-W-BEST-r2-a0.75-rho0.5-cone",
    "Q08": "H2-full-s3-Q08-W-A-r8-a1.0-rho0.5-cone",
    "Q20": "H2-full-s3-Q20-W-B-r32-a0.25-rho1.0-cone",
    "G2": "H2-full-s3-G2-W-A-r8-a0.5-rho1.0-cone",
}
EXPECTED_LAYERS = {
    "L1": [8, 9, 10, 11],
    "O1": [8, 9, 10, 11],
    "Q08": [18, 19, 20, 21, 22],
    "Q20": [11, 12, 13, 14, 15],
    "G2": [11, 12, 13, 14, 15, 16],
}
MULTIMODAL = {"Q08", "Q20", "G2"}
EVAL_MODEL, EVAL_REVISION = MODEL_PINS["Q08"]
EMBEDDER_PIN = ("nvidia/llama-embed-nemotron-8b", "aa3b43a495a9b280d1bdb716da37c54bb495d630")
DATASET_PIN = ("szyszy/GEN", "a0f143c1ae9c35b684898a5f9c0ab6efb49be486")
IMMUTABLE_CODE_HASHES = {
    "src/h2_lib.py": "edcb255a053b08879b0a3b0fd8162c284b9255ee6ec8d846bd41460b2b117f28",
    "src/smoke_h2.py": "751a399e598ba5590bfa57646d9caaed4b783068e91836ad402295726eb8ca9c",
}

TEST_COUNT = 2000
CONTROL_COUNT = 400
VAL2_COUNT = 512
CALIBRATION_COUNT = 64
LEX_CALIBRATION_COUNT = 96
RECONSTRUCTION_COUNT = 8
CHECKPOINT_PROMPTS = 16
MAX_NEW_TOKENS = 1024
SEED_BASE = 1000
SEED_MODULUS = 2**31 - 1
RANDOM_SEED_BASE = 17011
ALPHA_LIMIT = 4.0
LEX_K_VALUES = (1, 2, 4, 8, 16, 32)
LEX_Q_VALUES = (0.125, 0.25, 0.5, 0.75, 1.0)
GENERATED_ARMS = ("RAND-RANK-DOSE",)
# PC1-DOSE dropped in v4: nested top direction carries ~1/17 of the frozen
# cone's positive-part removal (v2 per-layer alpha 25.15, v3 total alpha 16.83
# on L1, cap 4.0). Recorded as an infeasibility finding, not generated.
TEST_GENERATED_ARMS = ("ANCHOR-BASE", "ANCHOR-CONE", *GENERATED_ARMS)
TEST_ARMS = ("HUMAN", "ANCHOR-BASE", "ANCHOR-CONE", *GENERATED_ARMS, "LEX-MATCH")
VAL2_ARMS = ("HUMAN", "VAL2-BASE", "VAL2-CONE", "FULL-SYM")
VAL1_ARMS = ("ANCHOR-BASE", "ANCHOR-CONE")
RANDOM_SEED_FORMULA = (
    '(17011 + int(sha256("H2-MRSC|RAND|<mid>|<layer>")[:8],16)) % 2147483647'
)
LEX_DELETE_FORMULA = (
    "int(sha256('H2-MRSC|LEX|<mid>|<prompt_id>|<token_position>'),16) < q*2**256; "
    "token_position is zero-based"
)
SAMPLER = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "max_new_tokens": MAX_NEW_TOKENS,
    "batch_size": 1,
    "dtype": "bfloat16",
}
EPOCH_FINGERPRINT = hashlib.sha256(json.dumps({
    "epoch_id": EPOCH_ID,
    "suite": SUITE,
    "test_arms": TEST_ARMS,
    "val1_arms": VAL1_ARMS,
    "val2_arms": VAL2_ARMS,
    "immutable_code": IMMUTABLE_CODE_HASHES,
    "anchor_policy": "same-process-contemporaneous",
}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
PROTECTED_CONTRASTS = (
    ("Follow: answer in exactly 3 words. Q: capital of France?", "Answer freely at length. Q: capital of France?"),
    ('Follow: return only JSON {"a":1}.', "Answer the same in prose."),
    ("Task: 2+2=? Reply with digit only.", "Task: 2+2=? Explain step by step."),
    ("Q: What is the capital of France? Answer Paris.", "Q: What is the capital of Germany? Answer Berlin."),
    ("Summarize in one sentence: The cat sat.", "Write a poem about the cat that sat."),
    ("Decline: I cannot help with that harmful request.", "Comply with this benign request: name a fruit."),
    ("Use exactly the headings A/B/C.", "Write with no headings."),
    ("Answer concisely.", "Answer with full background detail."),
)


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        frame.to_parquet(tmp, index=False)
        with open(tmp, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing frozen source: {path}")
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def stable_prompt_seed(prompt: str) -> int:
    return (SEED_BASE + int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)) % SEED_MODULUS


def random_basis_seed(model: str, layer: int) -> int:
    payload = f"H2-MRSC|RAND|{model}|{layer}".encode("ascii")
    return (RANDOM_SEED_BASE + int(hashlib.sha256(payload).hexdigest()[:8], 16)) % SEED_MODULUS


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def network_allowed() -> bool:
    """Network access is opt-in; archive preflight and local use stay offline by default."""
    return os.environ.get("H2_CONTROLS_ALLOW_NETWORK") == "1"


def eval_cache_dir() -> str | None:
    return os.environ.get("H2_CONTROLS_EVAL_CACHE") or None


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def _ordered_frame(frame: pd.DataFrame, ids: list[str], label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    if len(frame) != len(ids) or frame["prompt_id"].duplicated().any() or set(frame["prompt_id"]) != set(ids):
        raise ValueError(f"{label}: prompt IDs do not exactly match the frozen split")
    return frame.set_index("prompt_id").loc[ids].reset_index()


def _validate_manifest(root: Path) -> str:
    path = root / "manifest.yaml"
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    text = path.read_text(encoding="utf-8")
    for model, (repo, revision) in MODEL_PINS.items():
        pattern = rf"(?m)^\s*{re.escape(model)}:\s*\{{[^\n]*hf:\s*{re.escape(repo)},[^\n]*sha:\s*{revision}(?:,|\}})"
        if not re.search(pattern, text):
            raise ValueError(f"manifest pin mismatch for {model}; expected {repo}@{revision}")
    for label, (repo, revision) in {"embedding": EMBEDDER_PIN, "GEN": DATASET_PIN}.items():
        if repo not in text or revision not in text:
            raise ValueError(f"manifest pin mismatch for {label}; expected {repo}@{revision}")
    return sha256_file(path)


def _validate_split(rows: list[dict], count: int, name: str) -> list[str]:
    if len(rows) != count:
        raise ValueError(f"{name}: expected exactly {count} rows, found {len(rows)}")
    ids = [str(row.get("prompt_id")) for row in rows]
    if "None" in ids or len(set(ids)) != count:
        raise ValueError(f"{name}: prompt_id values must be present and unique")
    for row in rows:
        if row.get("dataset") != DATASET_PIN[0] or row.get("revision") != DATASET_PIN[1]:
            raise ValueError(f"{name}: dataset revision mismatch at prompt_id {row.get('prompt_id')}")
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("human_text"), str):
            raise TypeError(f"{name}: prompt and human_text must be strings")
    return ids


def _load_frozen_record(root: Path, model: str, geom: Path) -> tuple[dict, dict]:
    frozen = read_json(geom / "frozen.json")
    expected = FROZEN_RUNS[model]
    if frozen.get("frozen") != expected:
        raise ValueError(f"{model}: frozen run changed; expected {expected!r}")
    records = [row for row in read_jsonl(root / "runs.jsonl") if row.get("run_id") == expected]
    if len(records) != 1:
        raise ValueError(f"{model}: expected one frozen run record, found {len(records)}")
    record = records[0]
    required = {
        "model": model, "stage": "s3", "cone": "cone", "split": "VAL2-512",
        "status": "ok", "max_new": MAX_NEW_TOKENS,
    }
    for key, value in required.items():
        if record.get(key) != value:
            raise ValueError(f"{model}: frozen run metadata mismatch at {key}")
    if record.get("layers") != EXPECTED_LAYERS[model]:
        raise ValueError(f"{model}: frozen layer list changed")
    if not isinstance(record.get("rank"), int) or record["rank"] < 2:
        raise ValueError(f"{model}: invalid frozen rank")
    if not (0 < float(record.get("alpha", 0)) <= 1 and 0 <= float(record.get("rho", -1)) <= 1):
        raise ValueError(f"{model}: invalid frozen alpha/rho")
    return frozen, record


def _validate_geometry(geom: Path, record: dict) -> dict[str, str]:
    hashes = {}
    dimensions = set()
    for layer in record["layers"]:
        path = geom / f"full_layer{layer:02d}.npz"
        if not path.is_file():
            raise ValueError(f"missing frozen geometry: {path}")
        with np.load(path) as values:
            if not {"mu", "W", "V"} <= set(values.files):
                raise ValueError(f"{path}: expected mu, W, and V arrays")
            mu, whitening, vectors = values["mu"], values["W"], values["V"]
            if (mu.ndim != 1 or whitening.shape != (len(mu), len(mu)) or vectors.ndim != 2
                    or vectors.shape[1] != len(mu) or vectors.shape[0] < record["rank"]):
                raise ValueError(f"{path}: incompatible geometry shapes")
            if not all(np.isfinite(array).all() for array in (mu, whitening, vectors)):
                raise ValueError(f"{path}: geometry contains non-finite values")
            dimensions.add(len(mu))
        hashes[str(path.relative_to(geom.parents[3])).replace("\\", "/")] = sha256_file(path)
    if len(dimensions) != 1:
        raise ValueError("frozen layers have inconsistent hidden dimensions")
    return hashes


def prepare_sources(root: Path, model: str) -> dict:
    """Validate every immutable source needed by one model before model loading."""
    root = Path(root)
    if model not in MODELS:
        raise ValueError(f"unsupported model {model!r}")
    manifest_hash = _validate_manifest(root)
    split_dir = root / "data" / "splits"
    test_path, jmq_path = split_dir / "test.jsonl", split_dir / "test_jmq.jsonl"
    val1_path, val2_path = split_dir / "val1.jsonl", split_dir / "val2.jsonl"
    test = read_jsonl(test_path)
    test_ids = _validate_split(test, TEST_COUNT, "TEST")
    jmq = read_jsonl(jmq_path)
    jmq_ids = _validate_split(jmq, CONTROL_COUNT, "TEST-JMQ")
    if canonical_json(jmq) != canonical_json(test[:CONTROL_COUNT]) or jmq_ids != test_ids[:CONTROL_COUNT]:
        raise ValueError("TEST-JMQ is not the first fixed 400 TEST rows in exact order")
    val1 = read_jsonl(val1_path)
    val1_ids = _validate_split(val1, 256, "VAL1")
    val2 = read_jsonl(val2_path)
    val2_ids = _validate_split(val2, VAL2_COUNT, "VAL2")

    geom = root / "artifacts" / "geometry" / "H2" / model
    frozen, record = _load_frozen_record(root, model, geom)
    prefix = f"H2-full-s3-{model}-"
    cone_config = frozen["frozen"][len(prefix):]
    done_path = geom / "test_done"
    done = read_json(done_path)
    if done.get("n") != TEST_COUNT or done.get("frozen") != frozen["frozen"]:
        raise ValueError(f"{model}: TEST completion marker does not match the frozen run")
    test_metrics_path = geom / "test_metrics.json"
    test_metrics = read_json(test_metrics_path)
    if test_metrics.get("n") != TEST_COUNT or test_metrics.get("frozen") != frozen["frozen"]:
        raise ValueError(f"{model}: TEST metrics do not match the frozen run")

    test_gen_path = geom / "test_generations.parquet"
    if not test_gen_path.is_file():
        raise ValueError(f"{model}: missing archived TEST generations")
    test_frame = pd.read_parquet(test_gen_path)
    _require_columns(test_frame, {"prompt_id", "prompt", "human", "baseline", "ablated"}, f"{model} TEST")
    test_frame = _ordered_frame(test_frame, test_ids, f"{model} TEST")
    for index, (source, archived) in enumerate(zip(test, test_frame.itertuples(index=False))):
        if archived.prompt != source["prompt"] or archived.human != source["human_text"]:
            raise ValueError(f"{model}: TEST source mismatch at position {index}")
        if not isinstance(archived.baseline, str) or not isinstance(archived.ablated, str):
            raise TypeError(f"{model}: TEST output is not text at position {index}")

    s3_path = geom / "s3_generations.parquet"
    if not s3_path.is_file():
        raise ValueError(f"{model}: missing archived VAL2 S3 generations")
    s3 = pd.read_parquet(s3_path)
    _require_columns(s3, {"prompt_id", "baseline", "ablated", "config"}, f"{model} S3")
    full_config = cone_config.removesuffix("-cone") + "-full"
    cone = _ordered_frame(s3[s3["config"].astype(str) == cone_config], val2_ids, f"{model} VAL2 cone")
    full = _ordered_frame(s3[s3["config"].astype(str) == full_config], val2_ids, f"{model} VAL2 full")
    if cone["baseline"].tolist() != full["baseline"].tolist():
        raise ValueError(f"{model}: VAL2 cone/full archived baselines differ")
    for column in ("baseline", "ablated"):
        if not cone[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{model}: VAL2 cone {column} contains non-text values")
        if not full[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{model}: VAL2 full {column} contains non-text values")

    source_files = {
        "manifest.yaml": manifest_hash,
        "data/splits/test.jsonl": sha256_file(test_path),
        "data/splits/test_jmq.jsonl": sha256_file(jmq_path),
        "data/splits/val1.jsonl": sha256_file(val1_path),
        "data/splits/val2.jsonl": sha256_file(val2_path),
        str(done_path.relative_to(root)).replace("\\", "/"): sha256_file(done_path),
        str(test_metrics_path.relative_to(root)).replace("\\", "/"): sha256_file(test_metrics_path),
        str(test_gen_path.relative_to(root)).replace("\\", "/"): sha256_file(test_gen_path),
        str(s3_path.relative_to(root)).replace("\\", "/"): sha256_file(s3_path),
        str((geom / "frozen.json").relative_to(root)).replace("\\", "/"): sha256_file(geom / "frozen.json"),
    }
    for relative, expected_hash in IMMUTABLE_CODE_HASHES.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"immutable source hash mismatch: {relative}")
        source_files[relative] = expected_hash
    source_files.update(_validate_geometry(geom, record))
    semantic_sources = {
        "frozen_record": record,
        "test_jmq": [[row["prompt_id"], text_sha256(row["prompt"]), text_sha256(row["human_text"])] for row in jmq],
        "test_archive": [[row.prompt_id, text_sha256(row.baseline), text_sha256(row.ablated)]
                         for row in test_frame.iloc[:CONTROL_COUNT].itertuples()],
        "val1_calibration": [[row["prompt_id"], text_sha256(row["prompt"])] for row in val1[:CALIBRATION_COUNT]],
        "val1_lexical_calibration": [
            [row["prompt_id"], text_sha256(row["prompt"]), text_sha256(row["human_text"])]
            for row in val1[:LEX_CALIBRATION_COUNT]
        ],
        "val2_full_sym": [[row.prompt_id, text_sha256(row.ablated)] for row in full.itertuples()],
    }
    semantic_hashes = {key: sha256_bytes(canonical_json(value)) for key, value in semantic_sources.items()}
    source_hash = sha256_bytes(canonical_json({"files": source_files, "semantic_hashes": semantic_hashes}))
    return {
        "root": root, "geom": geom, "model": model, "frozen": frozen, "record": record,
        "test_jmq": jmq, "val1_calibration": val1[:CALIBRATION_COUNT],
        "val1_lexical": val1[:LEX_CALIBRATION_COUNT],
        "val2": val2,
        "test_frame": test_frame.iloc[:CONTROL_COUNT].copy(), "val2_cone": cone, "val2_full": full,
        "cone_config": cone_config, "full_config": full_config, "source_files": source_files,
        "semantic_hashes": semantic_hashes, "source_sha256": source_hash,
    }


def make_plan(source: dict) -> dict:
    model = source["model"]
    record = source["record"]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "suite": SUITE,
        "epoch_id": EPOCH_ID,
        "epoch_fingerprint": EPOCH_FINGERPRINT,
        "model": model,
        "model_id": MODEL_PINS[model][0],
        "model_revision": MODEL_PINS[model][1],
        "eval_tokenizer": {"model": EVAL_MODEL, "revision": EVAL_REVISION},
        "frozen": {
            "run_id": source["frozen"]["frozen"], "window": record["window"],
            "layers": record["layers"], "rank": record["rank"],
            "alpha": float(record["alpha"]), "rho": float(record["rho"]),
        },
        "arms": {
            "test_jmq": list(TEST_ARMS), "val1": list(VAL1_ARMS), "val2": list(VAL2_ARMS),
            "generated_test": ["ANCHOR-BASE", "ANCHOR-CONE", *GENERATED_ARMS],
            "cpu_test": ["LEX-MATCH"], "reused_test": ["HUMAN"],
            "reused_val2": list(VAL2_ARMS),
        },
        "counts": {
            "test_jmq": CONTROL_COUNT, "val2": VAL2_COUNT,
            "val1_contemporaneous_anchors": LEX_CALIBRATION_COUNT,
            "dose_calibration_val1": CALIBRATION_COUNT,
            "lexical_calibration_val1": LEX_CALIBRATION_COUNT,
            "runtime_reconstruction_test_jmq": RECONSTRUCTION_COUNT,
        },
        "sampler": SAMPLER,
        "seed": {
            "base": SEED_BASE,
            "prompt_rule": "(1000 + int(sha256(prompt)[:8],16)) % (2**31-1)",
            "modulus": SEED_MODULUS,
            "random_basis_rule": RANDOM_SEED_FORMULA,
            "random_basis_by_layer": {str(layer): random_basis_seed(model, layer) for layer in record["layers"]},
        },
        "dose_calibration": {
            "split": "first fixed 64 VAL1 rows", "statistic": "mean generated-token L2 removal norm per layer",
            "target": "summed frozen-cone dose across layers at unit-dose removal",
            "matching": "total-dose: one alpha per arm; same value applied at every frozen layer",
            "maximum_alpha": ALPHA_LIMIT,
            "overflow": "fail; never clip",
        },
        "pc1_basis": "exact frozen final cleaned/QR basis first column: frozen[:, :1]",
        "dropped_arms": {
            "PC1-DOSE": ("infeasible under dose matching: v2 per-layer alpha 25.15 "
                         "(L1 layer 8), v3 total-dose alpha 16.83 (L1), cap 4.0; "
                         "nested top direction carries ~1/17 of cone removal; "
                         "recorded as multidirectionality evidence, not generated"),
        },
        "lexical_control": {
            "calibration": "contemporaneous no-hook/frozen-cone generations on first 96 VAL1 rows",
            "K": list(LEX_K_VALUES), "q": list(LEX_Q_VALUES),
            "ranking": "positive baseline-minus-human unigram excess, descending squared excess",
            "deletion": LEX_DELETE_FORMULA,
            "selection": (
                "min |absolute L2-1 shift(LEX)-absolute L2-1 shift(H2)|; "
                "ties fewer deleted occurrences, lower q, lower K"
            ),
            "matched": "same direction and absolute magnitude ratio in [0.8,1.25]",
        },
        "runtime_reconstruction_preflight": {
            "split": "first fixed 8 TEST-JMQ rows", "arms": ["ANCHOR-BASE", "ANCHOR-CONE"],
            "gate": (
                "same-runtime immutable smoke_h2.generate_batch/generate_with_hooks versus "
                "new no-hook/GeneratedTokenHook exact token/text equivalence"
            ),
            "archive_replay": "environment-identity diagnostic only; never hook-equivalence evidence",
            "failure": "abort before dose calibration or new controls",
        },
        "runtime_strata": {
            "test_jmq": CURRENT_STRATUM,
            "val1": CURRENT_STRATUM,
            "val2": ARCHIVED_STRATUM,
            "pooling_policy": "never pool absolute outputs across runtime strata",
        },
        "budget_gpu_hours": {
            "nominal": [45, 51], "hard_cap": 72,
            "reason": "contemporaneous TEST baseline/cone plus VAL1-96 anchors; PC1 dropped",
        },
        "immutable_runtime_sources": IMMUTABLE_CODE_HASHES,
        "model_loading": {
            "default": "local_files_only=True",
            "network_opt_in": "H2_CONTROLS_ALLOW_NETWORK=1",
            "eval_cache_env": "H2_CONTROLS_EVAL_CACHE",
            "revision_policy": "manifest-pinned exact revision",
        },
        "checkpoint_prompts": CHECKPOINT_PROMPTS,
        "supersedes": {
            "epoch": PRIOR_SUITE,
            "adoptable": ["controls_generations.progress.parquet", "controls_val1_anchors.parquet"],
            "regenerated": ["equivalence gate", "dose calibration", "lexical policy", "completion markers"],
            "reason": "v4 lex-merge defect; generation semantics unchanged",
        },
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "source_files": source["source_files"],
        "semantic_hashes": source["semantic_hashes"],
        "source_sha256": source["source_sha256"],
        "full_sym_config": source["full_config"],
    }
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan


def ensure_plan(path: Path, plan: dict, write: bool) -> None:
    if path.exists():
        saved = read_json(path)
        if saved != plan:
            raise ValueError(f"{path}: existing plan differs; refusing resume")
    elif write:
        atomic_json(plan, path)


def _row(source: dict, plan: dict, split: str, arm: str, position: int,
         prompt_id: str, prompt: str, text: str, provenance: str, seed: int = -1) -> dict:
    stratum = ARCHIVED_STRATUM if split == "val2" else CURRENT_STRATUM
    return {
        "schema_version": SCHEMA, "suite": SUITE, "model": source["model"],
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
        "runtime_stratum": stratum,
        "runtime_fingerprint_sha256": (
            "archived-original-runtime-unavailable" if stratum == ARCHIVED_STRATUM
            else source.get("runtime_fingerprint_sha256", "runtime-not-yet-established")
        ),
        "model_revision": MODEL_PINS[source["model"]][1], "split": split, "arm": arm,
        "prompt_id": str(prompt_id), "position": int(position),
        "prompt_sha256": text_sha256(prompt), "text": text, "text_sha256": text_sha256(text),
        "provenance": provenance, "seed": int(seed), "source_sha256": source["source_sha256"],
        "plan_sha256": plan["plan_sha256"],
    }


def archived_rows(source: dict, plan: dict) -> list[dict]:
    rows = []
    for position, split_row in enumerate(source["test_jmq"]):
        rows.append(_row(
            source, plan, "test_jmq", "HUMAN", position, str(split_row["prompt_id"]),
            split_row["prompt"], split_row["human_text"], "archived-human",
        ))
    for position, (split_row, cone, full) in enumerate(zip(
            source["val2"], source["val2_cone"].itertuples(), source["val2_full"].itertuples())):
        values = {
            "HUMAN": split_row["human_text"], "VAL2-BASE": cone.baseline,
            "VAL2-CONE": cone.ablated, "FULL-SYM": full.ablated,
        }
        for arm, text in values.items():
            rows.append(_row(source, plan, "val2", arm, position, str(split_row["prompt_id"]),
                             split_row["prompt"], text, "archived-core"))
    return rows


def _val1_row(source: dict, plan: dict, arm: str, position: int, text: str) -> dict:
    split_row = source["val1_lexical"][position]
    row = _row(
        source, plan, "val1", arm, position, str(split_row["prompt_id"]),
        split_row["prompt"], text, "generated-contemporaneous",
        stable_prompt_seed(split_row["prompt"]),
    )
    row["schema_version"] = VAL1_SCHEMA
    return row


def _validate_val1_anchors(frame: pd.DataFrame, source: dict, plan: dict, complete: bool) -> pd.DataFrame:
    required = {
        "schema_version", "suite", "epoch_id", "epoch_fingerprint", "model", "model_revision",
        "runtime_stratum", "runtime_fingerprint_sha256", "split", "arm", "prompt_id", "position",
        "prompt_sha256", "text", "text_sha256", "provenance", "seed", "source_sha256", "plan_sha256",
    }
    _require_columns(frame, required, "VAL1 anchors")
    frame = frame.copy()
    if len(frame) and set(frame["suite"].astype(str)) == {PRIOR_SUITE}:
        frame = _adopt_prior_rows(frame, source, plan)
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    expected_meta = {
        "schema_version": VAL1_SCHEMA, "suite": SUITE, "epoch_id": EPOCH_ID,
        "epoch_fingerprint": EPOCH_FINGERPRINT, "model": source["model"],
        "model_revision": MODEL_PINS[source["model"]][1], "runtime_stratum": CURRENT_STRATUM,
        "runtime_fingerprint_sha256": source["runtime_fingerprint_sha256"],
        "split": "val1", "source_sha256": source["source_sha256"],
        "plan_sha256": plan["plan_sha256"], "provenance": "generated-contemporaneous",
    }
    for column, value in expected_meta.items():
        if set(frame[column].astype(str)) != {str(value)}:
            raise ValueError(f"VAL1 anchors: metadata mismatch at {column}")
    if frame.duplicated(["arm", "prompt_id"]).any() or not set(frame["arm"]) <= set(VAL1_ARMS):
        raise ValueError("VAL1 anchors: duplicate or unexpected arm rows")
    expected_prompt = {
        str(row["prompt_id"]): (position, text_sha256(row["prompt"]), stable_prompt_seed(row["prompt"]))
        for position, row in enumerate(source["val1_lexical"])
    }
    for row in frame.itertuples():
        expected = expected_prompt.get(row.prompt_id)
        if (expected is None or (int(row.position), row.prompt_sha256, int(row.seed)) != expected
                or not isinstance(row.text, str) or text_sha256(row.text) != row.text_sha256):
            raise ValueError("VAL1 anchors: prompt/seed/text hash mismatch")
    if complete:
        expected = {(arm, prompt_id) for arm in VAL1_ARMS for prompt_id in expected_prompt}
        if set(zip(frame["arm"], frame["prompt_id"])) != expected or len(frame) != len(expected):
            raise ValueError("VAL1 anchors: final artifact is partial")
    return frame


def generate_val1_anchors(model, tokenizer, layer_map: dict[int, object],
                          bases: dict[int, dict[str, np.ndarray]], source: dict, plan: dict,
                          reconstruction_hash: str) -> tuple[pd.DataFrame, Path, Path]:
    output_path = source["geom"] / "controls_val1_anchors.parquet"
    progress_path = source["geom"] / "controls_val1_anchors.progress.parquet"
    done_path = source["geom"] / "control_val1_anchors_done.json"
    if output_path.exists() and done_path.exists():
        try:
            frame = _validate_val1_anchors(pd.read_parquet(output_path), source, plan, complete=True)
            done = read_json(done_path)
            if (done.get("anchors_sha256") != sha256_file(output_path)
                    or done.get("plan_sha256") != plan["plan_sha256"]
                    or done.get("runtime_fingerprint_sha256") != source["runtime_fingerprint_sha256"]
                    or done.get("reconstruction_preflight_sha256") != reconstruction_hash):
                raise ValueError("VAL1 anchor completion marker is stale")
            return frame, output_path, done_path
        except ValueError:
            pass
        raw = pd.read_parquet(output_path)
        if set(raw["suite"].astype(str)) != {PRIOR_SUITE}:
            raise ValueError("VAL1 anchor artifact/marker is incomplete; refusing overwrite")
        frame = _validate_val1_anchors(
            _adopt_prior_rows(raw, source, plan), source, plan, complete=True
        ).sort_values(["arm", "position"]).reset_index(drop=True)
        atomic_parquet(frame, output_path)
        anchors_hash = sha256_file(output_path)
        atomic_json({
            "schema_version": "h2-mrsc-400.val1-anchors-done.v5", "suite": SUITE,
            "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
            "model": source["model"], "source_sha256": source["source_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "runtime_fingerprint_sha256": source["runtime_fingerprint_sha256"],
            "reconstruction_preflight_sha256": reconstruction_hash,
            "anchors_sha256": anchors_hash, "rows": len(frame),
            "counts": {arm: LEX_CALIBRATION_COUNT for arm in VAL1_ARMS},
            "adopted_from_epoch": PRIOR_SUITE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }, done_path)
        progress_path.unlink(missing_ok=True)
        print(f"[{source['model']}] adopted blocked prior-epoch VAL1 anchors ({len(frame)} rows)", flush=True)
        return frame, output_path, done_path
    if output_path.exists() != done_path.exists():
        raise ValueError("VAL1 anchor artifact/marker is incomplete; refusing overwrite")
    frame = pd.read_parquet(progress_path) if progress_path.exists() else pd.DataFrame()
    if len(frame):
        frame = _validate_val1_anchors(frame, source, plan, complete=False)
    prompts = [row["prompt"] for row in source["val1_lexical"]]
    ids = [str(row["prompt_id"]) for row in source["val1_lexical"]]
    device = next(model.parameters()).device
    for arm in VAL1_ARMS:
        hooks, handles = {}, []
        try:
            if arm == "ANCHOR-CONE":
                for layer, values in bases.items():
                    hook = GeneratedTokenHook(
                        values["frozen"], values["mu"], float(source["record"]["alpha"]), device
                    )
                    hooks[layer] = hook
                    handles.append(layer_map[layer].register_forward_hook(hook))
            existing = set(frame.loc[frame.get("arm", pd.Series(dtype=str)) == arm, "prompt_id"]) if len(frame) else set()
            todo = [index for index, prompt_id in enumerate(ids) if prompt_id not in existing]
            for start in range(0, len(todo), CHECKPOINT_PROMPTS):
                indices = todo[start:start + CHECKPOINT_PROMPTS]
                rows = [
                    _val1_row(
                        source, plan, arm, index,
                        _generate_one(model, tokenizer, prompts[index], hooks or None),
                    )
                    for index in indices
                ]
                frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
                frame = _validate_val1_anchors(frame, source, plan, complete=False)
                atomic_parquet(frame.sort_values(["arm", "position"]), progress_path)
                print(f"[{source['model']}] VAL1 {arm} {len(existing) + start + len(indices)}/{len(ids)}", flush=True)
        finally:
            for handle in handles:
                handle.remove()
    frame = _validate_val1_anchors(frame, source, plan, complete=True).sort_values(
        ["arm", "position"]
    ).reset_index(drop=True)
    atomic_parquet(frame, output_path)
    anchors_hash = sha256_file(output_path)
    atomic_json({
        "schema_version": "h2-mrsc-400.val1-anchors-done.v5", "suite": SUITE,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
        "model": source["model"], "source_sha256": source["source_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "runtime_fingerprint_sha256": source["runtime_fingerprint_sha256"],
        "reconstruction_preflight_sha256": reconstruction_hash,
        "anchors_sha256": anchors_hash, "rows": len(frame),
        "counts": {arm: LEX_CALIBRATION_COUNT for arm in VAL1_ARMS},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }, done_path)
    progress_path.unlink(missing_ok=True)
    return frame, output_path, done_path


def _token_ids(tokenizer, texts: list[str]) -> list[list[int]]:
    return [list(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]


def _frequency(ids: list[list[int]]) -> tuple[Counter, int]:
    counts = Counter(token for document in ids for token in document)
    return counts, sum(counts.values())


def _counter_l2(left: Counter, n_left: int, right: Counter, n_right: int) -> float:
    if n_left <= 0 or n_right <= 0:
        raise ValueError("LEX-MATCH received an empty token corpus")
    return math.sqrt(sum((left.get(key, 0) / n_left - right.get(key, 0) / n_right) ** 2
                         for key in left.keys() | right.keys()))


def _corpus_l2(left_ids: list[list[int]], right_ids: list[list[int]]) -> float:
    left, n_left = _frequency(left_ids)
    right, n_right = _frequency(right_ids)
    return _counter_l2(left, n_left, right, n_right)


def _lex_delete(ids: list[list[int]], prompt_ids: list[str], selected: set[int],
                q: float, model: str) -> tuple[list[list[int]], int]:
    threshold = int(q * (1 << 256))
    output, deleted = [], 0
    for prompt_id, document in zip(prompt_ids, ids):
        kept = []
        for token_position, token in enumerate(document):
            payload = f"H2-MRSC|LEX|{model}|{prompt_id}|{token_position}".encode()
            remove = token in selected and int(hashlib.sha256(payload).hexdigest(), 16) < threshold
            if remove:
                deleted += 1
            else:
                kept.append(token)
        output.append(kept)
    return output, deleted


def _shift_report(baseline_distance: float, h2_distance: float, lex_distance: float) -> dict:
    h2_shift = h2_distance - baseline_distance
    lex_shift = lex_distance - baseline_distance

    def direction(value: float) -> str:
        if value < -1e-15:
            return "toward-human"
        if value > 1e-15:
            return "away-from-human"
        return "none"

    h2_magnitude = abs(h2_shift)
    ratio = abs(lex_shift) / h2_magnitude if h2_magnitude > 0 else None
    same_direction = direction(h2_shift) == direction(lex_shift) and direction(h2_shift) != "none"
    return {
        "baseline_l2_1": baseline_distance,
        "h2_l2_1": h2_distance,
        "lex_l2_1": lex_distance,
        "h2_signed_shift": h2_shift,
        "lex_signed_shift": lex_shift,
        "h2_direction": direction(h2_shift),
        "lex_direction": direction(lex_shift),
        "absolute_magnitude_ratio": ratio,
        "same_direction": same_direction,
        "matched": bool(same_direction and ratio is not None and 0.8 <= ratio <= 1.25),
    }


def _ordered_arm_texts(frame: pd.DataFrame, arm: str, count: int) -> list[str]:
    selected = frame[frame["arm"] == arm].sort_values("position")
    if len(selected) != count or selected["prompt_id"].duplicated().any():
        raise ValueError(f"{arm}: expected exactly {count} ordered rows")
    return selected["text"].tolist()


def calibrate_lex_policy(tokenizer, source: dict, plan: dict,
                         val1_anchors: pd.DataFrame, val1_anchors_hash: str) -> tuple[dict, Path]:
    """Select and atomically freeze LEX K/q/tokens before opening v5 TEST generation."""
    baseline_ids = _token_ids(
        tokenizer, _ordered_arm_texts(val1_anchors, "ANCHOR-BASE", LEX_CALIBRATION_COUNT)
    )
    cone_ids = _token_ids(
        tokenizer, _ordered_arm_texts(val1_anchors, "ANCHOR-CONE", LEX_CALIBRATION_COUNT)
    )
    human_ids = _token_ids(tokenizer, [row["human_text"] for row in source["val1_lexical"]])
    prompt_ids = [str(row["prompt_id"]) for row in source["val1_lexical"]]
    base_counts, base_total = _frequency(baseline_ids)
    human_counts, human_total = _frequency(human_ids)
    baseline_distance = _corpus_l2(baseline_ids, human_ids)
    h2_distance = _corpus_l2(cone_ids, human_ids)
    h2_absolute_shift = abs(h2_distance - baseline_distance)
    ranked = sorted(
        (token for token in base_counts
         if base_counts[token] / base_total > human_counts.get(token, 0) / human_total),
        key=lambda token: (
            -(base_counts[token] / base_total - human_counts.get(token, 0) / human_total) ** 2,
            token,
        ),
    )
    if not ranked:
        raise ValueError(f"{source['model']}: VAL1-96 has no positive baseline-excess unigram")
    best_key, best = None, None
    candidates = []
    for k in LEX_K_VALUES:
        selected_tokens = ranked[:k]
        for q in LEX_Q_VALUES:
            filtered, deleted = _lex_delete(
                baseline_ids, prompt_ids, set(selected_tokens), q, source["model"]
            )
            lex_distance = _corpus_l2(filtered, human_ids)
            lex_absolute_shift = abs(lex_distance - baseline_distance)
            difference = abs(lex_absolute_shift - h2_absolute_shift)
            key = (difference, deleted, q, k)
            candidates.append({
                "K": k, "q": q, "deleted_occurrences": deleted,
                "absolute_shift": lex_absolute_shift, "absolute_difference": difference,
            })
            if best_key is None or key < best_key:
                best_key = key
                best = (k, q, selected_tokens, filtered, deleted, lex_distance)
    k, q, selected_tokens, _, calibration_deleted, calibration_lex_distance = best
    policy = {
        "schema_version": "h2-mrsc-400.lexical-policy.v5", "suite": SUITE,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
        "model": source["model"], "model_revision": MODEL_PINS[source["model"]][1],
        "source_sha256": source["source_sha256"], "plan_sha256": plan["plan_sha256"],
        "runtime_stratum": CURRENT_STRATUM,
        "runtime_fingerprint_sha256": source["runtime_fingerprint_sha256"],
        "val1_anchors_sha256": val1_anchors_hash,
        "method": "deterministic occurrence deletion calibrated only on contemporaneous VAL1-96 anchors",
        "eval_model": EVAL_MODEL, "eval_revision": EVAL_REVISION,
        "calibration_split": "first fixed 96 VAL1 rows, contemporaneous v5 runtime",
        "K": k, "q": q, "selected_token_ids": selected_tokens,
        "ranking": "descending squared positive baseline-minus-human unigram excess; token-id tie-break",
        "deletion_rule": LEX_DELETE_FORMULA,
        "selection_tie_break": "fewer deleted occurrences, lower q, lower K",
        "calibration_deleted_occurrences": calibration_deleted,
        "calibration_val1": _shift_report(
            baseline_distance, h2_distance, calibration_lex_distance
        ),
        "candidates": candidates,
    }
    path = source["geom"] / "controls_lexical_policy.json"
    if path.exists() and read_json(path) != policy:
        raise ValueError(f"{path}: frozen lexical policy differs; refusing TEST generation")
    if not path.exists():
        atomic_json(policy, path)
    return policy, path


def apply_lex_policy(tokenizer, source: dict, plan: dict,
                     policy: dict, test_current: pd.DataFrame) -> tuple[list[dict], dict]:
    test_baseline_ids = _token_ids(
        tokenizer, _ordered_arm_texts(test_current, "ANCHOR-BASE", CONTROL_COUNT)
    )
    test_cone_ids = _token_ids(
        tokenizer, _ordered_arm_texts(test_current, "ANCHOR-CONE", CONTROL_COUNT)
    )
    test_human_ids = _token_ids(tokenizer, [row["human_text"] for row in source["test_jmq"]])
    test_prompt_ids = [str(row["prompt_id"]) for row in source["test_jmq"]]
    applied, deleted = _lex_delete(
        test_baseline_ids, test_prompt_ids, set(policy["selected_token_ids"]),
        float(policy["q"]), source["model"],
    )
    texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in applied]
    rows = [
        _row(
            source, plan, "test_jmq", "LEX-MATCH", position, str(split_row["prompt_id"]),
            split_row["prompt"], text, "cpu-lexical",
        )
        for position, (split_row, text) in enumerate(zip(source["test_jmq"], texts))
    ]
    report = {
        "test_deleted_occurrences": deleted,
        "application_test_jmq": _shift_report(
            _corpus_l2(test_baseline_ids, test_human_ids),
            _corpus_l2(test_cone_ids, test_human_ids),
            _corpus_l2(applied, test_human_ids),
        ),
    }
    return rows, report


def _basis_hash(bases: dict[int, dict[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for layer in sorted(bases):
        digest.update(str(layer).encode("ascii") + b"\0")
        for name in ("mu", "frozen", "pc1", "random"):
            array = np.ascontiguousarray(bases[layer][name], dtype="<f4")
            digest.update(name.encode("ascii") + b"\0" + str(array.shape).encode("ascii") + b"\0")
            digest.update(array.tobytes())
    return digest.hexdigest()


def _orthogonal_basis(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix)
    return q


def _protected_basis(vectors: list[np.ndarray], dimension: int) -> np.ndarray:
    if not vectors:
        return np.zeros((dimension, 0), dtype=np.float64)
    matrix = np.stack(vectors)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    basis = vt[singular > 1e-6].T
    return _orthogonal_basis(basis) if basis.shape[1] else basis


def _clean_basis(raw: np.ndarray, protected: np.ndarray, rho: float) -> np.ndarray:
    basis = raw.copy()
    if protected.shape[1]:
        basis -= rho * protected @ (protected.T @ basis)
    return _orthogonal_basis(basis)


def _map_basis(inverse: np.ndarray, whitened_rows: np.ndarray,
               protected: np.ndarray, rho: float) -> np.ndarray:
    raw = inverse @ whitened_rows.T
    raw /= np.linalg.norm(raw, axis=0, keepdims=True) + 1e-12
    return _clean_basis(raw, protected, rho)


def _mean_states(model, tokenizer, prompt: str, completion: str, layers: set[int], device) -> dict[int, np.ndarray]:
    import torch

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        tokenize=False, add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=2048, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    start = min(len(prompt_ids), encoded["input_ids"].shape[1])
    positions = torch.arange(start, encoded["input_ids"].shape[1], device=device)
    if positions.numel() and tokenizer.eos_token_id is not None and encoded["input_ids"][0, -1] == tokenizer.eos_token_id:
        positions = positions[:-1]
    if not positions.numel():
        raise ValueError("protected contrast has no assistant tokens")
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False)
    return {layer: output.hidden_states[layer + 1][0, positions].float().mean(0).cpu().numpy()
            for layer in layers}


def build_control_bases(model, tokenizer, source: dict, layer_map: dict[int, object]) -> dict[int, dict[str, np.ndarray]]:
    record, geom = source["record"], source["geom"]
    layers = set(record["layers"])
    if not layers <= set(layer_map):
        raise ValueError(f"{source['model']}: model is missing frozen residual layers")
    device = next(model.parameters()).device
    nuisance = {layer: [] for layer in layers}
    for left, right in PROTECTED_CONTRASTS:
        left_states = _mean_states(model, tokenizer, "Task.", left, layers, device)
        right_states = _mean_states(model, tokenizer, "Task.", right, layers, device)
        for layer in layers:
            nuisance[layer].append(left_states[layer] - right_states[layer])

    output = {}
    for layer in sorted(layers):
        with np.load(geom / f"full_layer{layer:02d}.npz") as values:
            mu = np.asarray(values["mu"])
            whitening = np.asarray(values["W"])
            vectors = np.asarray(values["V"])
        try:
            inverse = np.linalg.inv(whitening)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(whitening)
        protected = _protected_basis(nuisance[layer], len(mu))

        frozen = _map_basis(inverse, vectors[:record["rank"]], protected, float(record["rho"]))
        pc1 = frozen[:, :1]
        rng = np.random.default_rng(random_basis_seed(source["model"], layer))
        random_rows = _orthogonal_basis(rng.standard_normal((len(mu), record["rank"]))).T
        random_control = _map_basis(inverse, random_rows, protected, float(record["rho"]))
        output[layer] = {"mu": mu, "frozen": frozen, "pc1": pc1, "random": random_control}
    return output


def _hidden_tensor(output):
    return output[0] if isinstance(output, tuple) else output


class GeneratedTokenHook:
    """Positive-cone removal that never touches the first (prefill) forward."""

    def __init__(self, basis: np.ndarray, mu: np.ndarray, alpha: float, device):
        import torch

        self.basis = torch.as_tensor(basis, device=device, dtype=torch.bfloat16)
        self.mu = torch.as_tensor(mu, device=device, dtype=torch.bfloat16)
        self.alpha = float(alpha)
        self.prefill_len = 0
        self.prefill_seen = False

    def reset(self, prefill_len: int) -> None:
        self.prefill_len = int(prefill_len)
        self.prefill_seen = False

    def __call__(self, _module, _inputs, output):
        import torch

        hidden = _hidden_tensor(output)
        if hidden.ndim != 3:
            raise ValueError(f"residual hook expected [batch,tokens,hidden], got {tuple(hidden.shape)}")
        if not self.prefill_seen:
            self.prefill_seen = True
            return output
        token_count = hidden.shape[1]
        start = 0 if token_count == 1 else self.prefill_len
        if token_count <= start:
            return output
        generated = hidden[:, start:, :]
        basis = self.basis.to(hidden.dtype)
        difference = generated - self.mu.to(hidden.dtype)
        coefficients = torch.clamp(torch.matmul(difference, basis), min=0)
        removed = self.alpha * torch.matmul(coefficients, basis.T)
        changed = hidden.clone()
        changed[:, start:, :] = generated - removed
        return (changed,) + output[1:] if isinstance(output, tuple) else changed


class DoseObserver:
    def __init__(self, controls: dict[str, np.ndarray], mu: np.ndarray, frozen_alpha: float, device):
        import torch

        self.controls = {name: torch.as_tensor(value, device=device, dtype=torch.bfloat16)
                         for name, value in controls.items()}
        self.mu = torch.as_tensor(mu, device=device, dtype=torch.bfloat16)
        self.frozen_alpha = float(frozen_alpha)
        self.prefill_seen = False
        self.prefill_len = 0
        self.sums = {name: 0.0 for name in controls}
        self.tokens = 0

    def reset(self, prefill_len: int) -> None:
        self.prefill_seen = False
        self.prefill_len = int(prefill_len)

    def __call__(self, _module, _inputs, output):
        import torch

        hidden = _hidden_tensor(output)
        if not self.prefill_seen:
            self.prefill_seen = True
            return output
        start = 0 if hidden.shape[1] == 1 else self.prefill_len
        if hidden.shape[1] <= start:
            return output
        generated = hidden[:, start:, :]
        difference = generated - self.mu.to(hidden.dtype)
        with torch.no_grad():
            for name, basis in self.controls.items():
                basis = basis.to(hidden.dtype)
                coefficients = torch.clamp(torch.matmul(difference, basis), min=0)
                norms = torch.matmul(coefficients, basis.T).float().norm(dim=-1)
                scale = self.frozen_alpha if name == "frozen" else 1.0
                self.sums[name] += scale * norms.sum().item()
            self.tokens += generated.shape[0] * generated.shape[1]
        return output


def _find_layers(model) -> dict[int, object]:
    candidates = ("language_model.layers", "model.language_model.layers", "model.layers",
                  "transformer.h", "model.model.layers", "layers")
    for path in candidates:
        try:
            value = model
            for part in path.split("."):
                value = getattr(value, part)
            if len(value) >= 8:
                return {index: value[index] for index in range(len(value))}
        except (AttributeError, TypeError):
            continue
    raise ValueError("could not locate residual block list")


def load_target(model_code: str):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("H2-MRSC generation requires CUDA")
    repo, revision = MODEL_PINS[model_code]
    local_only = not network_allowed()
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        repo, revision=revision, trust_remote_code=True, local_files_only=local_only, token=token,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"{model_code}: tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    loader = AutoModelForImageTextToText if model_code in MULTIMODAL else AutoModelForCausalLM
    model = loader.from_pretrained(
        repo, revision=revision, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        trust_remote_code=True, local_files_only=local_only, token=token,
    ).eval()
    floating = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if floating != {torch.bfloat16}:
        raise ValueError(f"{model_code}: model is not entirely BF16 (floating dtypes={floating})")
    commits = {getattr(model.config, "_commit_hash", None), tokenizer.init_kwargs.get("_commit_hash")}
    if any(commit not in (None, revision) for commit in commits):
        raise ValueError(f"{model_code}: loaded cache commit does not match manifest revision")
    return tokenizer, model, _find_layers(model)


def load_eval_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        EVAL_MODEL, revision=EVAL_REVISION, trust_remote_code=True,
        local_files_only=not network_allowed(), token=os.environ.get("HF_TOKEN") or None,
        cache_dir=eval_cache_dir(),
    )


def _encode_prompt(tokenizer, prompt: str, device):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def _generate_one(model, tokenizer, prompt: str, hooks: dict[int, object] | None = None) -> str:
    import torch

    device = next(model.parameters()).device
    set_seed(stable_prompt_seed(prompt))
    encoded = _encode_prompt(tokenizer, prompt, device)
    prefill_len = encoded["input_ids"].shape[1]
    for hook in (hooks or {}).values():
        hook.reset(prefill_len)
    with torch.inference_mode():
        generated = model.generate(
            **encoded, do_sample=True, temperature=SAMPLER["temperature"], top_p=SAMPLER["top_p"],
            top_k=SAMPLER["top_k"], repetition_penalty=SAMPLER["repetition_penalty"],
            max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0, prefill_len:], skip_special_tokens=True)


def _runtime_fingerprint(model, tokenizer) -> tuple[dict, str]:
    import torch
    import transformers

    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory": properties.total_memory,
        }
    value = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "model_class": model.__class__.__module__ + "." + model.__class__.__qualname__,
        "tokenizer_class": tokenizer.__class__.__module__ + "." + tokenizer.__class__.__qualname__,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "floating_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
        "sampler": SAMPLER,
    }
    return value, sha256_bytes(canonical_json(value))


def _text_diagnostic(tokenizer, expected: str, actual: str) -> dict:
    expected_ids = list(tokenizer(expected, add_special_tokens=False)["input_ids"])
    actual_ids = list(tokenizer(actual, add_special_tokens=False)["input_ids"])
    first = next((index for index, pair in enumerate(zip(expected_ids, actual_ids)) if pair[0] != pair[1]), None)
    if first is None and len(expected_ids) != len(actual_ids):
        first = min(len(expected_ids), len(actual_ids))
    return {
        "expected_text_sha256": text_sha256(expected),
        "actual_text_sha256": text_sha256(actual),
        "expected_token_ids": expected_ids,
        "actual_token_ids": actual_ids,
        "first_divergence": None if first is None else {
            "token_position": first,
            "expected_token_id": expected_ids[first] if first < len(expected_ids) else None,
            "actual_token_id": actual_ids[first] if first < len(actual_ids) else None,
            "expected_length": len(expected_ids),
            "actual_length": len(actual_ids),
        },
        "exact": expected == actual,
    }


def _load_immutable_runtime_sources(root: Path):
    source_dir = (root / "src").resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import h2_lib
    import smoke_h2

    modules = {"src/h2_lib.py": h2_lib, "src/smoke_h2.py": smoke_h2}
    actual = {}
    for relative, module in modules.items():
        path = Path(module.__file__).resolve()
        if path != (root / relative).resolve():
            raise ValueError(f"immutable runtime module imported from wrong path: {path}")
        actual[relative] = sha256_file(path)
        if actual[relative] != IMMUTABLE_CODE_HASHES[relative]:
            raise ValueError(f"immutable runtime source changed: {relative}")
    return h2_lib, smoke_h2, actual


def runtime_reconstruction_preflight(model, tokenizer, layer_map: dict[int, object],
                                     bases: dict[int, dict[str, np.ndarray]],
                                     source: dict, plan: dict) -> tuple[Path, dict, dict[str, list[str]]]:
    """Gate old/new same-runtime equivalence; record archive replay separately."""
    marker_path = source["geom"] / "controls_reconstruction_preflight.json"
    marker_path.unlink(missing_ok=True)
    _, smoke_h2, immutable_hashes = _load_immutable_runtime_sources(source["root"])
    prompts = [row["prompt"] for row in source["test_jmq"][:RECONSTRUCTION_COUNT]]
    prompt_ids = [str(row["prompt_id"]) for row in source["test_jmq"][:RECONSTRUCTION_COUNT]]
    archived = source["test_frame"].iloc[:RECONSTRUCTION_COUNT]
    old = {
        "ANCHOR-BASE": smoke_h2.generate_batch(
            model, tokenizer, prompts, seed_base=SEED_BASE, max_new=MAX_NEW_TOKENS
        )
    }
    new = {"ANCHOR-BASE": [_generate_one(model, tokenizer, prompt) for prompt in prompts]}
    device = next(model.parameters()).device
    old_hooks = {
        layer: (values["frozen"], values["mu"], float(source["record"]["alpha"]))
        for layer, values in bases.items()
    }
    old["ANCHOR-CONE"] = smoke_h2.generate_with_hooks(
        model, tokenizer, prompts, old_hooks, seed_base=SEED_BASE, max_new=MAX_NEW_TOKENS
    )[0]
    hooks, handles = {}, []
    try:
        for layer, values in bases.items():
            hook = GeneratedTokenHook(
                values["frozen"], values["mu"], float(source["record"]["alpha"]), device
            )
            hooks[layer] = hook
            handles.append(layer_map[layer].register_forward_hook(hook))
        new["ANCHOR-CONE"] = [_generate_one(model, tokenizer, prompt, hooks) for prompt in prompts]
    finally:
        for handle in handles:
            handle.remove()
    equivalence = {
        arm: [
            {"prompt_id": prompt_id, **_text_diagnostic(tokenizer, expected, actual)}
            for prompt_id, expected, actual in zip(prompt_ids, old[arm], new[arm])
        ]
        for arm in ("ANCHOR-BASE", "ANCHOR-CONE")
    }
    gate_passed = all(item["exact"] for values in equivalence.values() for item in values)
    archived_text = {
        "ANCHOR-BASE": archived["baseline"].tolist(),
        "ANCHOR-CONE": archived["ablated"].tolist(),
    }
    archive_replay = {
        arm: [
            {"prompt_id": prompt_id, **_text_diagnostic(tokenizer, expected, actual)}
            for prompt_id, expected, actual in zip(prompt_ids, archived_text[arm], new[arm])
        ]
        for arm in ("ANCHOR-BASE", "ANCHOR-CONE")
    }
    archive_exact = all(item["exact"] for values in archive_replay.values() for item in values)
    fingerprint, fingerprint_hash = _runtime_fingerprint(model, tokenizer)
    source["runtime_fingerprint_sha256"] = fingerprint_hash
    marker = {
        "schema_version": "h2-mrsc-400.reconstruction-preflight.v5", "suite": SUITE,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
        "model": source["model"], "model_revision": MODEL_PINS[source["model"]][1],
        "frozen": source["frozen"]["frozen"], "source_sha256": source["source_sha256"],
        "plan_sha256": plan["plan_sha256"], "basis_sha256": _basis_hash(bases),
        "prompt_ids": prompt_ids, "sampler": SAMPLER,
        "immutable_code_hashes": immutable_hashes,
        "new_code_hashes": {"src/h2_controls.py": sha256_file(Path(__file__).resolve())},
        "runtime_fingerprint": fingerprint,
        "runtime_fingerprint_sha256": fingerprint_hash,
        "same_runtime_equivalence": equivalence,
        "equivalence_status": "pass" if gate_passed else "fail",
        "archive_replay": archive_replay,
        "archive_replay_status": (
            "archive-bit-replay-match" if archive_exact
            else "archive-bit-replay-mismatch-expected-runtime-drift"
        ),
        "archive_replay_is_gate": False,
    }
    atomic_json(marker, marker_path)
    if not gate_passed:
        mismatches = {
            arm: [item["prompt_id"] for item in values if not item["exact"]]
            for arm, values in equivalence.items()
        }
        raise ValueError(f"{source['model']}: old/new same-runtime equivalence failed: {mismatches}")
    return marker_path, marker, new


def calibrate_doses(model, tokenizer, layer_map: dict[int, object], bases: dict[int, dict[str, np.ndarray]],
                    source: dict, plan: dict, path: Path) -> tuple[dict[str, dict[int, float]], dict]:
    basis_sha = _basis_hash(bases)
    progress_path = path.with_name("controls_calibration.progress.json")
    if path.exists():
        saved = read_json(path)
        if (saved.get("schema_version") != "h2-mrsc-400.calibration.v5"
                or saved.get("epoch_fingerprint") != EPOCH_FINGERPRINT
                or saved.get("runtime_fingerprint_sha256") != source["runtime_fingerprint_sha256"]
                or saved.get("plan_sha256") != plan["plan_sha256"]
                or saved.get("source_sha256") != source["source_sha256"]
                or saved.get("basis_sha256") != basis_sha
                or saved.get("prompt_ids") != [str(row["prompt_id"]) for row in source["val1_calibration"]]
                or saved.get("alpha_limit") != ALPHA_LIMIT
                or set(saved.get("alphas", {})) != set(GENERATED_ARMS)):
            raise ValueError(f"{path}: calibration metadata differs; refusing resume")
        alphas = {arm: {int(layer): float(value) for layer, value in values.items()}
                  for arm, values in saved["alphas"].items()}
        if any(set(values) != set(bases) for values in alphas.values()):
            raise ValueError(f"{path}: calibration layers differ; refusing resume")
        if any(not math.isfinite(value) or value <= 0 or value > ALPHA_LIMIT
               for values in alphas.values() for value in values.values()):
            raise ValueError(f"{path}: saved alpha violates (0,{ALPHA_LIMIT}]")
        progress_path.unlink(missing_ok=True)
        return alphas, saved

    device = next(model.parameters()).device
    frozen_alpha = float(source["record"]["alpha"])
    observers, handles = {}, []
    prompt_ids = [str(row["prompt_id"]) for row in source["val1_calibration"]]
    completed = 0
    progress = None
    if progress_path.exists():
        progress = read_json(progress_path)
        if (progress.get("schema_version") != "h2-mrsc-400.calibration-progress.v5"
                or progress.get("epoch_fingerprint") != EPOCH_FINGERPRINT
                or progress.get("runtime_fingerprint_sha256") != source["runtime_fingerprint_sha256"]
                or progress.get("plan_sha256") != plan["plan_sha256"]
                or progress.get("source_sha256") != source["source_sha256"]
                or progress.get("basis_sha256") != basis_sha
                or progress.get("prompt_ids") != prompt_ids):
            raise ValueError(f"{progress_path}: calibration progress differs; refusing resume")
        completed = progress.get("completed_prompts")
        if not isinstance(completed, int) or not 0 <= completed <= CALIBRATION_COUNT:
            raise ValueError(f"{progress_path}: invalid completed prompt count")
    try:
        for layer, values in bases.items():
            observer = DoseObserver(
                {"frozen": values["frozen"],
                 "RAND-RANK-DOSE": values["random"]}, values["mu"], frozen_alpha, device,
            )
            observers[layer] = observer
            handles.append(layer_map[layer].register_forward_hook(observer))
            if progress is not None:
                state = progress.get("observers", {}).get(str(layer), {})
                if (not isinstance(state.get("tokens"), int) or state["tokens"] < 0
                        or set(state.get("sums", {})) != set(observer.sums)
                        or not all(math.isfinite(float(value)) and float(value) >= 0
                                   for value in state["sums"].values())):
                    raise ValueError(f"{progress_path}: invalid observer state for layer {layer}")
                observer.tokens = int(state["tokens"])
                observer.sums = {name: float(value) for name, value in state["sums"].items()}
        for index, row in enumerate(source["val1_calibration"][completed:], completed + 1):
            _generate_one(model, tokenizer, row["prompt"], observers)
            if index % 8 == 0:
                print(f"[{source['model']}] dose calibration {index}/{CALIBRATION_COUNT}", flush=True)
            if index % CHECKPOINT_PROMPTS == 0 or index == CALIBRATION_COUNT:
                atomic_json({
                    "schema_version": "h2-mrsc-400.calibration-progress.v5", "suite": SUITE,
                    "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT,
                    "model": source["model"], "plan_sha256": plan["plan_sha256"],
                    "source_sha256": source["source_sha256"], "basis_sha256": basis_sha,
                    "runtime_fingerprint_sha256": source["runtime_fingerprint_sha256"],
                    "prompt_ids": prompt_ids, "completed_prompts": index,
                    "observers": {str(layer): {"tokens": observer.tokens, "sums": observer.sums}
                                  for layer, observer in observers.items()},
                }, progress_path)
    finally:
        for handle in handles:
            handle.remove()

    per_layer = {}
    for layer, observer in observers.items():
        if observer.tokens <= 0:
            raise ValueError(f"{source['model']} layer {layer}: calibration observed no generated tokens")
        doses = {name: value / observer.tokens for name, value in observer.sums.items()}
        target = doses["frozen"]
        if not math.isfinite(target) or target <= 0:
            raise ValueError(f"{source['model']} layer {layer}: invalid frozen target dose {target}")
        per_layer[layer] = {"tokens": observer.tokens, "target_frozen_dose": target, "unit_doses": doses}
    total_target = sum(record["target_frozen_dose"] for record in per_layer.values())
    alphas = {arm: {} for arm in GENERATED_ARMS}
    layers = {}
    for arm in GENERATED_ARMS:
        total_unit = sum(record["unit_doses"][arm] for record in per_layer.values())
        alpha = total_target / total_unit if total_unit > 0 else float("inf")
        if not math.isfinite(alpha) or alpha <= 0 or alpha > ALPHA_LIMIT:
            raise ValueError(
                f"{source['model']} {arm}: total-dose calibrated alpha {alpha} exceeds "
                f"(0,{ALPHA_LIMIT}]; refusing to clip"
            )
        for layer, record in per_layer.items():
            alphas[arm][layer] = alpha
            layers.setdefault(str(layer), {
                "tokens": record["tokens"], "target_frozen_dose": record["target_frozen_dose"],
                "unit_doses": {other: record["unit_doses"][other] for other in GENERATED_ARMS},
                "alphas": {},
            })["alphas"][arm] = alpha
    saved = {
        "schema_version": "h2-mrsc-400.calibration.v5", "suite": SUITE,
        "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT, "model": source["model"],
        "plan_sha256": plan["plan_sha256"], "source_sha256": source["source_sha256"],
        "runtime_fingerprint_sha256": source["runtime_fingerprint_sha256"],
        "basis_sha256": basis_sha, "prompt_ids": prompt_ids,
        "alpha_limit": ALPHA_LIMIT, "overflow_policy": "fail-not-clip",
        "matching": "total-dose: one alpha per arm matches summed layer doses; same value applied at every layer",
        "total_target_dose": total_target,
        "total_unit_doses": {arm: sum(record["unit_doses"][arm] for record in per_layer.values()) for arm in GENERATED_ARMS},
        "layers": layers,
        "alphas": {arm: {str(layer): value for layer, value in values.items()} for arm, values in alphas.items()},
    }
    atomic_json(saved, path)
    progress_path.unlink(missing_ok=True)
    return alphas, saved


def _validate_generation_frame(frame: pd.DataFrame, source: dict, plan: dict, complete: bool) -> pd.DataFrame:
    required = {
        "schema_version", "suite", "epoch_id", "epoch_fingerprint", "model", "model_revision",
        "runtime_stratum", "runtime_fingerprint_sha256", "split", "arm", "prompt_id", "position",
        "prompt_sha256", "text", "text_sha256", "provenance", "seed", "source_sha256", "plan_sha256",
    }
    _require_columns(frame, required, "control generations")
    frame = frame.copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    expected_meta = {
        "schema_version": SCHEMA, "suite": SUITE, "epoch_id": EPOCH_ID,
        "epoch_fingerprint": EPOCH_FINGERPRINT, "model": source["model"],
        "model_revision": MODEL_PINS[source["model"]][1], "source_sha256": source["source_sha256"],
        "plan_sha256": plan["plan_sha256"],
    }
    for column, value in expected_meta.items():
        if set(frame[column].astype(str)) != {str(value)}:
            raise ValueError(f"control generations: metadata mismatch at {column}")
    if frame.duplicated(["split", "arm", "prompt_id"]).any():
        raise ValueError("control generations: duplicate split/arm/prompt rows")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("control generations: all outputs must be strings")
    if any(text_sha256(row.text) != row.text_sha256 for row in frame.itertuples()):
        raise ValueError("control generations: stale text hash")
    allowed = {("test_jmq", arm) for arm in TEST_ARMS} | {("val2", arm) for arm in VAL2_ARMS}
    if not set(zip(frame["split"], frame["arm"])) <= allowed:
        raise ValueError("control generations: unexpected split/arm")
    split_rows = {"test_jmq": source["test_jmq"], "val2": source["val2"]}
    expected_prompt = {
        (split, str(row["prompt_id"])): (position, text_sha256(row["prompt"]))
        for split, rows in split_rows.items() for position, row in enumerate(rows)
    }
    for row in frame.itertuples():
        key = (row.split, row.prompt_id)
        if key not in expected_prompt or (int(row.position), row.prompt_sha256) != expected_prompt[key]:
            raise ValueError("control generations: prompt position/hash mismatch")
        expected_stratum = ARCHIVED_STRATUM if row.split == "val2" else CURRENT_STRATUM
        expected_runtime = (
            "archived-original-runtime-unavailable" if row.split == "val2"
            else source["runtime_fingerprint_sha256"]
        )
        if row.runtime_stratum != expected_stratum or row.runtime_fingerprint_sha256 != expected_runtime:
            raise ValueError("control generations: runtime stratum/fingerprint mismatch")
        generated = row.split == "test_jmq" and row.arm in TEST_GENERATED_ARMS
        expected_seed = stable_prompt_seed(split_rows[row.split][int(row.position)]["prompt"]) if generated else -1
        if generated:
            expected_provenance = "generated-contemporaneous"
        elif row.arm == "LEX-MATCH":
            expected_provenance = "cpu-lexical"
        elif row.split == "test_jmq" and row.arm == "HUMAN":
            expected_provenance = "archived-human"
        else:
            expected_provenance = "archived-core"
        if int(row.seed) != expected_seed or row.provenance != expected_provenance:
            raise ValueError("control generations: seed/provenance mismatch")
    if complete:
        expected = {("test_jmq", arm, str(row["prompt_id"])) for arm in TEST_ARMS for row in source["test_jmq"]}
        expected |= {("val2", arm, str(row["prompt_id"])) for arm in VAL2_ARMS for row in source["val2"]}
        actual = set(zip(frame["split"], frame["arm"], frame["prompt_id"]))
        if actual != expected or len(frame) != len(expected):
            raise ValueError("control generations: final artifact is partial or has unexpected rows")
    return frame


PRIOR_SCHEMAS = {
    "generations": "h2-mrsc-400.generations.v4",
    "val1": "h2-mrsc-400.val1-anchors.v4",
}
PRIOR_STRATUM = "contemporaneous-v4-runtime"


def _adopt_prior_rows(frame: pd.DataFrame, source: dict, plan: dict) -> pd.DataFrame:
    """Adopt verified prior-epoch (v4) rows into the current epoch.

    Only the blocked v4 epoch qualifies, only without LEX rows (LEX is
    applied post-hoc and can never pre-exist progress), and every text hash,
    prompt mapping, seed, and count is re-verified by the current validator
    afterwards. Text content is never modified, only epoch bookkeeping.
    """
    frame = frame.copy()
    if set(frame["suite"].astype(str)) != {PRIOR_SUITE}:
        raise ValueError("adoption requires exactly the prior blocked epoch")
    if set(frame["model"].astype(str)) != {source["model"]}:
        raise ValueError("adoption model mismatch")
    if "LEX-MATCH" in set(frame["arm"].astype(str)):
        raise ValueError("prior progress must not contain LEX rows")
    if "runtime_fingerprint_sha256" not in source:
        raise ValueError("adoption requires a fresh reconstruction fingerprint")
    frame["schema_version"] = VAL1_SCHEMA if set(frame["split"].astype(str)) == {"val1"} else SCHEMA
    frame["suite"] = SUITE
    frame["epoch_id"] = EPOCH_ID
    frame["epoch_fingerprint"] = EPOCH_FINGERPRINT
    frame["source_sha256"] = source["source_sha256"]
    frame["plan_sha256"] = plan["plan_sha256"]
    mask = frame["split"] != "val2"
    frame.loc[mask, "runtime_stratum"] = CURRENT_STRATUM
    frame.loc[mask, "runtime_fingerprint_sha256"] = source["runtime_fingerprint_sha256"]
    return frame


def _merge_fixed_rows(frame: pd.DataFrame | None, fixed: list[dict], source: dict, plan: dict) -> pd.DataFrame:
    fixed_frame = pd.DataFrame(fixed)
    if frame is None:
        return fixed_frame
    if len(frame) and set(frame["suite"].astype(str)) == {PRIOR_SUITE}:
        frame = _adopt_prior_rows(frame, source, plan)
    frame = _validate_generation_frame(frame, source, plan, complete=False)
    fixed_keys = set(zip(fixed_frame["split"], fixed_frame["arm"], fixed_frame["prompt_id"].astype(str)))
    existing_fixed = frame[[key in fixed_keys for key in zip(frame["split"], frame["arm"], frame["prompt_id"])]].copy()
    expected_lookup = {(row.split, row.arm, str(row.prompt_id)): row.text_sha256 for row in fixed_frame.itertuples()}
    actual_lookup = {(row.split, row.arm, str(row.prompt_id)): row.text_sha256 for row in existing_fixed.itertuples()}
    if actual_lookup and actual_lookup != expected_lookup:
        raise ValueError("control progress: archived/CPU rows differ from the current immutable plan")
    generated = frame[~frame.apply(lambda row: (row["split"], row["arm"], row["prompt_id"]) in fixed_keys, axis=1)]
    return pd.concat([fixed_frame, generated], ignore_index=True)


def _merge_lex_rows(frame: pd.DataFrame, lex_rows: list[dict], source: dict, plan: dict) -> pd.DataFrame:
    """Append CPU LEX-MATCH rows to validated progress.

    LEX rows are produced after TEST generation and therefore can never
    pre-exist progress; merging them through _merge_fixed_rows would always
    fail (fixed-lookup mismatch). They are validated as new rows instead.
    """
    frame = _validate_generation_frame(frame, source, plan, complete=False)
    lex_frame = pd.DataFrame(lex_rows)
    expected_ids = [str(row["prompt_id"]) for row in source["test_jmq"]]
    if (len(lex_frame) != len(expected_ids)
            or set(lex_frame["arm"].astype(str)) != {"LEX-MATCH"}
            or set(lex_frame["prompt_id"].astype(str)) != set(expected_ids)):
        raise ValueError("LEX rows must be exactly one complete TEST-JMQ arm")
    existing = set(zip(frame["split"].astype(str), frame["prompt_id"].astype(str), frame["arm"].astype(str)))
    if any((str(row.split), str(row.prompt_id), "LEX-MATCH") in existing for row in lex_frame.itertuples()):
        raise ValueError("LEX rows already present; refusing duplicate merge")
    return _validate_generation_frame(
        pd.concat([frame, lex_frame], ignore_index=True), source, plan, complete=False
    )


def _sort_output(frame: pd.DataFrame) -> pd.DataFrame:
    split_order = {"test_jmq": 0, "val2": 1}
    arm_order = {arm: index for index, arm in enumerate(TEST_ARMS + VAL2_ARMS)}
    frame = frame.copy()
    frame["_split"] = frame["split"].map(split_order)
    frame["_arm"] = frame["arm"].map(arm_order)
    return frame.sort_values(["_split", "_arm", "position"]).drop(columns=["_split", "_arm"]).reset_index(drop=True)


def run_model(root: Path, model_code: str, preflight_only: bool = False) -> None:
    source = prepare_sources(root, model_code)
    plan = make_plan(source)
    geom = source["geom"]
    plan_path = geom / "controls_plan.json"
    ensure_plan(plan_path, plan, write=not preflight_only)
    print(f"[{model_code}] archived-output preflight passed source={source['source_sha256'][:12]} "
          f"plan={plan['plan_sha256'][:12]}", flush=True)
    if preflight_only:
        return

    output_path = geom / "controls_generations.parquet"
    done_path = geom / "control_generation_done.json"
    if output_path.exists() and done_path.exists():
        done = read_json(done_path)
        calibration_path = geom / "controls_calibration.json"
        reconstruction_path = geom / "controls_reconstruction_preflight.json"
        val1_path = geom / "controls_val1_anchors.parquet"
        val1_done_path = geom / "control_val1_anchors_done.json"
        lexical_policy_path = geom / "controls_lexical_policy.json"
        reconstruction = read_json(reconstruction_path)
        if (reconstruction.get("equivalence_status") != "pass"
                or reconstruction.get("epoch_fingerprint") != EPOCH_FINGERPRINT
                or reconstruction.get("plan_sha256") != plan["plan_sha256"]):
            raise ValueError(f"{model_code}: equivalence gate marker is invalid")
        source["runtime_fingerprint_sha256"] = reconstruction["runtime_fingerprint_sha256"]
        final = _validate_generation_frame(pd.read_parquet(output_path), source, plan, complete=True)
        if (done.get("schema_version") != "h2-mrsc-400.generation-done.v5"
                or done.get("epoch_fingerprint") != EPOCH_FINGERPRINT
                or done.get("plan_sha256") != plan["plan_sha256"]
                or done.get("source_sha256") != source["source_sha256"]
                or done.get("generations_sha256") != sha256_file(output_path)
                or not calibration_path.is_file()
                or done.get("calibration_sha256") != sha256_file(calibration_path)
                or not reconstruction_path.is_file()
                or done.get("reconstruction_preflight_sha256") != sha256_file(reconstruction_path)
                or not val1_path.is_file() or not val1_done_path.is_file()
                or done.get("val1_anchors_sha256") != sha256_file(val1_path)
                or done.get("val1_anchors_done_sha256") != sha256_file(val1_done_path)
                or not lexical_policy_path.is_file()
                or done.get("lexical_policy_sha256") != sha256_file(lexical_policy_path)
                or done.get("runtime_fingerprint_sha256") != source["runtime_fingerprint_sha256"]
                or done.get("equivalence_status") != "pass"
                or done.get("rows") != len(final)):
            raise ValueError(f"{model_code}: generation completion marker is stale")
        print(f"[{model_code}] reuse complete controls ({len(final)} rows)", flush=True)
        return
    if output_path.exists() != done_path.exists():
        raise ValueError(f"{model_code}: final control artifact/marker is incomplete; refusing overwrite")

    tokenizer, model, layer_map = load_target(model_code)
    try:
        bases = build_control_bases(model, tokenizer, source, layer_map)
        reconstruction_path, reconstruction, _ = runtime_reconstruction_preflight(
            model, tokenizer, layer_map, bases, source, plan
        )
        reconstruction_hash = sha256_file(reconstruction_path)
        calibration_path = geom / "controls_calibration.json"
        alphas, calibration = calibrate_doses(model, tokenizer, layer_map, bases, source, plan, calibration_path)
        val1_anchors, val1_path, val1_done_path = generate_val1_anchors(
            model, tokenizer, layer_map, bases, source, plan, reconstruction_hash
        )
        eval_tokenizer = load_eval_tokenizer()
        lexical_policy, lexical_policy_path = calibrate_lex_policy(
            eval_tokenizer, source, plan, val1_anchors, sha256_file(val1_path)
        )
        fixed = archived_rows(source, plan)

        progress_path = geom / "controls_generations.progress.parquet"
        progress = pd.read_parquet(progress_path) if progress_path.exists() else None
        frame = _merge_fixed_rows(progress, fixed, source, plan)
        atomic_parquet(_sort_output(frame), progress_path)

        prompts = [row["prompt"] for row in source["test_jmq"]]
        ids = [str(row["prompt_id"]) for row in source["test_jmq"]]
        device = next(model.parameters()).device
        for arm in TEST_GENERATED_ARMS:
            hook_objects, handles = {}, []
            try:
                if arm != "ANCHOR-BASE":
                    for layer, values in bases.items():
                        if arm == "ANCHOR-CONE":
                            basis, alpha = values["frozen"], float(source["record"]["alpha"])
                        else:
                            basis, alpha = values["random"], alphas[arm][layer]
                        hook = GeneratedTokenHook(basis, values["mu"], alpha, device)
                        hook_objects[layer] = hook
                        handles.append(layer_map[layer].register_forward_hook(hook))
                existing = set(frame.loc[(frame["split"] == "test_jmq") & (frame["arm"] == arm), "prompt_id"])
                todo = [index for index, prompt_id in enumerate(ids) if prompt_id not in existing]
                for start in range(0, len(todo), CHECKPOINT_PROMPTS):
                    indices = todo[start:start + CHECKPOINT_PROMPTS]
                    new_rows = []
                    for index in indices:
                        text = _generate_one(model, tokenizer, prompts[index], hook_objects or None)
                        new_rows.append(_row(
                            source, plan, "test_jmq", arm, index, ids[index], prompts[index], text,
                            "generated-contemporaneous", stable_prompt_seed(prompts[index]),
                        ))
                    frame = pd.concat([frame, pd.DataFrame(new_rows)], ignore_index=True)
                    frame = _validate_generation_frame(frame, source, plan, complete=False)
                    atomic_parquet(_sort_output(frame), progress_path)
                    print(f"[{model_code}] {arm} {len(existing) + start + len(indices)}/{CONTROL_COUNT}", flush=True)
            finally:
                for handle in handles:
                    handle.remove()

        lex_rows, lexical_application = apply_lex_policy(
            eval_tokenizer, source, plan, lexical_policy, frame
        )
        frame = _merge_lex_rows(frame, lex_rows, source, plan)
        atomic_parquet(_sort_output(frame), progress_path)
        calibration = dict(
            calibration,
            lexical_policy_sha256=sha256_file(lexical_policy_path),
            lexical_application=lexical_application,
        )
        atomic_json(calibration, calibration_path)
        frame = _sort_output(_validate_generation_frame(frame, source, plan, complete=True))
        atomic_parquet(frame, output_path)
        _validate_generation_frame(pd.read_parquet(output_path), source, plan, complete=True)
        generation_hash = sha256_file(output_path)
        atomic_json({
            "schema_version": "h2-mrsc-400.generation-done.v5", "suite": SUITE,
            "epoch_id": EPOCH_ID, "epoch_fingerprint": EPOCH_FINGERPRINT, "model": model_code,
            "model_revision": MODEL_PINS[model_code][1], "frozen": source["frozen"]["frozen"],
            "source_sha256": source["source_sha256"], "plan_sha256": plan["plan_sha256"],
            "generations_sha256": generation_hash, "calibration_sha256": sha256_file(calibration_path),
            "reconstruction_preflight_sha256": reconstruction_hash,
            "equivalence_status": reconstruction["equivalence_status"],
            "archive_replay_status": reconstruction["archive_replay_status"],
            "runtime_fingerprint_sha256": source["runtime_fingerprint_sha256"],
            "val1_anchors_sha256": sha256_file(val1_path),
            "val1_anchors_done_sha256": sha256_file(val1_done_path),
            "lexical_policy_sha256": sha256_file(lexical_policy_path),
            "rows": len(frame),
            "counts": {
                "test_jmq_per_arm": CONTROL_COUNT, "test_jmq_rows": len(TEST_ARMS) * CONTROL_COUNT,
                "val1_per_arm": LEX_CALIBRATION_COUNT, "val1_rows": len(VAL1_ARMS) * LEX_CALIBRATION_COUNT,
                "val2_per_arm": VAL2_COUNT, "val2_rows": len(VAL2_ARMS) * VAL2_COUNT,
            },
            "generated_test_arms": list(TEST_GENERATED_ARMS),
            "runtime_strata": {"test_jmq": CURRENT_STRATUM, "val2": ARCHIVED_STRATUM},
            "budget_gpu_hours": {"nominal": [45, 51], "hard_cap": 72},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }, done_path)
        progress_path.unlink(missing_ok=True)
        print(f"[{model_code}] control generation complete sha256={generation_hash}", flush=True)
    finally:
        del model
        import torch
        torch.cuda.empty_cache()


class _TinyTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for token in text.split():
            if token not in self.vocab:
                index = len(self.vocab) + 1
                self.vocab[token] = index
                self.reverse[index] = token
            ids.append(self.vocab[token])
        return {"input_ids": ids}

    def decode(self, ids, **_kwargs):
        return " ".join(self.reverse[index] for index in ids)


def self_test() -> None:
    import torch

    assert stable_prompt_seed("x") == (SEED_BASE + int(hashlib.sha256(b"x").hexdigest()[:8], 16)) % SEED_MODULUS
    expected_random = (
        17011 + int(hashlib.sha256(b"H2-MRSC|RAND|L1|8").hexdigest()[:8], 16)
    ) % 2147483647
    assert random_basis_seed("L1", 8) == expected_random
    assert random_basis_seed("L1", 8) != random_basis_seed("L1", 9)

    hook = GeneratedTokenHook(np.eye(2, 1), np.zeros(2), 1.0, torch.device("cpu"))
    hook.reset(1)
    prefill = torch.tensor([[[2.0, 3.0]]], dtype=torch.bfloat16)
    assert torch.equal(hook(None, None, prefill), prefill), "a one-token prefill was modified"
    decoded = hook(None, None, prefill)
    assert decoded[0, 0, 0] == 0 and decoded[0, 0, 1] == 3, "generated token was not cone-ablated"

    source = {
        "model": "L1", "source_sha256": "a" * 64,
        "runtime_fingerprint_sha256": "d" * 64,
        "test_frame": pd.DataFrame({"baseline": ["red red blue", "red green"], "ablated": ["red blue", "green"]}),
        "test_jmq": [
            {"prompt_id": "1", "prompt": "p1", "human_text": "blue green"},
            {"prompt_id": "2", "prompt": "p2", "human_text": "green blue"},
        ],
        "s2_config": "W-X-r2-a0.5-rho1.0",
        "s2_frozen": pd.DataFrame({
            "baseline": ["red red blue", "red green"], "ablated": ["red blue", "green"]
        }),
        "val1_lexical": [
            {"prompt_id": "v1", "prompt": "vp1", "human_text": "blue green"},
            {"prompt_id": "v2", "prompt": "vp2", "human_text": "green blue"},
        ],
    }
    plan = {"plan_sha256": "b" * 64}
    tok = _TinyTokenizer()
    left_ids = [tok(text, add_special_tokens=False)["input_ids"] for text in ["red red blue", "red green"]]
    right_ids = [tok(text, add_special_tokens=False)["input_ids"] for text in ["blue green", "green blue"]]
    assert _corpus_l2(left_ids, right_ids) >= 0.0
    filtered, deleted = _lex_delete(left_ids, ["1", "2"], {tok("red", add_special_tokens=False)["input_ids"][0]}, 1.0, "L1")
    assert deleted >= 0
    report = _shift_report(0.5, 0.4, 0.42)
    assert report["same_direction"] in (True, False) and "matched" in report

    with tempfile.TemporaryDirectory(prefix="h2-controls-self-test-") as directory:
        path = Path(directory) / "artifact.parquet"
        atomic_parquet(pd.DataFrame([{"a": 1}, {"a": 2}]), path)
        assert len(pd.read_parquet(path)) == 2
        plan_path = Path(directory) / "plan.json"
        ensure_plan(plan_path, plan, write=True)
        try:
            ensure_plan(plan_path, {"plan_sha256": "c" * 64}, write=True)
            raise AssertionError("mismatched plan was accepted")
        except ValueError:
            pass

    merge_source = {
        "model": "L1", "source_sha256": "e" * 64,
        "runtime_fingerprint_sha256": "d" * 64,
        "test_jmq": [
            {"prompt_id": "t1", "prompt": "write one", "human_text": "human one"},
            {"prompt_id": "t2", "prompt": "write two", "human_text": "human two"},
        ],
        "val2": [
            {"prompt_id": "v1", "prompt": "val one", "human_text": "human val"},
        ],
    }
    merge_plan = {"plan_sha256": "f" * 64}
    prog_rows = [
        _row(merge_source, merge_plan, "test_jmq", "HUMAN", 0, "t1", "write one", "human one", "archived-human"),
        _row(merge_source, merge_plan, "test_jmq", "HUMAN", 1, "t2", "write two", "human two", "archived-human"),
        _row(merge_source, merge_plan, "test_jmq", "ANCHOR-BASE", 0, "t1", "write one", "base one",
             "generated-contemporaneous", stable_prompt_seed("write one")),
        _row(merge_source, merge_plan, "test_jmq", "ANCHOR-BASE", 1, "t2", "write two", "base two",
             "generated-contemporaneous", stable_prompt_seed("write two")),
    ]
    prog = pd.DataFrame(prog_rows)
    lex_rows = [
        _row(merge_source, merge_plan, "test_jmq", "LEX-MATCH", 0, "t1", "write one", "lex one", "cpu-lexical"),
        _row(merge_source, merge_plan, "test_jmq", "LEX-MATCH", 1, "t2", "write two", "lex two", "cpu-lexical"),
    ]
    merged = _merge_lex_rows(prog, lex_rows, merge_source, merge_plan)
    assert len(merged) == 6 and set(merged["arm"]) == {"HUMAN", "ANCHOR-BASE", "LEX-MATCH"}
    try:
        _merge_lex_rows(merged, lex_rows, merge_source, merge_plan)
        raise AssertionError("duplicate LEX merge was accepted")
    except ValueError:
        pass
    try:
        human_fixed = [
            _row(merge_source, merge_plan, "test_jmq", "HUMAN", 0, "t1", "write one", "human one", "archived-human"),
            _row(merge_source, merge_plan, "test_jmq", "HUMAN", 1, "t2", "write two", "human two", "archived-human"),
        ]
        _merge_fixed_rows(prog, human_fixed + lex_rows, merge_source, merge_plan)
        raise AssertionError("fixed-merge with post-hoc rows was accepted")
    except ValueError:
        pass
    print("H2 controls synthetic self-test passed (CPU only)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight", action="store_true", help="validate archives and plans without loading models")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.models:
        parser.error("--models is required unless --self-test is used")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    for model_code in args.models:
        run_model(args.root, model_code, args.preflight)


if __name__ == "__main__":
    main()
