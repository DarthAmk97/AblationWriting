"""Embed frozen H2 VAL2/TEST documents for MMD scoring.

Only the exact frozen S3 cone row is accepted for VAL2. TEST additionally
requires its completion marker. Each model/split artifact is replaced
atomically and reused only while its source hashes and embedding pin match.

Usage: python src/mmd_embed.py --models O1 L1 --splits val2 test
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


EMBEDDER = "nvidia/llama-embed-nemotron-8b"
REVISION = "aa3b43a495a9b280d1bdb716da37c54bb495d630"
EXPECTED_COUNTS = {"val2": 512, "test": 2000}
SOURCES = ("human", "baseline", "ablated")
POOLING = "attention-mask mean in FP32, then L2 normalize"
ARTIFACT_COLUMNS = {
    "prompt_id", "source", "text_sha256", "n_tokens", "truncated", "embedding",
    "embedder", "revision", "split", "frozen", "max_length", "pooling",
}


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_parquet(frame, path):
    """Write beside the destination, fsync, then atomically replace it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    tmp = Path(handle.name)
    handle.close()
    try:
        frame.to_parquet(tmp, index=False)
        with open(tmp, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    tmp = Path(handle.name)
    try:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(tmp, path)
    finally:
        if not handle.closed:
            handle.close()
        tmp.unlink(missing_ok=True)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _split_rows(root, split):
    path = root / "data" / "splits" / f"{split}.jsonl"
    if not path.is_file():
        raise ValueError(f"missing frozen split: {path}")
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    expected = EXPECTED_COUNTS[split]
    if len(rows) != expected:
        raise ValueError(f"{split}: expected exactly {expected} source rows, found {len(rows)}")
    ids = [str(row.get("prompt_id")) for row in rows]
    if len(set(ids)) != expected or "None" in ids:
        raise ValueError(f"{split}: prompt_id values must be present and unique")
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("human_text"), str):
            raise ValueError(f"{split}: prompt and human_text must be strings")
    return rows


def load_frozen(root, model):
    path = root / "artifacts" / "geometry" / "H2" / model / "frozen.json"
    frozen = _read_json(path).get("frozen")
    prefix = f"H2-full-s3-{model}-"
    if not isinstance(frozen, str) or not frozen.startswith(prefix) or not frozen.endswith("-cone"):
        raise ValueError(f"{model}: frozen.json does not name a frozen S3 cone configuration")
    return frozen, frozen[len(prefix):]


def _check_text_frame(frame, ids, columns, label):
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    if len(frame) != len(ids):
        raise ValueError(f"{label}: expected {len(ids)} rows, found {len(frame)}")
    if frame["prompt_id"].duplicated().any() or set(frame["prompt_id"]) != set(ids):
        raise ValueError(f"{label}: prompt IDs do not exactly match the frozen split")
    for column in columns - {"prompt_id"}:
        if frame[column].isna().any() or not frame[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{label}: {column} must contain only strings")
    return frame.set_index("prompt_id").loc[ids]


def collect_documents(root, model, split):
    """Return ordered document records after enforcing the frozen source contract."""
    root = Path(root)
    split = split.lower()
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"unsupported split: {split}")
    rows = _split_rows(root, split)
    ids = [str(row["prompt_id"]) for row in rows]
    frozen, config = load_frozen(root, model)
    geom = root / "artifacts" / "geometry" / "H2" / model

    if split == "val2":
        path = geom / "s3_generations.parquet"
        if not path.is_file():
            raise ValueError(f"{model}: missing frozen S3 generations: {path}")
        frame = pd.read_parquet(path)
        required = {"prompt_id", "baseline", "ablated", "config"}
        if not required <= set(frame.columns):
            raise ValueError(f"{model} VAL2: missing columns {sorted(required - set(frame.columns))}")
        selected = frame[frame["config"].astype(str) == config]
        selected = _check_text_frame(selected, ids, required, f"{model} VAL2 frozen S3")
        texts = [
            (row["human_text"], selected.loc[pid, "baseline"], selected.loc[pid, "ablated"])
            for row, pid in zip(rows, ids)
        ]
    else:
        done_path = geom / "test_done"
        if not done_path.is_file():
            raise ValueError(f"{model}: TEST is incomplete; missing {done_path}")
        done = _read_json(done_path)
        if done.get("n") != EXPECTED_COUNTS["test"] or done.get("frozen") != frozen:
            raise ValueError(f"{model}: test_done does not match the exact TEST count and freeze")
        path = geom / "test_generations.parquet"
        if not path.is_file():
            raise ValueError(f"{model}: missing completed TEST generations: {path}")
        required = {"prompt_id", "prompt", "human", "baseline", "ablated"}
        frame = _check_text_frame(pd.read_parquet(path), ids, required, f"{model} TEST")
        texts = []
        for row, pid in zip(rows, ids):
            if frame.loc[pid, "prompt"] != row["prompt"] or frame.loc[pid, "human"] != row["human_text"]:
                raise ValueError(f"{model} TEST: source mismatch at prompt_id {pid}")
            texts.append((row["human_text"], frame.loc[pid, "baseline"], frame.loc[pid, "ablated"]))

    documents = []
    for pid, values in zip(ids, texts):
        for source, text in zip(SOURCES, values):
            documents.append({
                "prompt_id": pid,
                "source": source,
                "text": text,
                "text_sha256": text_sha256(text),
            })
    return documents, frozen


def validate_embedding_artifact(path, documents, model, split, frozen, max_length=None):
    """Load and validate an embedding artifact; raise on any stale/partial content."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing embedding artifact: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"unreadable embedding artifact {path}: {exc}") from exc
    missing = ARTIFACT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if len(frame) != len(documents):
        raise ValueError(f"{path}: expected {len(documents)} rows, found {len(frame)}")
    expected_meta = {
        "embedder": EMBEDDER, "revision": REVISION, "split": split,
        "frozen": frozen, "pooling": POOLING,
    }
    for column, expected in expected_meta.items():
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"{path}: wrong {column} metadata")
    lengths = set(frame["max_length"].astype(int))
    if len(lengths) != 1 or next(iter(lengths)) <= 0 or (max_length is not None and lengths != {max_length}):
        raise ValueError(f"{path}: wrong max_length metadata")
    frame = frame.copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    frame["source"] = frame["source"].astype(str)
    if frame.duplicated(["prompt_id", "source"]).any():
        raise ValueError(f"{path}: duplicate prompt/source rows")
    expected = {(doc["prompt_id"], doc["source"]): doc["text_sha256"] for doc in documents}
    actual = {(row.prompt_id, row.source): row.text_sha256 for row in frame.itertuples()}
    if actual != expected:
        raise ValueError(f"{path}: source keys or text hashes are stale")
    if frame["n_tokens"].isna().any() or (frame["n_tokens"].astype(int) <= 0).any():
        raise ValueError(f"{path}: invalid token counts")
    if not pd.api.types.is_bool_dtype(frame["truncated"]):
        raise ValueError(f"{path}: truncated must be boolean")
    vectors = [np.asarray(value, dtype=np.float32) for value in frame["embedding"]]
    dimensions = {vector.shape for vector in vectors}
    if len(dimensions) != 1 or not dimensions or len(next(iter(dimensions))) != 1:
        raise ValueError(f"{path}: embeddings must have one consistent vector shape")
    matrix = np.stack(vectors)
    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{path}: embeddings must be finite and non-empty")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2e-4, rtol=0):
        raise ValueError(f"{path}: embeddings are not L2-normalized")
    return frame


def _load_embedder(device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDER, revision=REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("embedder tokenizer has neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        EMBEDDER, revision=REVISION, dtype=dtype, trust_remote_code=True
    )
    model.to(device).eval()
    return tokenizer, model, device


def _embed_documents(documents, tokenizer, model, device, batch_size, max_length):
    import torch

    output_rows = []
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start:start + batch_size]
        texts = [doc["text"] for doc in batch_docs]
        token_counts = [
            len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            for text in texts
        ]
        encoded = tokenizer(
            texts, add_special_tokens=True, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            denominator = mask.sum(dim=1)
            if (denominator == 0).any():
                raise ValueError("embedder produced an empty attention mask")
            pooled = (hidden * mask).sum(dim=1) / denominator
            norms = pooled.norm(p=2, dim=1, keepdim=True)
            if (norms <= 0).any() or not torch.isfinite(pooled).all():
                raise ValueError("embedder produced invalid pooled vectors")
            pooled = (pooled / norms).cpu().numpy().astype(np.float32)
        used_counts = inputs["attention_mask"].sum(dim=1).cpu().tolist()
        for doc, vector, n_tokens, used in zip(batch_docs, pooled, token_counts, used_counts):
            output_rows.append({
                "prompt_id": doc["prompt_id"],
                "source": doc["source"],
                "text_sha256": doc["text_sha256"],
                "n_tokens": n_tokens,
                "truncated": bool(n_tokens > used),
                "embedding": vector,
            })
    return output_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--splits", nargs="+", required=True, type=str.lower, choices=sorted(EXPECTED_COUNTS)
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight", action="store_true", help="validate inputs without loading the embedder")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive")
    if len(set(args.models)) != len(args.models) or len(set(args.splits)) != len(args.splits):
        parser.error("models and splits must not contain duplicates")

    jobs = []
    for model in args.models:
        for split in args.splits:
            documents, frozen = collect_documents(args.root, model, split)
            path = args.root / "artifacts" / "geometry" / "H2" / model / f"mmd_embeddings_{split}.parquet"
            try:
                validate_embedding_artifact(path, documents, model, split, frozen, args.max_length)
                print(f"[{model}] reuse {split} embeddings ({len(documents)} documents)", flush=True)
            except ValueError as exc:
                print(f"[{model}] build {split} embeddings: {exc}", flush=True)
                jobs.append((model, split, documents, frozen, path))
    if not jobs:
        return
    if args.preflight:
        print(f"PREFLIGHT passed: {len(jobs)} embedding artifacts pending", flush=True)
        return

    tokenizer, embedder, device = _load_embedder(args.device)  # exactly one model load per invocation
    for model, split, documents, frozen, path in jobs:
        rows = _embed_documents(
            documents, tokenizer, embedder, device, args.batch_size, args.max_length
        )
        for row in rows:
            row.update({
                "embedder": EMBEDDER,
                "revision": REVISION,
                "split": split,
                "frozen": frozen,
                "max_length": args.max_length,
                "pooling": POOLING,
            })
        atomic_write_parquet(pd.DataFrame(rows), path)
        validate_embedding_artifact(path, documents, model, split, frozen, args.max_length)
        print(f"[{model}] wrote {path} ({len(rows)} documents)", flush=True)


if __name__ == "__main__":
    main()
