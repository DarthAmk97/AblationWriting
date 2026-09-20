"""Build the global evidence registry: every paper-relevant artifact, asset, and
reproducibility record in one hash-verified manifest.

Covers the project root, the verified dump_mirror snapshot, and (optionally)
a remote VAST inventory. Excludes caches, env internals, transport copies
(hf_outgoing content, hf_incoming pulls — only their manifests are kept),
and operational loop logs.

Usage:
  python src/build_evidence_registry.py [--remote-inventory PATH]
Re-run after any new finals land; outputs evidence_registry.json +
EVIDENCE_REGISTRY.md (both registered as authoritative local-ahead files).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HF_COMMITS = {
    "9da4909fc4c1ded93b929e336f14aa421ab4931f": "MMD 10 embeddings + 5 metric/done pairs",
    "ab7d386a5e046fb052f5ad365660ec10b83b05dd": "MMD-integrated downstream bundle (tables/figures/exports)",
    "5e93418e31e4bb15a9d1c671784226b927b373cf": "H2-MRSC controls freeze (v1 epoch)",
    "362e5ff4b49056b67e56e4124d6115abd9770105": "H2-MRSC v2 contemporaneous-anchor epoch",
    "375a32edc81ea4d54229b7a2599a30035d5b1d6e": "H2-MRSC v3 total-dose epoch",
    "b73159ac5d92a9ce4c7d497602a868021c810af3": "H2-MRSC v4 RAND-only epoch",
    "b58cf66371431591138a857e9ce6ef92a8feefe4": "H2-MRSC v5 lex-merge fix + v4 adoption",
    "48bd29ba8446e2821614e46170a7349845c1f550": "paired MMD/JMQ/L2 table + figure",
    "91c34c3b0b5d463d1a9571551baa43c15c388275": "grouped distribution-shift table + figure",
    "f47a4292642670b69b000f17864e09f062720797": "qualitative audit (domain/style/truncation/themes/probe)",
    "75b4d51f09721488c961b99d93060966caf55c47": "refusal-passenger paragraph + evidence registry update",
    "82c6ed83f0cebb5a9ed160bf63c91aef2154d1a3": "controls narrative (PC1 infeasibility)",
}

EXCLUDE_DIRS = {".hf_cache", "__pycache__", ".git", ".venv", "node_modules",
                ".opencode", ".ruff_cache"}


def category(rel: str) -> str:
    p = rel.replace("\\", "/")
    if p.startswith("dump_mirror/data/splits"):
        return "splits"
    if p.startswith("dump_mirror/artifacts/geometry"):
        return "geometry-core"
    if p.startswith("dump_mirror/judging/inputs"):
        return "judge-feed"
    if p.startswith("dump_mirror/"):
        return "mirror-core"
    if p.startswith("artifacts/geometry/H2"):
        return "geometry-H2"
    if p.startswith("data/splits"):
        return "splits"
    if p.startswith("judgments/"):
        return "judgments"
    if p.startswith("judging/"):
        return "judge-feed"
    if p.startswith("metrics/tables"):
        return "tables"
    if p.startswith("metrics/"):
        return "metrics"
    if p.startswith("figures/data"):
        return "figure-data"
    if p.startswith("figures/"):
        return "figures"
    if p.startswith("exports/"):
        return "exports"
    if p.startswith("paper/"):
        return "paper"
    if p.startswith("src/"):
        return "code"
    if p.startswith("hf_incoming/") or p.startswith("hf_outgoing/"):
        return "transport"
    if p.startswith("logs/"):
        return "logs"
    if p.endswith(".py") and "/" not in p:
        return "code"
    if p in {"manifest.yaml", "claims_ledger.md", "state_ledger.md", "failures.md",
             "lab_notebook.md", "compute_budget.md", "DUMP_SPEC.md",
             "CONTROL_PROTOCOL.md", "LOCAL_SYNC_POLICY.md",
             "loop-ticker-prompt.md", ".local_ahead.json"}:
        return "provenance"
    return "other"


def status_of(rel: str) -> str:
    p = rel.replace("\\", "/")
    if ".blocked-" in p or ".v1-blocked-" in p:
        return "superseded"
    if ".progress." in p or p.endswith(".tmp") or p.endswith(".lock") or p.endswith(".pid"):
        return "transient"
    if p.startswith("hf_outgoing/") and "/manifests/" not in p and "STAGE_MANIFEST" not in p:
        return "transport-copy"
    if p.startswith("hf_incoming/") and "STAGE_MANIFEST" not in p:
        return "transport-copy"
    return "final"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_local() -> list[dict]:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        parts = set(path.relative_to(ROOT).parts)
        if parts & EXCLUDE_DIRS or path.suffix == ".pyc":
            continue
        if parts & {"hf_incoming", "hf_outgoing"} and not (
                "STAGE_MANIFEST" in path.name or "/manifests/" in rel):
            continue
        if path.name.startswith(".") and path.name != ".local_ahead.json":
            continue
        entries.append({"path": rel, "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "category": category(rel), "status": status_of(rel),
                        "location": "local"})
    return entries


def collect_remote(path: Path) -> list[dict]:
    entries = []
    text = path.read_text(encoding="utf-16" if path.read_bytes()[:2] == b"\xff\xfe" else "utf-8")
    for line in text.splitlines():
        if line.count("|") < 2:
            continue
        rel, size, _ = line.rsplit("|", 2)
        rel = rel.strip().lstrip("./")
        if not rel or ".hf_cache" in rel or ".cache/huggingface" in rel:
            continue
        entries.append({"path": "vast:/root/AblationWriting/" + rel,
                        "bytes": int(float(size)), "sha256": None,
                        "category": category(rel), "status": status_of(rel),
                        "location": "vast"})
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remote-inventory", type=Path, default=None)
    args = ap.parse_args()
    entries = collect_local()
    if args.remote_inventory and args.remote_inventory.is_file():
        entries.extend(collect_remote(args.remote_inventory))
    by_cat: dict[str, list[dict]] = {}
    for entry in entries:
        by_cat.setdefault(entry["category"], []).append(entry)
    created = datetime.datetime.now(datetime.timezone.utc).isoformat()
    registry = {"created_utc": created,
                "scope": "AblationWriting paper bundle: root + dump_mirror + VAST inventory",
                "hf_commits": HF_COMMITS,
                "counts": {cat: len(rows) for cat, rows in sorted(by_cat.items())},
                "bytes": {cat: sum(r["bytes"] for r in rows if isinstance(r["bytes"], int))
                          for cat, rows in sorted(by_cat.items())},
                "entries": sorted(entries, key=lambda r: (r["category"], r["path"]))}
    out_json = ROOT / "evidence_registry.json"
    out_json.write_text(json.dumps(registry, indent=1) + "\n")
    lines = ["# Evidence registry",
             "",
             f"Generated `{created}` by `src/build_evidence_registry.py`. Every paper-relevant artifact, asset, and reproducibility record: path, bytes, SHA-256, and store location. Re-run after new finals land.",
             "",
             "## By category",
             "",
             "| Category | Files | Bytes (MiB) |",
             "|---|---:|---:|"]
    for cat in sorted(by_cat):
        mib = sum(r["bytes"] for r in by_cat[cat] if isinstance(r["bytes"], int)) / 2**20
        lines.append(f"| {cat} | {len(by_cat[cat])} | {mib:.1f} |")
    lines += ["",
              "## Key artifacts (final status)",
              "",
              "| Artifact | SHA-256 (prefix) | Location |",
              "|---|---|---|"]
    key_substrings = ["preservation_full.json", "mmd_metrics.json", "mmd_hf_sha256sums",
                      "jmq_overall.parquet", "jmq_bootstrap.parquet", "combined_test_jmq",
                      "paired_mmd_jmq_l2", "grouped_distribution_shift", "jmq_domain",
                      "style_bigword", "jmq_case_probe", "controls_generations.parquet",
                      "control_generation_done.json", "manifest.yaml", "CONTROL_PROTOCOL.md",
                      "draft_", "H_jmq.tex", "refs.bib", "val2_floor.json"]
    seen = set()
    for entry in registry["entries"]:
        if entry["status"] != "final" or entry["location"] != "local":
            continue
        if any(key in entry["path"] for key in key_substrings) and entry["path"] not in seen:
            seen.add(entry["path"])
            lines.append(f"| `{entry['path']}` | `{entry['sha256'][:16]}` | local + HF dump |")
    lines += ["",
              "## Verification",
              "",
              "- Local files: recompute SHA-256 and compare with `evidence_registry.json`.",
              "- HF dump: file paths mirror this workspace; commits listed in the JSON `hf_commits` map.",
              "- `dump_mirror/` is the immutable verified snapshot; never overwritten while `.local_ahead.json` exists.",
              "- VAST-only transients (progress files, live logs) are re-pulled on completion; finals are promoted by hash.",
              ""]
    (ROOT / "EVIDENCE_REGISTRY.md").write_text("\n".join(lines) + "\n")
    print(f"entries={len(entries)} categories={len(by_cat)}")


if __name__ == "__main__":
    main()
