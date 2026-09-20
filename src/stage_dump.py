"""
stage_dump.py — assemble /root/AblationWriting/dump/ + README + dump_manifest.json, push to HF dump repo.
Idempotent: rerun anytime; pushes on content change (caller decides when). CPU-only, no GPU touched.
Repo: amkkk/AblationWriting-H2-dump (created exist_ok). Excludes: .hf_cache, .hf_token, .opencode_key, weights, __pycache__.
Usage: python src/stage_dump.py [--push|--no-push]
"""
import json, shutil, sys, time
from pathlib import Path
ROOT = Path("/root/AblationWriting")
DUMP = ROOT / "dump"
REPO = "amkkk/AblationWriting-H2-dump"

COPIES = [  # (src_rel, dst_rel); missing sources are recorded as pending, not errors
    ("runs.jsonl", "runs.jsonl"),
    ("manifest.yaml", "manifest.yaml"),
    ("lab_notebook.md", "lab_notebook.md"),
    ("claims_ledger.md", "claims_ledger.md"),
    ("failures.md", "failures.md"),
    ("state_ledger.md", "state_ledger.md"),
    ("compute_budget.md", "compute_budget.md"),
    ("DUMP_SPEC.md", "DUMP_SPEC.md"),
    ("PROGRESS.md", "PROGRESS.md"),
    ("ENV_SCOPE.txt", "ENV_SCOPE.txt"),
    ("writeup.html", "writeup.html"),
    ("data/splits/split_summary.json", "data/splits/split_summary.json"),
]
GLOBS = [
    ("runs.jsonl.bak-*", ""),
    ("src/*.py", "src"),
    ("src/*.sh", "src"),
    ("paper/*.txt", "paper"),
    ("paper/tables/*", "paper/tables"),
    ("data/splits/*.jsonl", "data/splits"),
    ("artifacts/metrics/*.json", "artifacts/metrics"),
    ("artifacts/geometry/H2/*/*.json", "artifacts/geometry"),
    ("artifacts/geometry/H2/*/*layer*.npz", "artifacts/geometry"),
    ("artifacts/geometry/H2/*/*.parquet", "artifacts/geometry"),
    ("artifacts/geometry/H2/*/test_done", "artifacts/geometry"),
    ("judging/**/*.parquet", "judging"),
    ("judgments/*.parquet", "judgments"),
    ("metrics/*", "metrics"),
    ("metrics/tables/*", "metrics/tables"),
    ("figures/data/*", "figures/data"),
    ("paper/sections/*.tex", "paper/sections"),
    ("paper/appendix/*.tex", "paper/appendix"),
]


def code_hashes():
    """sha256 of intervention code so a pack rebuild is verifiable (method = weights + code)."""
    import hashlib
    out = {}
    for p in sorted((ROOT / "src").glob("*")):
        if p.suffix not in {".py", ".sh"}:
            continue
        if p.exists():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    (DUMP / "code_hashes.json").write_text(json.dumps(out, indent=2))
    return out


def main(push=True):
    import os
    os.makedirs(DUMP, exist_ok=True)
    manifest = {"staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}, "pending": []}
    for src, dst in COPIES:
        s, d = ROOT / src, DUMP / dst
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            manifest["files"][dst] = d.stat().st_size
        else:
            manifest["pending"].append(src + " (missing)")
    import glob
    for pat, base in GLOBS:
        hits = glob.glob(str(ROOT / pat), recursive=True)
        if not hits:
            manifest["pending"].append(pat + " (none yet)")
            continue
        for h in hits:
            if Path(h).name.startswith("test_progress"):
                continue
            if not os.path.isfile(h):
                continue
            rel = Path(h).relative_to(ROOT)
            d = DUMP / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(h, d)
            manifest["files"][str(rel)] = d.stat().st_size
    # counts for README + tick
    runs = sum(1 for _ in open(DUMP / "runs.jsonl")) if (DUMP / "runs.jsonl").exists() else 0
    frozen = sorted((DUMP / "artifacts" / "geometry").glob("H2/*/frozen.json")) if (DUMP / "artifacts" / "geometry").exists() else []
    frozen_models = []
    for f in frozen:
        try:
            j = json.load(open(f))
            if j.get("frozen"):
                frozen_models.append(f.parent.name + ":" + j["frozen"])
        except Exception:
            pass
    jmq = list((DUMP / "judging").rglob("*.parquet")) if (DUMP / "judging").exists() else []
    manifest["runs"] = runs
    manifest["frozen"] = frozen_models
    manifest["jmq_inputs_ready"] = len(jmq) > 0
    if not jmq:
        manifest["pending"].append("judging/inputs/test_jmq_pairs.parquet (built pre-TEST)")
    manifest["code"] = code_hashes()
    (DUMP / "dump_manifest.json").write_text(json.dumps(manifest, indent=2))
    readme = ["# AblationWriting H2 — local-takeover dump", ""]
    readme.append(f"Staged: `{manifest['staged_at']}` from `/root/AblationWriting` (VAST RTX 3090).")
    readme.append("No weights, no caches. Models re-resolve via SHAs in `manifest.yaml`. GEN refs: CC-BY-NC — keep this repo private/gated.")
    readme.append("")
    readme.append(f"- runs.jsonl rows: **{runs}**")
    readme.append(f"- frozen: {', '.join(frozen_models) if frozen_models else '(none yet)'}")
    readme.append(f"- JMQ inputs ready: **{manifest['jmq_inputs_ready']}**")
    readme.append("")
    readme.append("## Navigate")
    readme.append("- Provenance: manifest.yaml, lab_notebook.md, claims_ledger.md, failures.md, state_ledger.md, compute_budget.md, DUMP_SPEC.md")
    readme.append("- Splits + refs: data/splits/ (prompt IDs, human/AI texts, hashes)")
    readme.append("- Stage gates per model: artifacts/geometry/H2/<MID>/{screen0,s1_top12,s2_top4,frozen,preservation_mini}.json")
    readme.append("- Texts per prompt per config: artifacts/geometry/H2/<MID>/{s2,s3}_generations.parquet")
    readme.append("- Judge feed (when ready): judging/inputs/test_jmq_pairs.parquet → run local_jmq.py → push judgments/*.parquet back here")
    readme.append("- Figure tables: figures/data/ ; metrics: metrics/")
    readme.append("")
    readme.append("## Pending on VAST")
    for p in manifest["pending"]:
        readme.append(f"- [ ] {p}")
    (DUMP / "README.md").write_text("\n".join(readme) + "\n")
    print(f"STAGED {len(manifest['files'])} files, runs={runs}, frozen={len(frozen_models)}, jmq_ready={manifest['jmq_inputs_ready']}", flush=True)
    print("PENDING:", "; ".join(manifest["pending"]) if manifest["pending"] else "none", flush=True)
    if push:
        tok = (ROOT / ".hf_token").read_text().strip()
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        api.create_repo(REPO, exist_ok=True, private=True)
        api.upload_folder(folder_path=str(DUMP), repo_id=REPO, commit_message=f"dump {manifest['staged_at']} runs={runs}")
        print(f"PUSHED {REPO}", flush=True)
        (DUMP / ".last_push").write_text(manifest["staged_at"])


if __name__ == "__main__":
    main(push="--no-push" not in sys.argv)
