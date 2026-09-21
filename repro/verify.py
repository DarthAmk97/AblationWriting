"""repro/verify.py - one-command verification for every results table.

Tier 1 (byte-exact re-render from frozen inputs with the real pipeline code):
  metrics/tables/controls.md            <- src/control_stats.write_table
  metrics/tables/combined_test_jmq.md   <- src/combined_results.write_table
  metrics/tables/paired_mmd_jmq_l2.md   <- src/combined_results.write_paired_table
  metrics/tables/grouped_distribution_shift.md <- src/combined_results.write_grouped_distribution_table
  metrics/tables/jmq.md                 <- src/jmq_stats.write_outputs (timestamp line normalized)
  metrics/tables/mmd.md                 <- src/mmd_score._write_table (backup/restore)
  figures/data/combined_test_jmq.parquet, figures/data/jmq_overall.parquet (frame equality)
Tier 2 (hash-pinned, rendered once by ad-hoc analysis; provenance in repro/README.md):
  jmq_domain, jmq_truncation, style_bigword, jmq_rationale_themes, preliminary_summary
PNGs are hash-recorded only: exact bytes depend on the matplotlib build.

Usage:
  python repro/verify.py                  # verify everything against committed files
  python repro/verify.py --write-manifest # (re)record repro/manifest.json after review
  python repro/verify.py --only controls  # subset by key
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUMP_REPO = "amkkk/AblationWriting-H2-dump"
DUMP_COMMIT = "dd9e18b06e25ce2edccdf93737aae3da9b3e0c8b"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"

TIER2 = {
    "metrics/tables/jmq_domain.md": "ad-hoc domain grid from metrics/jmq_domain_wins.parquet; see exports/jmq_case_probe.md",
    "metrics/tables/jmq_truncation.md": "ad-hoc truncation audit from metrics/jmq_truncation_audit.parquet",
    "metrics/tables/style_bigword.md": "ad-hoc diction stats from metrics/style_bigword.parquet",
    "metrics/tables/jmq_rationale_themes.md": "ad-hoc rationale coding from metrics/jmq_rationale_themes.parquet",
    "metrics/tables/preliminary_summary.md": "hand-written preliminary snapshot, superseded by combined tables",
}

RESULTS = []


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def note(label, ok, detail=""):
    RESULTS.append((label, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + label + ((" :: " + detail) if detail else ""), flush=True)
    return bool(ok)


def norm_jmq(text):
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    if lines and lines[0].startswith("<!-- generated"):
        lines = lines[1:]
    return "\n".join(lines)


def norm_bytes(data):
    """Line-ending normalization: VAST renders LF, Windows writers emit CRLF.
    Tables are compared on content; raw shas stay pinned in the manifest."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
def verify_controls(root, tmp):
    sys.path.insert(0, str(root / "src"))
    import pandas as pd
    from control_stats import write_table
    frame = pd.read_parquet(root / "metrics/control_metrics.parquet")
    out = tmp / "controls.md"
    write_table(frame, out)
    return note("controls.md", norm_bytes(out.read_bytes()) == norm_bytes((root / "metrics/tables/controls.md").read_bytes()))


def verify_combined(root, tmp):
    sys.path.insert(0, str(root / "src"))
    import pandas as pd
    from combined_results import load_rows, write_table, write_paired_table, write_grouped_distribution_table
    data = load_rows(root)
    want_data = pd.read_parquet(root / "metrics/combined_test_jmq.parquet")
    try:
        pd.testing.assert_frame_equal(data.reset_index(drop=True), want_data.reset_index(drop=True), check_dtype=False)
        note("combined_test_jmq.parquet:data", True)
    except AssertionError as error:
        return note("combined_test_jmq.parquet:data", False, str(error)[:200])
    want_fig = pd.read_parquet(root / "figures/data/combined_test_jmq.parquet")
    try:
        pd.testing.assert_frame_equal(data.reset_index(drop=True), want_fig.reset_index(drop=True), check_dtype=False)
        note("figures/data/combined_test_jmq.parquet", True)
    except AssertionError as error:
        return note("figures/data/combined_test_jmq.parquet", False, str(error)[:200])
    ok = True
    for writer, name in ((write_table, "combined_test_jmq.md"), (write_paired_table, "paired_mmd_jmq_l2.md"),
                         (write_grouped_distribution_table, "grouped_distribution_shift.md")):
        out = tmp / name
        writer(data, out)
        ok = note(name, norm_bytes(out.read_bytes()) == norm_bytes((root / "metrics/tables" / name).read_bytes())) and ok
    return ok


def verify_jmq(root, tmp):
    sys.path.insert(0, str(root / "src"))
    import pandas as pd
    from jmq_stats import analyze, write_outputs
    data = pd.read_parquet(root / "judgments/jmq_overall.parquet")
    feed = pd.read_parquet(root / "dump_mirror/judging/inputs/test_jmq_pairs.parquet")
    stats, audit = analyze(data, feed)
    write_outputs(stats, audit, tmp)
    got = norm_jmq((tmp / "metrics/tables/jmq.md").read_text(encoding="utf-8"))
    want = norm_jmq((root / "metrics/tables/jmq.md").read_text(encoding="utf-8"))
    ok = note("jmq.md", got == want)
    try:
        pd.testing.assert_frame_equal(stats.reset_index(drop=True),
                                      pd.read_parquet(root / "metrics/jmq_bootstrap.parquet").reset_index(drop=True),
                                      check_dtype=False)
        note("jmq_bootstrap.parquet:stats", True)
    except AssertionError as error:
        ok = note("jmq_bootstrap.parquet:stats", False, str(error)[:200]) and ok
    return ok


def verify_mmd(root):
    sys.path.insert(0, str(root / "src"))
    from mmd_score import _write_table
    target = root / "metrics/tables/mmd.md"
    backup = target.read_bytes()
    try:
        _write_table(root)
        return note("mmd.md", norm_bytes(target.read_bytes()) == norm_bytes(backup))
    finally:
        target.write_bytes(backup)


def verify_paper_tables(root):
    sys.path.insert(0, str(root / "src"))
    import paper_tables
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="repro-paper-"))
    ok = True
    try:
        for name, func in (("T_control_main.tex", paper_tables.control_main),
                           ("T_jmq_main.tex", paper_tables.jmq_main),
                           ("T_style.tex", paper_tables.style_table),
                           ("T_domain.tex", paper_tables.domain_table),
                           ("T_domain_top.tex", paper_tables.domain_top),
                           ("T_stages.tex", paper_tables.stage_table),
                           ("T_refusal.tex", paper_tables.refusal_table),
                           ("T_drift.tex", paper_tables.drift_table)):
            before = {p.name: sha_file(p) for p in (root / "paper/tables").glob("T_*.tex")}
            func()
            got = {p.name: sha_file(p) for p in (root / "paper/tables").glob("T_*.tex")}
            for key in before:
                (tmp / key).write_bytes((root / "paper/tables" / key).read_bytes())
            ok = note("paper/" + name, got.get(name) == before.get(name)) and ok
    finally:
        for key in before:
            (root / "paper/tables" / key).write_bytes((tmp / key).read_bytes())
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def verify_pinned(root, manifest):
    ok = True
    pins = manifest.get("pins", {})
    for rel, why in TIER2.items():
        want = pins.get(rel)
        got = sha_file(root / rel)
        if want is None:
            ok = note(rel, False, "no pin in manifest; " + why) and ok
        else:
            ok = note(rel, got == want, "" if got == want else "got " + got[:12]) and ok
    return ok


def env_record():
    import platform
    record = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "matplotlib", "scipy"):
        try:
            record[package] = __import__(package).__version__
        except Exception:
            record[package] = "missing"
    return record


def main():
    parser = argparse.ArgumentParser(description="verify every results table")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--only", default="", help="comma-separated keys: controls,combined,jmq,mmd,pinned,paper")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root
    only = set(value for value in args.only.split(",") if value) or {"controls", "combined", "jmq", "mmd", "pinned", "paper"}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    with tempfile.TemporaryDirectory(prefix="repro-verify-") as directory:
        tmp = Path(directory)
        if "controls" in only:
            verify_controls(root, tmp)
        if "combined" in only:
            verify_combined(root, tmp)
        if "jmq" in only:
            verify_jmq(root, tmp)
        if "mmd" in only:
            verify_mmd(root)
        if "pinned" in only:
            verify_pinned(root, manifest)
        if "paper" in only:
            verify_paper_tables(root)
    if args.write_manifest:
        pins = {rel: sha_file(root / rel) for rel in TIER2}
        tables = {}
        for name in ("combined_test_jmq.md", "paired_mmd_jmq_l2.md", "grouped_distribution_shift.md",
                     "jmq.md", "mmd.md", "controls.md"):
            tables["metrics/tables/" + name] = sha_file(root / "metrics/tables" / name)
        for name in ("T_control_main.tex", "T_jmq_main.tex", "T_style.tex",
                     "T_domain.tex", "T_domain_top.tex", "T_stages.tex",
                     "T_refusal.tex", "G_panels.tex", "G_refusal_examples.tex",
                     "G_writing_examples.tex"):
            tables["paper/tables/" + name] = sha_file(root / "paper/tables" / name)
        manifest = {"dump_commit": DUMP_COMMIT, "dump_repo": DUMP_REPO, "env": env_record(),
                    "pins": pins, "tables": tables}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("wrote " + str(MANIFEST), flush=True)
    failed = [label for label, ok, _ in RESULTS if not ok]
    print("SUMMARY %d passed %d failed" % (len(RESULTS) - len(failed), len(failed)), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())