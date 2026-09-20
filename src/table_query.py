"""table_query.py — frozen/VAL2 rows + TEST headline rows + preservation rows.
Prints all three tables; --out DIR also archives them as metrics/tables/*.md (dumped to HF).
Usage: python src/table_query.py [--out metrics/tables]"""
import argparse
import json
import time
from pathlib import Path
ROOT = Path("/root/AblationWriting")

MODELS = {    "L1": "meta-llama/Llama-3.2-1B-Instruct",
    "S3": "HuggingFaceTB/SmolLM3-3B",
    "Q08": "Qwen/Qwen3.5-0.8B",
    "G4": "google/gemma-4-E4B-it",
    "G2": "google/gemma-4-E2B-it",
    "O1": "allenai/OLMo-2-0425-1B-Instruct",
    "Q20": "Qwen/Qwen3.5-2B",
    "LF12": "LiquidAI/LFM2-1.2B",
}

SKIP = {"G4": "Skipped 2026-09-13 (cost/benefit; smoke + Screen0 only)"}


def uplift(base, abl):
    """% reduction in distance (lower = closer to human). Positive = improvement."""
    if not isinstance(base, (int, float)) or not isinstance(abl, (int, float)) or not base:
        return None
    return 100.0 * (base - abl) / base


def pair(b, kb, ka, fmt="%.5f"):
    bv, av = b.get(kb), b.get(ka)
    if not isinstance(bv, (int, float)) or not isinstance(av, (int, float)):
        return "—"
    u = uplift(bv, av)
    us = f" ({u:+.1f}%)" if u is not None else ""
    return f"{fmt % bv}→{fmt % av}{us}"


def select_row(mid, runs):
    rs = [r for r in runs if r.get("model") == mid and r.get("R_L2") is not None]
    frozen_path = ROOT / "artifacts" / "geometry" / "H2" / mid / "frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text()).get("frozen")
        if frozen:
            return next(r for r in rs if r.get("run_id") == frozen), "Frozen VAL2"
        val2 = [r for r in rs if r.get("stage") == "s3" and r.get("cone") == "cone"]
        if val2:
            return max(val2, key=lambda r: r.get("R_L2", -9)), "Reverted on VAL2"
    full = [r for r in rs if r.get("stage") == "s2"]
    if full:
        active = any(r.get("stage") == "s3" for r in rs)
        return max(full, key=lambda r: r.get("R_L2", -9)), "VAL2 in progress" if active else "VAL2 pending"
    return max(rs, key=lambda r: r.get("R_L2", -9)), "Smoke only; full run pending"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="archive dir, e.g. metrics/tables")
    a = ap.parse_args()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runs = [json.loads(l) for l in open(ROOT / "runs.jsonl") if l.strip()]
    val2, test, preserv, stages = [], [], [], []
    val2 += ["| Model | Best (R + raw L2-1) | L2-2 | L2-3 | JSD | Lengths b/a/h | Top drivers | Comment |"]
    val2 += ["|---|---|---|---|---|---|---|---|"]
    for mid, model in MODELS.items():
        b, comment = select_row(mid, runs)
        comment = SKIP.get(mid, comment)
        top = ", ".join(repr(x) for x in b.get("top_words", ["—"])[:5])
        lb = b.get("len_base")
        la = b.get("len_abl")
        lh = b.get("len_human")
        lens = (f"{lb:.0f}/{la:.0f}/{lh:.0f}"
                if all(isinstance(v, (int, float)) for v in (lb, la, lh)) else "—")
        val2.append(f"| {model} | R={b.get('R_L2', 0):+.3f}; {pair(b, 'L2_base', 'L2_abl')} | "
                    f"{pair(b, 'L2_2_base', 'L2_2_abl')} | {pair(b, 'L2_3_base', 'L2_3_abl')} | "
                    f"{pair(b, 'JSD_base', 'JSD_abl', '%.4f')} | {lens} | {top} | {comment} |")
    test += ["| Model | TEST R L2-1 | L2-1 b→a | JSD b→a | kappa | Status |"]
    test += ["|---|---|---|---|---|---|"]
    for mid, model in MODELS.items():
        mp = ROOT / "artifacts" / "geometry" / "H2" / mid / "test_metrics.json"
        dp = ROOT / "artifacts" / "geometry" / "H2" / mid / "test_done"
        if not mp.exists():
            test.append(f"| {model} | — | — | — | — | Not started |")
            continue
        t = json.loads(mp.read_text())
        r = t.get("R_L2_1")
        rs = f"{r:+.3f}" if isinstance(r, (int, float)) else "—"
        k = t.get("kappa")
        ks = f"{k:.2f}" if isinstance(k, (int, float)) else "—"
        status = "Done" if dp.exists() else "In progress"
        test.append(f"| {model} | {rs} | {pair(t, 'L2_1_base', 'L2_1_abl')} | "
                    f"{pair(t, 'JSD_base', 'JSD_abl', '%.4f')} | {ks} | {status} |")
    preserv += ["| Model | IFEval b→a (rel) | Capability b→a (rel) | Harm-refusal b→a (rel) | False-refusal | Pass |"]
    preserv += ["|---|---|---|---|---|---|"]
    for mid, model in MODELS.items():
        pp = ROOT / "artifacts" / "geometry" / "H2" / mid / "preservation_full.json"
        if not pp.exists():
            preserv.append(f"| {model} | — | — | — | — | Not run |")
            continue
        p = json.loads(pp.read_text())
        g = p.get("gates", {})
        ie = p.get("ifeval", {})
        cb = p.get("composite", {})
        sf = p.get("safety", {})

        def rel(b, av):
            if not all(isinstance(v, (int, float)) for v in (b, av)):
                return "—"
            rr = av / b if b else 0
            return f"{b:.3f}→{av:.3f} ({rr * 100:.1f}%)"
        preserv.append(f"| {model} | {rel(ie.get('base'), ie.get('abl'))} | "
                       f"{rel(cb.get('base'), cb.get('abl'))} | "
                       f"{rel(sf.get('harm_ref_b'), sf.get('harm_ref_a'))} | "
                       f"{(p.get('gates', {}).get('false_refusal_pp', 0) * 100):+.1f}pp | "
                       f"{p.get('pass')} |")
    stages += ["| Model | smoke (n, best R) | S1 (n, best R) | S2 (n, best R) | S3 (n, best R) |"]
    stages += ["|---|---|---|---|---|"]
    stages += ["<!-- Screen0 produces stability windows (screen0_windows.json), not scored configs, so it has no column -->"]
    for mid, model in MODELS.items():
        gen = [r for r in runs if r.get("model") == mid and r.get("R_L2") is not None
               and r.get("hypothesis") != "H2-test"]

        def cell(rows):
            if not rows:
                return "—"
            b = max(rows, key=lambda r: r.get("R_L2", -9))
            return f"n={len(rows)}, R={b.get('R_L2', 0):+.3f}"

        stages.append(f"| {model} | {cell([r for r in gen if not r.get('stage')])} | "
                      f"{cell([r for r in gen if r.get('stage') == 's1'])} | "
                      f"{cell([r for r in gen if r.get('stage') == 's2'])} | "
                      f"{cell([r for r in gen if r.get('stage') == 's3'])} |")
    for section in (val2, test, preserv, stages):
        for line in section:
            print(line, flush=True)
        print(flush=True)
    if a.out:
        out = ROOT / a.out
        out.mkdir(parents=True, exist_ok=True)
        for name, section in (("val2.md", val2), ("test.md", test),
                              ("preservation.md", preserv), ("stages.md", stages)):
            (out / name).write_text(f"<!-- staged {stamp} from runs.jsonl + artifacts on VAST -->\n"
                                    + "\n".join(section) + "\n")


if __name__ == "__main__":
    main()
