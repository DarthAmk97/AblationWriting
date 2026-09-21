"""
appendix_examples.py — build Appendix G tables: 5 refusal-centric + 5 normal p->b:a:h triples.
Uses dump generations (local) or VAST path: artifacts/geometry/H2/<MID>/s3_generations.parquet (VAL2).
Chooses seeded-random, readable, mid-length, non-refusal-baseline for normal panel; refusal panel PROVES
overlap directly: baseline refusal vs ablated compliance vs human (borderline prompts where human answers).
Writes paper/tables/G_refusal_examples.tex and G_writing_examples.tex (longtable, verbatim-safe).
Usage: python src/appendix_examples.py --models L1 Q08 S3 --out paper/tables
Needs: pandas, pyarrow. No GPU. Reads dump locally (default) or --vast /root/AblationWriting.
"""
import argparse, json, re, textwrap
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = re.compile(r"(i'?m sorry|i cannot|i can'?t|cannot comply|unable to|not able to|against policy|helpline|trusted adult)", re.I)

def _esc_text(s):
    return (str(s).replace("\\", "\\textbackslash ").replace("$", "\\$")
            .replace("^", "\\^").replace("~", "\\~").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#").replace("{", "\\{").replace("}", "\\}"))


def esc(s):
    # Inline $...$ spans in GEN abstracts carry undefined macros (\RR etc);
    # flatten them to plain words instead of escaping (which breaks math mode).
    parts = str(s).split("$")
    if len(parts) % 2 == 0:
        return _esc_text(str(s))
    out = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            out.append(_esc_text(part))
        else:
            out.append(re.sub(r"[$\\^{}_]", "", part))
    return "".join(out)

def pick(df, splits, want_refusal, n=5, seed=7):
    import random
    rng = random.Random(seed)
    base_ref = df["baseline"].str[:700].str.contains(REF, regex=True)
    pool = df[base_ref] if want_refusal else df[~base_ref]
    # mid-length honest random sample (readable, not extreme)
    pool = pool[(pool["baseline"].str.split().str.len().between(40, 280)) & (pool["ablated"].str.split().str.len().between(40, 280))]
    idx = rng.sample(list(pool.index), k=min(n, len(pool)))
    rows = []
    for i in idx:
        pid = str(df.loc[i, "prompt_id"])
        prompt = splits.get(pid, {}).get("prompt", "")[:700]
        human = splits.get(pid, {}).get("human_text", "")[:900]
        rows.append((prompt, df.loc[i, "baseline"][:900], df.loc[i, "ablated"][:900], human))
    return rows

def write_tex(rows, out_path, caption):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{longtable}{p{0.22\\textwidth}p{0.24\\textwidth}p{0.24\\textwidth}p{0.24\\textwidth}}",
             "\\caption{" + caption + "} \\\\ \\hline",
             "Prompt & Baseline & Ablated & Human \\\\ \\hline", "\\endfirsthead",
             "Prompt & Baseline & Ablated & Human \\\\ \\hline", "\\endhead"]
    for p, b, a, h in rows:
        lines.append(" & ".join(esc(x).replace("\n", " ")[:900] for x in [p, b, a, h]) + " \\\\ \\hline")
    lines.append("\\end{longtable}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {out_path} rows={len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["L1"])
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--vast", default=None, help="override root, e.g. /root/AblationWriting for VAST run")
    a = ap.parse_args()
    root = Path(a.vast) if a.vast else ROOT
    splits = {}
    for name in ["val2", "val1", "test"]:
        p = root / "data" / "splits" / f"{name}.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            splits[str(r["prompt_id"])] = r
    out = Path(a.out) if Path(a.out).is_absolute() else root / a.out
    all_ref, all_norm = [], []
    for mid in a.models:
        pq = root / "artifacts" / "geometry" / "H2" / mid / "s3_generations.parquet"
        if not pq.exists():
            print(f"{mid}: no {pq} (stage not done)", flush=True)
            continue
        df = pd.read_parquet(pq)
        # use frozen config only if available, else pool
        frozen = None
        fj = root / "artifacts" / "geometry" / "H2" / mid / "frozen.json"
        if fj.exists():
            try:
                frozen = json.load(open(fj)).get("frozen", "")
            except Exception:
                pass
        if frozen:
            # config string is window-r..-a..-rho.. inside run_id; match substring
            sub = frozen.split(mid + "-", 1)[-1].split("-cone")[0] if mid + "-" in frozen else None
            cand = df[df["config"].astype(str).str.contains(sub.split("-a")[0])] if sub else df
            df = cand if len(cand) >= 10 else df
        all_ref += pick(df, splits, True, n=2, seed=7)   # ~2 per model -> 10 total across panel; trimmed to 5 in aggregate
        all_norm += pick(df, splits, False, n=2, seed=7)
    # take 5 each aggregate
    import random
    rng = random.Random(7)
    if len(all_ref) > 5:
        all_ref = rng.sample(all_ref, 5)
    if len(all_norm) > 5:
        all_norm = rng.sample(all_norm, 5)
    write_tex(all_ref, out / "G_refusal_examples.tex", "Refusal-adjacent prompts: prompt -> baseline : ablated : human (ablation and refusal overlap).")
    write_tex(all_norm, out / "G_writing_examples.tex", "Normal writing prompts: prompt -> baseline : ablated : human (voice only).")


if __name__ == "__main__":
    main()
