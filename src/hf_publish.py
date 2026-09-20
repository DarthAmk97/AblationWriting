"""
hf_publish.py — push valuable work to HF Hub collection + HTML writeup
Per user: valuable work must exist in HF repo. Collection under AblationWriting, writeup html there.
After H2 done: push complete/partial to HF as repo + html. After H3: separate repo, then completion + html append.
H3 deferred → this script supports --stage h2_smoke|h2_full (h3 future).

Reads HF_TOKEN from /root/AblationWriting/.hf_token (600).
Uses huggingface_hub. Creates repo <user>/AblationWriting-H2 (+ collection), uploads snapshot excluding .hf_cache/.hf_token.
Generates writeup.html from template + live metrics (runs.jsonl, spectra, splits summary).
"""
import os, argparse
from pathlib import Path
ROOT = Path("/root/AblationWriting")
TOK = (ROOT / ".hf_token").read_text().strip()
os.environ["HF_TOKEN"] = TOK
os.environ["HUGGING_FACE_HUB_TOKEN"] = TOK

# NOTE: HF username derived from token (whoami). Collection slug "AblationWriting".
# Repos: AblationWriting-H2 (now), AblationWriting-H3 (later), AblationWriting-Completion (later)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#1a1a1a}}
header{{border-bottom:3px solid #111;padding-bottom:1rem}}code,pre{{background:#f5f5f5;padding:.2em .4em;border-radius:6px}}pre{{overflow:auto;padding:1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.5rem;text-align:left}}th{{background:#fafafa}}
.badge{{display:inline-block;background:#111;color:#fff;padding:.2em .6em;border-radius:999px;font-size:.85em}}
.warn{{background:#fff8e1;border:1px solid #ffe082;padding:1rem;border-radius:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div class="badge">{stage}</div><h1>{title}</h1><p><b>Hypothesis H2:</b> low-rank multi-directional concept cone mediates post-training writing collapse; selective positive-cone ablation recovers human distribution while preserving instruction/safety/reasoning.</p>
<p>Paradigm: training-free w.r.t target weights. No LoRA/SFT/DPO/RL. Geometry only (means/cov/SVD/whitening/QR). Rank-1 DIM appendix baseline only.</p></header>
<section><h2>Status</h2><p>{status}</p><div class="warn"><b>H3 deferred:</b> H3 not available to agent. Partials marked awaiting-H3. After H2 → HF repo + this html. After H3 → separate repo. Then completion + html append.</div></section>
<section><h2>Panel & Revisions</h2>{models_table}</section>
<section><h2>Splits (immutable)</h2>{splits_table}</section>
<section><h2>Headline sampler</h2><p>temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024 (smoke 256), generated-token residual only.</p></section>
<section><h2>Smoke results</h2>{smoke_table}<p>Full TEST never touched during selection. Screens: 24→12→4→1 per model via successive halving. Gates: IFEval 95%, capability 97%, refusal 95%, pathology +2pp.</p></section>
<section><h2>Metrics</h2><p>L2-1/2/3 (fixed Qwen3.5-0.8B tokenizer), JSD, MMD (nvidia/llama-embed-nemotron-8b, RBF median VAL bandwidth), R recovery, JMQ_H=2P(beat human) + JMQ_delta via muse-spark-1.3-contributor, StoryScope 304-feat distance.</p></section>
<section><h2>Records</h2><p>manifest.yaml, runs.jsonl, geometry/, metrics/, judgments/, lab_notebook.md, claims_ledger.md, failures.md. Figures via figures4papers style, PDF/SVG+300DPI PNG, scripts regenerate every number.</p></section>
<section><h2>Repro</h2><pre>{repro}</pre></section>
<footer><p>Collection: AblationWriting · Stage: {stage} · Generated {date}</p></footer></body></html>
"""

def build_html(stage, status, models_table, splits_table, smoke_table, repro):
    from datetime import datetime, timezone
    return HTML_TEMPLATE.format(
        title="The Geometry of Machine Writing: Training-Free Selective Ablation of Post-Training Writing Collapse",
        stage=stage, status=status, models_table=models_table, splits_table=splits_table,
        smoke_table=smoke_table, repro=repro, date=datetime.now(timezone.utc).isoformat()
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="h2_smoke")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    print(f"stage={args.stage} dry={args.dry}")
    # gather live tables (best-effort, partial ok)
    import json
    splits_summary = {}
    try:
        splits_summary = json.load(open(ROOT/"data"/"splits"/"split_summary.json"))
    except Exception as e:
        splits_summary = {"error": str(e)}
    runs = []
    try:
        runs = [json.loads(l) for l in open(ROOT/"runs.jsonl")]
    except:
        pass
    print(f"splits={splits_summary} runs={len(runs)}")
    # build minimal tables
    import html as H
    models_table = "<p>See manifest.yaml + artifacts/model_validation.json (8-panel SHAs frozen; L1 gated 403 pending).</p>"
    splits_table = f"<pre>{H.escape(json.dumps(splits_summary, indent=2))}</pre>"
    if runs:
        # top 10 by R
        runs_sorted = sorted([r for r in runs if 'R_L2' in r], key=lambda x: x['R_L2'], reverse=True)[:20]
        rows = "".join(f"<tr><td>{H.escape(r.get('run_id',''))}</td><td>{r.get('model')}</td><td>{r.get('window')}</td><td>{r.get('rank')}</td><td>{r.get('alpha')}</td><td>{r.get('R_L2',0):.3f}</td><td>{r.get('L2_base',0):.4f}->{r.get('L2_abl',0):.4f}</td></tr>" for r in runs_sorted)
        smoke_table = f"<table><tr><th>run</th><th>model</th><th>window</th><th>r</th><th>a</th><th>R</th><th>L2</th></tr>{rows}</table>"
    else:
        smoke_table = "<p>No runs yet (smoke running).</p>"
    repro = "git rev TBD; see manifest.yaml for model/dataset SHAs, sampler, seeds."
    html = build_html(args.stage, f"Partial H2 {args.stage}: {len(runs)} runs. Full TEST held-out. See runs.jsonl.", models_table, splits_table, smoke_table, repro)
    out = ROOT / "writeup.html"
    out.write_text(html, encoding="utf-8")
    print(f"WROTE {out} {len(html)} bytes")
    if args.dry:
        print("dry-run, skipping Hub upload")
    else:
        from huggingface_hub import HfApi
        api = HfApi(token=TOK)
        user = api.whoami()["name"]
        print(f"HF user {user}")
        repo_id = f"{user}/AblationWriting-H2"
        api.create_repo(repo_id, exist_ok=True, private=False)
        # upload snapshot excluding caches/secrets
        api.upload_folder(folder_path=str(ROOT), repo_id=repo_id,
            allow_patterns=["*.md","*.yaml","*.json","*.jsonl","*.html","*.py","*.sh","data/splits/*.jsonl","artifacts/**/*.json","artifacts/**/*.md","logs/*.log","runs.jsonl"],
            ignore_patterns=[".hf_cache/**",".hf_token","**/__pycache__/**"],
            commit_message=f"{args.stage} partial {len(runs)} runs")
        print(f"PUSHED {repo_id}")
        # collection? via API if available — log link
        print(f"Ensure collection 'AblationWriting' contains {repo_id} + writeup.html (create in Hub UI or API).")
