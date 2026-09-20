"""
H2 validate_models.py — check 8-model panel + datasets + embedding
Runs with /venv/main (no torch needed). Uses HF Hub API.
Writes: manifest_models.json, model_availability.md
Operates ONLY within /root/AblationWriting
"""
import os, json, sys, traceback
from pathlib import Path

ROOT = Path("/root/AblationWriting")
HF_HOME = ROOT / ".hf_cache"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")

# token from file if present
tok_path = ROOT / ".hf_token"
HF_TOKEN = None
if tok_path.exists():
    HF_TOKEN = tok_path.read_text().strip()
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=HF_TOKEN)

MODELS = {
    "Q08": "Qwen/Qwen3.5-0.8B",
    "Q20": "Qwen/Qwen3.5-2B",
    "G2": "google/gemma-4-E2B-it",
    "G4": "google/gemma-4-E4B-it",
    "L1": "meta-llama/Llama-3.2-1B-Instruct",
    "O1": "allenai/OLMo-2-0425-1B-Instruct",
    "LF12": "LiquidAI/LFM2-1.2B",
    "S3": "HuggingFaceTB/SmolLM3-3B",
}
EMBED = "nvidia/llama-embed-nemotron-8b"
DATASETS = {
    "GEN": "szyszy/GEN",
    "PAIRED": "dmitva/human_ai_generated_text",
}

out = {"models": {}, "embed": {}, "datasets": {}}

def check_model(mid, repo):
    try:
        info = api.model_info(repo)
        # get sha
        sha = info.sha
        # siblings count, tags
        res = {
            "repo": repo,
            "exists": True,
            "sha": sha,
            "gated": getattr(info, "gated", None),
            "private": getattr(info, "private", False),
            "tags": getattr(info, "tags", [])[:20],
            "lastModified": str(getattr(info, "last_modified", "")),
        }
        # try config.json to get arch
        try:
            cfg_path = hf_hub_download(repo, "config.json", token=HF_TOKEN)
            import json as js
            cfg = js.load(open(cfg_path))
            res["architectures"] = cfg.get("architectures", [])
            res["model_type"] = cfg.get("model_type", "")
            res["hidden_size"] = cfg.get("hidden_size", None)
            res["num_hidden_layers"] = cfg.get("num_hidden_layers", None)
            res["tie_word_embeddings"] = cfg.get("tie_word_embeddings", None)
            # multimodal flags
            res["is_multimodal"] = any(k in cfg for k in ["vision_config", "vision_encoder", "image_token_id", "multimodal"])
            res["config_keys_sample"] = list(cfg.keys())[:30]
        except Exception as e:
            res["config_error"] = f"{type(e).__name__}: {e}"
        return res
    except Exception as e:
        return {"repo": repo, "exists": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}

def check_dataset(repo):
    try:
        info = api.dataset_info(repo)
        return {"repo": repo, "exists": True, "sha": info.sha, "tags": getattr(info, "tags", [])[:20], "lastModified": str(getattr(info, "last_modified", ""))}
    except Exception as e:
        return {"repo": repo, "exists": False, "error": f"{type(e).__name__}: {e}"}

for k, repo in MODELS.items():
    print(f"=== checking {k} {repo} ===", flush=True)
    out["models"][k] = check_model(k, repo)
    print(json.dumps(out["models"][k], indent=2)[:3000], flush=True)

print("=== embed ===", flush=True)
out["embed"] = check_model("embed", EMBED)
print(json.dumps(out["embed"], indent=2)[:2000], flush=True)

for k, repo in DATASETS.items():
    print(f"=== dataset {k} {repo} ===", flush=True)
    out["datasets"][k] = check_dataset(repo)
    print(json.dumps(out["datasets"][k], indent=2), flush=True)

# StoryScope: try to locate - likely github not HF. Check common candidates
# We just record that StoryScope is code/feature resource, not HF dataset
out["storyscope_note"] = "StoryScope prompts/features from Russell et al arXiv:2604.03136; human story text not redistributed; freeze 100 prompts later. Check github release."

(ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
with open(ROOT / "artifacts" / "model_validation.json", "w") as f:
    json.dump(out, f, indent=2)

# markdown summary
lines = ["# H2 Model/Dataset Availability (PRE-TEST)", "", f"Generated: validation run", ""]
lines.append("## Models")
for k in MODELS:
    m = out["models"][k]
    if m.get("exists"):
        lines.append(f"- **{k}** `{m['repo']}` SHA `{m['sha']}` arch={m.get('architectures')} type={m.get('model_type')} hidden={m.get('hidden_size')} layers={m.get('num_hidden_layers')} multimodal={m.get('is_multimodal')} gated={m.get('gated')}")
        if m.get("config_error"):
            lines.append(f"  - config_error: {m['config_error']}")
    else:
        lines.append(f"- **{k}** `{m['repo']}` MISSING: {m.get('error')}")
lines.append("")
lines.append("## Embedding")
e = out["embed"]
lines.append(f"- `{EMBED}` exists={e.get('exists')} sha={e.get('sha')} err={e.get('error','')}")
lines.append("")
lines.append("## Datasets")
for k in DATASETS:
    d = out["datasets"][k]
    lines.append(f"- **{k}** `{d['repo']}` exists={d.get('exists')} sha={d.get('sha')} err={d.get('error','')}")
lines.append("")
lines.append("## Notes")
lines.append("- Qwen3.5 family is multimodal Image-Text-to-Text with hybrid Gated DeltaNet + Attention + MoE per model card. H2 residual intervention must target language backbone only; log any incompatibility before TEST.")
lines.append("- LFM2-1.2B is hybrid Liquid Foundation Model; same caution.")
lines.append("- If any of 8 missing/gated without access, log exclusion before TEST with exact incompatibility per protocol Sec 2.")
(ROOT / "artifacts" / "model_availability.md").write_text("\n".join(lines))
print("\n".join(lines))
print("WROTE artifacts/model_validation.json + model_availability.md")
