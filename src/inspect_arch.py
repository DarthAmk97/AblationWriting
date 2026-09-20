"""
inspect_arch.py — load tokenizer + config + minimal model skeleton to find residual hook points
For Q08 (multimodal) and O1 (text). No generation yet.
"""
import os
from pathlib import Path
ROOT = Path("/root/AblationWriting")
os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".hf_cache" / "hub")
tok = (ROOT / ".hf_token").read_text().strip() if (ROOT / ".hf_token").exists() else None
from transformers import AutoConfig, AutoTokenizer

for repo in ["Qwen/Qwen3.5-0.8B", "allenai/OLMo-2-0425-1B-Instruct", "LiquidAI/LFM2-1.2B", "HuggingFaceTB/SmolLM3-3B"]:
    print(f"\n===== {repo} =====", flush=True)
    try:
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, token=tok)
        print(f"arch={cfg.architectures} type={getattr(cfg,'model_type',None)}", flush=True)
        print(f"keys: {[k for k in dir(cfg) if not k.startswith('_')][:50]}", flush=True)
        # try to get layers
        for attr in ["num_hidden_layers", "num_layers", "n_layer", "num_blocks"]:
            if hasattr(cfg, attr):
                print(f"  {attr}={getattr(cfg, attr)}", flush=True)
        for attr in ["hidden_size", "d_model", "n_embd", "model_dim"]:
            if hasattr(cfg, attr):
                print(f"  {attr}={getattr(cfg, attr)}", flush=True)
        # text config for multimodal?
        if hasattr(cfg, "text_config"):
            tc = cfg.text_config
            print(f"  text_config: hidden={getattr(tc,'hidden_size',None)} layers={getattr(tc,'num_hidden_layers',None)}", flush=True)
        if hasattr(cfg, "language_config"):
            lc = cfg.language_config
            print(f"  language_config: {lc}", flush=True)
        # tokenizer
        try:
            t = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, token=tok)
            print(f"  tokenizer: class={t.__class__.__name__} vocab={t.vocab_size} chat_template={bool(getattr(t,'chat_template',None))}", flush=True)
        except Exception as e:
            print(f"  tokenizer fail: {e}", flush=True)
    except Exception as e:
        import traceback
        print(f"FAIL {e}", flush=True)
        traceback.print_exc()
