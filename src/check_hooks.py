"""
check_hooks.py — print residual block paths for hook targeting
"""
import os
from pathlib import Path
ROOT = Path("/root/AblationWriting")
os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".hf_cache" / "hub")
tok = (ROOT / ".hf_token").read_text().strip() if (ROOT / ".hf_token").exists() else None
import torch
from transformers import AutoModelForCausalLM, AutoModel
from transformers import Olmo2ForCausalLM

def show(repo, cls_name):
    print(f"\n===== {repo} =====", flush=True)
    try:
        # Use AutoModel to get correct class
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
        try:
            m = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, trust_remote_code=True, token=tok, low_cpu_mem_usage=True)
            print(f"loaded as ImageTextToText: {m.__class__.__name__}", flush=True)
        except Exception as e1:
            print(f"ImageTextToText fail: {e1}", flush=True)
            m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, trust_remote_code=True, token=tok, low_cpu_mem_usage=True)
            print(f"loaded as CausalLM: {m.__class__.__name__}", flush=True)
        # list top modules
        print("top children:", [n for n,_ in m.named_children()], flush=True)
        # find layers
        for name, mod in m.named_modules():
            if "layers" in name and len(name.split(".")) <= 3:
                # print layer container
                try:
                    l = len(list(mod))
                    print(f"  container {name}: len={l} type={mod.__class__.__name__}", flush=True)
                except:
                    pass
        # count transformer blocks by looking for self_attn
        blocks = [n for n,_ in m.named_modules() if n.endswith("self_attn") or n.endswith("self_attention")]
        print(f"self_attn blocks sample: {blocks[:5]} total {len(blocks)}", flush=True)
        # show one block path
        if blocks:
            print(f"example block parent: {'.'.join(blocks[0].split('.')[:-1])}", flush=True)
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        import traceback
        print(f"FAIL {e}", flush=True)
        traceback.print_exc()

show("allenai/OLMo-2-0425-1B-Instruct", "olmo")
