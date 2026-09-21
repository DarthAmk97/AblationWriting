import sys, json
sys.path.insert(0, "src")
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from h2_controls import build_control_bases, GeneratedTokenHook, set_seed, stable_prompt_seed, _find_layers
BASE = "X:/android-home/tmp/opencode/base_Q20"
geom = Path("fetch_basis/Q20/basis")
record = {"rank": 32, "alpha": 0.25, "rho": 1.0, "layers": [11, 12, 13, 14, 15]}
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cuda:0"}, trust_remote_code=True).eval()
layer_map = _find_layers(model)
source = {"record": record, "geom": geom, "model": "Q20"}
bases = build_control_bases(model, tok, source, layer_map)
print("bases-ok", sorted(bases.keys()), flush=True)
dev = next(model.parameters()).device
hooks = {}
handles = []
for layer in sorted(bases.keys()):
    hook = GeneratedTokenHook(bases[layer]["frozen"], bases[layer]["mu"], 0.25, dev)
    hooks[layer] = hook
    handles.append(layer_map[layer].register_forward_hook(hook))
jobs = json.load(open("rephrase_paper_jobs3.json", encoding="utf-8"))
TASK = ("Rephrase the passage below in one single passage. Rules: output ONLY the rephrased passage, "
        "nothing before it and nothing after it. Never list multiple versions or options. Never start "
        "with Here is, Here are, or any preamble. Keep every number, name, and fact exactly as written, "
        "with identical digits. Add no new facts and no new sentences. Passage: ")
outs = []
for j in jobs:
    set_seed(stable_prompt_seed(j["job"] + "-v3"))
    rendered = tok.apply_chat_template([{"role": "user", "content": TASK + j["text"]}], tokenize=False, add_generation_prompt=True)
    enc = {k: v.to(dev) for k, v in tok(rendered, return_tensors="pt").items()}
    n = enc["input_ids"].shape[1]
    for hook in hooks.values():
        hook.reset(n)
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95, top_k=50, repetition_penalty=1.0, max_new_tokens=768, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    outs.append({"job": j["job"], "out": tok.decode(out[0, n:], skip_special_tokens=True)})
    json.dump(outs, open("rephrase_paper_outs3.json", "w", encoding="utf-8"), indent=1)
    print("did " + j["job"] + " " + str(len(outs)), flush=True)
for handle in handles:
    handle.remove()
print("ALL-DONE", flush=True)