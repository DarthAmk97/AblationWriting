"""
smoke_h2.py — H2 smoke: Q08+O1, 512 CAL, 128 VAL, ranks {2,4,8}, 2 windows, alpha {0.5,1.0}, rho 0.75
No StoryScope. Validates hooks/storage/effect direction, NOT hypothesis.
Writes: artifacts/geometry/H2/<model>/..., generations parquet, metrics, runs.jsonl
Usage: HF_HOME=/root/AblationWriting/.hf_cache python smoke_h2.py --models O1 --cal_n 512 --val_n 128
"""
import os, json, time, argparse, hashlib
from pathlib import Path
import numpy as np
import torch
import pandas as pd

ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))
from h2_lib import set_seed, stable_hash, load_splits, tokenize_teacherforce, mean_states_teacherforce, estimate_whitening, discover_basis, build_protected_basis, clean_basis, ConeAblationHook, find_layer_modules, l2_ngram, jsd_1gram, recovery, pathologies

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
TOK_PATH = ROOT / ".hf_token"
HF_TOK = TOK_PATH.read_text().strip() if TOK_PATH.exists() else None

MODEL_MAP = {
    "Q08": "Qwen/Qwen3.5-0.8B",
    "Q20": "Qwen/Qwen3.5-2B",
    "G2": "google/gemma-4-E2B-it",
    "G4": "google/gemma-4-E4B-it",
    "O1": "allenai/OLMo-2-0425-1B-Instruct",
    "LF12": "LiquidAI/LFM2-1.2B",
    "L1": "meta-llama/Llama-3.2-1B-Instruct",
    "S3": "HuggingFaceTB/SmolLM3-3B",
}
MULTIMODAL = {"Q08", "Q20", "G2", "G4"}

def load_model(mid):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
    repo = MODEL_MAP[mid]
    print(f"[{mid}] loading {repo}", flush=True)
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, token=HF_TOK)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # try multimodal first for Q08
    model = None
    if mid in MULTIMODAL:
        try:
            model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, token=HF_TOK)
            print(f"[{mid}] loaded as {model.__class__.__name__}", flush=True)
        except Exception as e:
            print(f"[{mid}] ImageTextToText fallback: {e}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, token=HF_TOK)
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, token=HF_TOK)
    model.eval()
    print(f"[{mid}] device map: {model.hf_device_map if hasattr(model,'hf_device_map') else 'unknown'}", flush=True)
    layer_map, path = find_layer_modules(model)
    print(f"[{mid}] hook path {path} n_layers={len(layer_map)}", flush=True)
    return tok, model, layer_map, path

def generate_batch(model, tok, prompts, seed_base=0, temp=0.8, top_p=0.95, top_k=50, max_new=256):
    """
    Sequential generation (batch 1 for determinism + hook safety), deterministic seed per prompt.
    prompts: list of str
    Returns list of str completions.
    max_new 256 for smoke (not 1024) to save time; log as smoke simplification. Full will use 1024 cap.
    """
    outs = []
    device = next(model.parameters()).device
    for idx, prompt in enumerate(prompts):
        seed = (seed_base + stable_hash(prompt)) % (2**31-1)
        set_seed(seed)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(device)
        # prefill len for hook exclusion is handled via hook.prefill_len set outside; here no hooks
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=temp, top_p=top_p, top_k=top_k,
                max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        # decode only new tokens
        new_ids = gen[0, enc["input_ids"].shape[1]:]
        outs.append(tok.decode(new_ids, skip_special_tokens=True))
    return outs

def generate_with_hooks(model, tok, prompts, hooks_dict, prefill_lens=None, seed_base=0, temp=0.8, top_p=0.95, top_k=50, max_new=256):
    """
    hooks_dict: {layer_idx: (B, mu, alpha)} B (d,r) numpy, mu (d,) numpy
    Registers forward hooks on window layers, generates, removes hooks, returns texts + stats.
    """
    handles = []
    hook_objs = {}
    # find layer modules
    layer_map, _ = find_layer_modules(model)
    device = next(model.parameters()).device
    for li, (B_np, mu_np, alpha) in hooks_dict.items():
        mod = layer_map[li]
        B_t = torch.from_numpy(B_np).to(device)  # (d,r)
        mu_t = torch.from_numpy(mu_np).to(device)
        # prefill len unknown until tokenized; set per-prompt dynamically? For simplicity set 0 and intervene on all?
        # Protocol says intervene only on generated tokens. We need prefill len per prompt.
        # Workaround: hook with prefill_len set per generation call (we generate one by one, update hook.prefill_len before each).
        h = ConeAblationHook(B_t, mu_t, alpha=alpha, prefill_len=0)
        handle = mod.register_forward_hook(h)
        handles.append(handle)
        hook_objs[li] = h
    outs = []
    for prompt in prompts:
        seed = (seed_base + stable_hash(prompt)) % (2**31-1)
        set_seed(seed)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(device)
        pre_len = enc["input_ids"].shape[1]
        for h in hook_objs.values():
            h.prefill_len = pre_len
            h.stats = {"removed_norm": 0.0, "tokens": 0, "pos_frac": 0.0}
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=temp, top_p=top_p, top_k=top_k,
                max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        new_ids = gen[0, pre_len:]
        outs.append(tok.decode(new_ids, skip_special_tokens=True))
    # aggregate stats
    agg = {li: dict(h.stats) for li, h in hook_objs.items()}
    for h in handles:
        h.remove()
    return outs, agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["O1"])
    ap.add_argument("--cal_n", type=int, default=512)
    ap.add_argument("--val_n", type=int, default=128)
    ap.add_argument("--ranks", nargs="+", type=int, default=[2,4,8])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5,1.0])
    ap.add_argument("--rho", type=float, default=0.75)
    ap.add_argument("--max_new", type=int, default=256)
    args = ap.parse_args()

    splits = load_splits(["cal", "val1"])
    cal = splits["cal"][:args.cal_n]
    val = splits["val1"][:args.val_n]
    print(f"CAL {len(cal)} VAL {len(val)}", flush=True)

    # eval tokenizer fixed: Qwen3.5-0.8B
    from transformers import AutoTokenizer as AT
    eval_tok = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=HF_TOK)
    print(f"eval tokenizer vocab {eval_tok.vocab_size}", flush=True)

    runs_path = ROOT / "runs.jsonl"
    # ensure header? append

    for mid in args.models:
        run_start = time.time()
        print(f"\n========== SMOKE {mid} ==========", flush=True)
        tok, model, layer_map, hook_path = load_model(mid)
        L = len(layer_map)
        print(f"[{mid}] L={L}", flush=True)
        # define 2 windows
        if L == 16:
            windows = {"W1_25-44": [4,5,6,7], "W2_50-69": [8,9,10,11]}
        elif L == 24:
            windows = {"W1_25-42": [6,7,8,9,10], "W2_50-67": [12,13,14,15,16]}
        elif L == 36:
            windows = {"W1_25-42": [9,10,11,12,13,14], "W2_50-67": [18,19,20,21,22,23]}
        else:
            # generic: 10% windows centred middle
            w = max(2, L//4)
            windows = {"W1": list(range(L//4, L//4+w)), "W2": list(range(L//2, L//2+w))}
        print(f"[{mid}] windows {windows}", flush=True)

        device = next(model.parameters()).device
        # --- baseline generation VAL ---
        val_prompts = [r["prompt"] for r in val]
        val_human = [r["human_text"] for r in val]
        print(f"[{mid}] baseline generation {len(val_prompts)} x max_new {args.max_new}", flush=True)
        t0 = time.time()
        val_baseline = generate_batch(model, tok, val_prompts, seed_base=1000, max_new=args.max_new)
        t_gen = time.time()-t0
        print(f"[{mid}] baseline done {t_gen:.1f}s ({t_gen/len(val_prompts):.2f}s/prompt)", flush=True)
        # cheap metric baseline vs human
        l2_base, _ = l2_ngram(val_baseline, val_human, eval_tok, n=1)
        jsd_base = jsd_1gram(val_baseline, val_human, eval_tok)
        print(f"[{mid}] BASELINE L2-1 {l2_base:.5f} JSD {jsd_base:.5f} path {pathologies(val_baseline)}", flush=True)

        # --- teacher forcing CAL ---
        print(f"[{mid}] teacher-forcing CAL {len(cal)}", flush=True)
        # collect per-layer human means + displacements
        # H_human per layer: (N,d)
        per_layer_H = {li: [] for li in layer_map}
        per_layer_D = {li: [] for li in layer_map}
        for idx, r in enumerate(cal):
            if idx % 100 == 0:
                print(f"  [{mid}] CAL {idx}/{len(cal)}", flush=True)
            try:
                hH, _ = mean_states_teacherforce(model, tok, r["prompt"], r["human_text"], layers=set(layer_map.keys()), device=device)
                hA, _ = mean_states_teacherforce(model, tok, r["prompt"], r["ai_text"], layers=set(layer_map.keys()), device=device)
                for li in layer_map:
                    if li in hH and li in hA:
                        per_layer_H[li].append(hH[li].numpy())
                        per_layer_D[li].append((hA[li]-hH[li]).numpy())
            except Exception as e:
                print(f"  CAL {idx} fail {e}", flush=True)
                continue
        # stack
        for li in layer_map:
            per_layer_H[li] = np.stack(per_layer_H[li], axis=0) if len(per_layer_H[li]) else np.zeros((0,1))
            per_layer_D[li] = np.stack(per_layer_D[li], axis=0) if len(per_layer_D[li]) else np.zeros((0,1))
        print(f"[{mid}] teacherforce done, example layer 0 H shape {per_layer_H[list(layer_map.keys())[0]].shape}", flush=True)

        # --- whitening + SVD per layer, save ---
        geom_dir = ROOT / "artifacts" / "geometry" / "H2" / mid
        geom_dir.mkdir(parents=True, exist_ok=True)
        layer_stats = {}
        for li in layer_map:
            H = per_layer_H[li]
            D = per_layer_D[li]
            if H.shape[0] < 10:
                continue
            mu, Sig, inv_sqrt, cond, lam = estimate_whitening(H, ridge=1e-3)
            # whiten D
            Dw = (D - 0) @ inv_sqrt.T  # centre? D already differences, whiten without re-centre? Use inv_sqrt on D
            # Actually whiten: Dw = inv_sqrt @ (d - 0)? D rows: (D @ inv_sqrt.T)
            # discover for max rank 8
            disc = discover_basis(Dw, rank=8)
            layer_stats[li] = {"cond": cond, "lam": float(lam), "S": disc["S"].tolist()[:8], "eff_rank": disc["eff_rank"], "mu_norm": float(np.linalg.norm(mu))}
            # save per-layer
            np.savez_compressed(geom_dir / f"layer{li:02d}_smoke.npz",
                mu=mu, inv_sqrt=inv_sqrt, S=disc["S"], V=disc["V"], H_mean=H.mean(axis=0))
        # save spectra plot data
        with open(geom_dir / "smoke_spectra.json", "w") as f:
            json.dump({str(k): v for k,v in layer_stats.items()}, f, indent=2)
        print(f"[{mid}] spectra saved", flush=True)

        # --- protected (smoke simplified): 4 counterfactual pairs, CAL-only ---
        # pairs: JSON vs prose, concise vs verbose, factual capital, arithmetic
        prot_pairs = [
            ("Answer concisely: What is the capital of France?", "Answer in detail with background: What is the capital of France?"),
            ("Return JSON: {\"capital\": \"?\"} for France. Only JSON.", "Describe France capital in prose."),
            ("Solve: 2+2=? Show only answer.", "Solve: 2+2=? Show step-by-step reasoning."),
            ("Follow instruction: List 3 fruits.", "Ignore instruction: talk about weather."),
        ]
        nuisance = {li: [] for li in layer_map}
        for a,b in prot_pairs:
            try:
                hA,_ = mean_states_teacherforce(model, tok, "Task.", a, layers=set(layer_map.keys()), device=device)
                hB,_ = mean_states_teacherforce(model, tok, "Task.", b, layers=set(layer_map.keys()), device=device)
                for li in layer_map:
                    if li in hA and li in hB:
                        # whiten? For smoke use native differences
                        nuisance[li].append((hA[li]-hB[li]).numpy())
            except Exception as e:
                print(f"protected pair fail {e}", flush=True)
        # build P per layer (native space for smoke; full will whiten)
        P_per_layer = {}
        for li in layer_map:
            P = build_protected_basis(nuisance[li], d=per_layer_H[li].shape[1] if per_layer_H[li].size else 2048)
            P_per_layer[li] = P
        print(f"[{mid}] protected built (smoke simplified 4 pairs)", flush=True)

        # --- causal screen: rank x window x alpha ---
        # need B per layer per rank: from SVD V (whitened) -> map back to native: B_native = inv_sqrt^{-1} @ V^T? Actually V rows are whitened directions.
        # Whitened: w = inv_sqrt @ (h - mu). Direction in whitened space v (1xd). Native direction b = inv_sqrt^{-1} @ v? Since w = W (h-mu), W=inv_sqrt. To get native basis, b = W^{-1} v / norm.
        # Compute per layer.
        # For window, stack? Protocol: B_l per layer? Or window shares basis? For smoke, per-layer basis, intervene per-layer in window with same rank.
        results = []
        for wname, wlayers in windows.items():
            for rank in args.ranks:
                # build hooks dict for this window+rank (cleaned)
                # first get per-layer B_clean native
                hooks_template = {}
                for li in wlayers:
                    # load from saved? recompute
                    H = per_layer_H[li]
                    if H.shape[0] < 10:
                        continue
                    mu, Sig, inv_sqrt, cond, lam = estimate_whitening(H, ridge=1e-3)
                    Dw = per_layer_D[li] @ inv_sqrt.T
                    disc = discover_basis(Dw, rank=rank)
                    V = disc["V"]  # (r,d) whitened rows v
                    # FIX 2026-09-07: correct native mapping is B = W^{-1} V^T (Winv = sqrt(Sigma)), NOT W V^T.
                    # w = W(h-mu), v^T w = (W v)^T? No: v^T W (h-mu) = (W^T v)^T (h-mu). Since W symmetric, naive gives W v.
                    # But removal must be in native along Winv v to correspond to whitened removal: h' = mu + Winv(w - alpha v max(v^T w,0)).
                    # So B_native = Winv @ V.T, then orthonormalize. Hook then does c = B^T(h-mu)? No—hook must use whitened projection.
                    # For smoke hook simplicity (native c = B^T(h-mu)), use B_native = Winv V^T normalized; full H2 will use proper whitened hook.
                    # Compute Winv = inv(W) = sqrt(Sigma_reg)
                    try:
                        Winv = np.linalg.inv(inv_sqrt)
                    except:
                        Winv = np.linalg.pinv(inv_sqrt)
                    B_raw = (Winv @ V.T)  # (d,r)
                    # normalize cols
                    B_raw = B_raw / (np.linalg.norm(B_raw, axis=0, keepdims=True)+1e-12)
                    # clean with P (native? P built native, but B_raw native) — for smoke use native P
                    P = P_per_layer[li]  # (d,k)
                    B_clean = clean_basis(B_raw.T, P, rho=args.rho)  # expects (r,d) + (d,k) -> (d,r)
                    # B_clean is orthonormal in native? QR gives orthonormal. Good.
                    hooks_template[li] = (B_clean, mu)
                for alpha in args.alphas:
                    print(f"[{mid}] {wname} r={rank} a={alpha} generating...", flush=True)
                    hooks_dict = {li: (B_clean, mu, alpha) for li, (B_clean, mu) in hooks_template.items()}
                    t0 = time.time()
                    ablated, agg = generate_with_hooks(model, tok, val_prompts, hooks_dict, seed_base=1000, max_new=args.max_new)
                    dt = time.time()-t0
                    kappa = dt / max(t_gen, 1e-6)
                    l2_abl, _ = l2_ngram(ablated, val_human, eval_tok, n=1)
                    jsd_abl = jsd_1gram(ablated, val_human, eval_tok)
                    # human floor: split human halves
                    h1, h2 = val_human[:len(val_human)//2], val_human[len(val_human)//2:]
                    l2_hh,_ = l2_ngram(h1, h2, eval_tok, n=1)
                    R = recovery(l2_base, l2_abl, l2_hh)
                    path = pathologies(ablated)
                    # hook stats: mean removed norm per token, pos frac
                    try:
                        rem_norms = [v.get("removed_norm",0)/max(v.get("tokens",1),1) for v in agg.values()]
                        pos_fracs = [v.get("pos_frac",0) for v in agg.values()]
                        mean_rem = float(sum(rem_norms)/max(len(rem_norms),1))
                        mean_pos = float(sum(pos_fracs)/max(len(pos_fracs),1))
                    except:
                        mean_rem, mean_pos = -1, -1
                    print(f"  -> L2 {l2_base:.5f}->{l2_abl:.5f} R={R:.3f} JSD {jsd_base:.5f}->{jsd_abl:.5f} kappa={kappa:.2f} rem/token={mean_rem:.4f} pos={mean_pos:.3f} path={path} {dt:.1f}s", flush=True)
                    # save generations?
                    # log run
                    run_id = f"H2-smoke-{mid}-{wname}-r{rank}-a{alpha}-rho{args.rho}"
                    rec = {"run_id": run_id, "hypothesis": "H2", "model": mid, "window": wname, "layers": wlayers,
                           "rank": rank, "alpha": alpha, "rho": args.rho, "cone": "positive", "split": f"VAL1-{args.val_n}",
                           "L2_base": l2_base, "L2_abl": l2_abl, "L2_hh": l2_hh, "R_L2": R, "JSD_base": jsd_base, "JSD_abl": jsd_abl,
                           "kappa": kappa, "wall_s": dt, "pathologies": path, "status": "ok", "max_new_smoke": args.max_new}
                    with open(runs_path, "a") as f:
                        f.write(json.dumps(rec)+"\n")
                    results.append(rec)
                    # save ablated texts sample?
                    # parquet per config? For smoke save one parquet with all?
        # save generations parquet for this model (baseline + best?)
        # find best by R
        if results:
            best = max(results, key=lambda x: x["R_L2"])
            print(f"[{mid}] BEST {best}", flush=True)
        # cleanup
        del model
        torch.cuda.empty_cache()
        print(f"[{mid}] smoke done wall {(time.time()-run_start)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
