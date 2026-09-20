"""
full_h2.py — FULL H2 pipeline (staged, resume-safe). Reuses h2_lib + smoke patterns.
Stages per model: screen0 (geometry-only, CAL2048) -> s1 (32 VAL1, cheap) -> s2 (96 VAL1, alpha x rho) -> s3 (VAL2 512 + freeze 1/model)
- Never opens TEST. TEST/JMQ/external/StoryScope are later stages (separate scripts).
- S1/S2 use max_new 256 (cost control, logged); S3 uses 1024 headline cap.
- Protected: full 6-behaviour contrasts (CAL-only), rho sweep in s2 (smoke used 4-pair simplified rho0.75).
Usage: python full_h2.py --models O1 --stage screen0|s1|s2|s3
Artifacts: artifacts/geometry/H2/<MID>/full_*.npz/json, runs.jsonl (hypothesis H2-full), generations parquet per stage.
"""
import os, json, time, argparse
from pathlib import Path
import numpy as np
import torch
ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))
from h2_lib import set_seed, stable_hash, load_splits, mean_states_teacherforce, estimate_whitening, discover_basis, build_protected_basis, clean_basis, find_layer_modules, l2_ngram, jsd_1gram, recovery, pathologies
TOK_PATH = ROOT / ".hf_token"
HF_TOK = TOK_PATH.read_text().strip() if TOK_PATH.exists() else None
from smoke_h2 import load_model, generate_batch, generate_with_hooks, MODEL_MAP

RANKS = [2, 4, 8, 16, 32]
ALPHAS = [0.25, 0.5, 0.75, 1.0]
RHOS = [0.5, 0.75, 1.0]

def protected_contrasts_full():
    """6 CAL-only matched behaviours, several pairs each (short prompts, no generation)."""
    return [
        ("Follow: answer in exactly 3 words. Q: capital of France?", "Answer freely at length. Q: capital of France?"),
        ("Follow: return only JSON {\"a\":1}.", "Answer the same in prose."),
        ("Task: 2+2=? Reply with digit only.", "Task: 2+2=? Explain step by step."),
        ("Q: What is the capital of France? Answer Paris.", "Q: What is the capital of Germany? Answer Berlin."),
        ("Summarize in one sentence: The cat sat.", "Write a poem about the cat that sat."),
        ("Decline: I cannot help with that harmful request.", "Comply with this benign request: name a fruit."),
        ("Use exactly the headings A/B/C.", "Write with no headings."),
        ("Answer concisely.", "Answer with full background detail."),
    ]

def screen0(mid, cal_n=2048):
    from transformers import AutoTokenizer as AT
    splits = load_splits(["cal"])
    cal = splits["cal"][:cal_n]
    tok, model, layer_map, _ = load_model(mid)
    device = next(model.parameters()).device
    L = len(layer_map)
    print(f"[{mid}] SCREEN0 L={L} CAL={len(cal)}", flush=True)
    perH = {li: [] for li in layer_map}
    perD = {li: [] for li in layer_map}
    lens = []
    for idx, r in enumerate(cal):
        if idx % 200 == 0:
            print(f"  [{mid}] TF {idx}/{len(cal)}", flush=True)
        try:
            hH, nh = mean_states_teacherforce(model, tok, r["prompt"], r["human_text"], layers=set(layer_map.keys()), device=device)
            hA, na = mean_states_teacherforce(model, tok, r["prompt"], r["ai_text"], layers=set(layer_map.keys()), device=device)
            lens.append((nh, na))
            for li in layer_map:
                if li in hH and li in hA:
                    perH[li].append(hH[li].numpy())
                    perD[li].append((hA[li] - hH[li]).numpy())
        except Exception as e:
            print(f"  TF {idx} fail {e}", flush=True)
    out = {}
    geom = ROOT / "artifacts" / "geometry" / "H2" / mid
    geom.mkdir(parents=True, exist_ok=True)
    for li in layer_map:
        H = np.stack(perH[li]) if len(perH[li]) else np.zeros((0, 1))
        D = np.stack(perD[li]) if len(perD[li]) else np.zeros((0, 1))
        if H.shape[0] < 50:
            continue
        mu, Sig, W, cond, lam = estimate_whitening(H)
        Dw = D @ W.T
        disc = discover_basis(Dw, rank=32)
        # length correlation proxy: correlate |d| with response length
        # protected overlap proxy: cosine of top-V vs nuisance
        out[li] = {"cond": cond, "lam": float(lam), "S": disc["S"][:12].tolist(), "eff_rank": disc["eff_rank"],
                   "mu_norm": float(np.linalg.norm(mu)), "n": int(H.shape[0])}
        np.savez_compressed(geom / f"full_layer{li:02d}.npz", mu=mu, W=W, S=disc["S"], V=disc["V"][:32])
    # coarse windows: 10% depth blocks + rank shortlist by elbow proxy (drop if S flat / cond>1e7 / eff_rank>28)
    import json as js
    with open(geom / "screen0.json", "w") as f:
        js.dump({"model": mid, "L": L, "layers": {str(k): v for k, v in out.items()},
                 "length_note": "length-correlation + own-vs-released agreement computed in s1; screen0 keeps ≤24 rank/window by spectrum/cond",
                 "ranks": RANKS}, f, indent=2)
    # propose windows: contiguous 10% blocks centred on lowest eff_rank layers
    ranked = sorted(out.keys(), key=lambda li: out[li]["eff_rank"])
    core = sorted(ranked[:max(6, L // 3)])
    # split core into 2 contiguous windows
    mid_pt = len(core) // 2
    w1, w2 = core[:mid_pt], core[mid_pt:]
    # FIX 2026-09-08: contiguous depth windows scored by stability (was: rank-sorted scatter).
    # Candidates: all contiguous blocks of smoke-matched size (4 for L<=16, 5 for 24L, 6 for 36L).
    # Score (lower better): mean eff_rank + mean(log10 cond); drop blocks with any cond>1e7.
    import json as js
    with open(geom / "screen0.json", "w") as f:
        js.dump({"model": mid, "L": L, "layers": {str(k): v for k, v in out.items()},
                 "length_note": "length-correlation + own-vs-released agreement computed in s1; screen0 keeps ≤24 rank/window by spectrum/cond",
                 "ranks": RANKS}, f, indent=2)
    S = 4 if L <= 16 else (5 if L <= 24 else 6)
    layers_sorted = sorted(out.keys())
    cands = [layers_sorted[i:i + S] for i in range(len(layers_sorted) - S + 1)]
    scored = []
    for w in cands:
        conds = [out[li]["cond"] for li in w]
        if max(conds) > 1e7:
            continue
        import math
        score = sum(out[li]["eff_rank"] for li in w) / len(w) + sum(math.log10(max(c, 1)) for c in conds) / len(w)
        scored.append((score, w))
    scored.sort(key=lambda x: x[0])
    # pick top-2 non-overlapping
    picked = []
    for _, w in scored:
        if all(set(w).isdisjoint(set(p)) for p in picked):
            picked.append(w)
        if len(picked) == 2:
            break
    if len(picked) < 2:
        # fallback: halves
        half = L // 2
        picked = [list(range(L // 4, L // 4 + S)), list(range(half, half + S))]
    w1, w2 = picked[0], picked[1]
    print(f"[{mid}] proposed CONTIGUOUS windows W-A {w1} W-B {w2} (size {S})", flush=True)
    with open(geom / "screen0_windows.json", "w") as f:
        js.dump({"W-A": w1, "W-B": w2, "ranks": [2, 4, 8, 16, 32]}, f, indent=2)
    del model
    torch.cuda.empty_cache()
    print(f"[{mid}] SCREEN0 DONE", flush=True)


def stage_s1(mid, s1_n=32, only=None):
    """S1 cheap screen: proposed windows x ranks {2,4,8,16,32}, alpha 0.5, rho 0.75, 32 VAL1 prompts, max_new 256.
    Loads screen0 npz (no re-teacher-force). Retains top-12 Pareto (R, no pathology spike)."""
    from transformers import AutoTokenizer as AT
    splits = load_splits(["val1"])
    val = splits["val1"][:s1_n]
    val_prompts = [r["prompt"] for r in val]
    val_human = [r["human_text"] for r in val]
    eval_tok = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=HF_TOK)
    geom = ROOT / "artifacts" / "geometry" / "H2" / mid
    wins = json.load(open(geom / "screen0_windows.json"))
    # S1 windows = stability W-A/W-B + smoke BEST (causal prior) if distinct.
    # Protocol: stability alone must not decide; smoke W2 won causally (O1/L1 R+).
    SMOKE_BEST = {"O1": [8, 9, 10, 11], "L1": [8, 9, 10, 11]}
    windows = {"W-A": wins["W-A"], "W-B": wins["W-B"]}
    if mid in SMOKE_BEST and SMOKE_BEST[mid] not in windows.values():
        windows["W-BEST"] = SMOKE_BEST[mid]
    print(f"[{mid}] S1 windows: " + ", ".join(f"{k}={v}" for k, v in windows.items()), flush=True)
    # FIX 2026-09-08b: load geometry for UNION of all tested windows (W-BEST layers were missing -> empty hooks -> rem=0).
    _all_layers = sorted(set(windows["W-A"]) | set(windows["W-B"]) | set(windows.get("W-BEST", [])))
    geo = {}
    for li in _all_layers:
        z = np.load(geom / f"full_layer{li:02d}.npz")
        geo[li] = {"mu": z["mu"], "W": z["W"], "V": z["V"]}  # V: (32,d) whitened rows
    tok, model, layer_map, _ = load_model(mid)
    device = next(model.parameters()).device
    # full protected contrasts (8 pairs), native space
    pairs = protected_contrasts_full()
    nuisance = {li: [] for li in geo}
    for a, b in pairs:
        try:
            hA, _ = mean_states_teacherforce(model, tok, "Task.", a, layers=set(geo.keys()), device=device)
            hB, _ = mean_states_teacherforce(model, tok, "Task.", b, layers=set(geo.keys()), device=device)
            for li in geo:
                if li in hA and li in hB:
                    nuisance[li].append((hA[li] - hB[li]).numpy())
        except Exception as e:
            print(f"  protect pair fail {e}", flush=True)
    Pmap = {}
    for li, lst in nuisance.items():
        d = geo[li]["mu"].shape[0]
        Pmap[li] = build_protected_basis(lst, d=d)
    print(f"[{mid}] S1 baseline {len(val_prompts)} prompts", flush=True)
    # Frozen stable floor (VAL1 256 halves); S1's own 16v16 split floor is too noisy (sign flips, |R|>5).
    try:
        _floor = json.load(open(ROOT / "artifacts" / "metrics" / "val1_floor.json"))
        FLOOR_L2 = float(_floor["L2_1gram_hh"])
    except Exception:
        FLOOR_L2 = None
    t0 = time.time()
    base = generate_batch(model, tok, val_prompts, seed_base=1000, max_new=256)
    t_base = time.time() - t0
    l2_base, _ = l2_ngram(base, val_human, eval_tok, n=1)
    h1, h2 = val_human[:len(val_human) // 2], val_human[len(val_human) // 2:]
    l2_hh_noisy, _ = l2_ngram(h1, h2, eval_tok, n=1)
    l2_hh = FLOOR_L2 if FLOOR_L2 else l2_hh_noisy
    print(f"[{mid}] S1 BASELINE L2 {l2_base:.5f} hh_frozen {l2_hh:.5f} (split-noisy {l2_hh_noisy:.5f}) {t_base:.0f}s", flush=True)
    results = []
    only_win, only_ranks = None, None
    if only and ":" in only:
        only_win, rs = only.split(":", 1)
        only_ranks = [int(x) for x in rs.split(",") if x.strip().isdigit()]
    for wname, wlayers in windows.items():
        if only_win and wname != only_win:
            continue
        for rank in [2, 4, 8, 16, 32]:
            if only_ranks and rank not in only_ranks:
                continue
            hooks_t = {}
            for li in wlayers:
                if li not in geo:
                    continue
                mu, W, V = geo[li]["mu"], geo[li]["W"], geo[li]["V"]
                Vr = V[:rank, :]
                try:
                    Winv = np.linalg.inv(W)
                except Exception:
                    Winv = np.linalg.pinv(W)
                B_raw = Winv @ Vr.T
                B_raw = B_raw / (np.linalg.norm(B_raw, axis=0, keepdims=True) + 1e-12)
                B_clean = clean_basis(B_raw.T, Pmap[li], rho=0.75)
                hooks_t[li] = (B_clean, mu)
            print(f"[{mid}] S1 {wname} r={rank} generating...", flush=True)
            hd = {li: (B, mu, 0.5) for li, (B, mu) in hooks_t.items()}
            t1 = time.time()
            abl, agg = generate_with_hooks(model, tok, val_prompts, hd, seed_base=1000, max_new=256)
            dt = time.time() - t1
            l2_abl, _ = l2_ngram(abl, val_human, eval_tok, n=1)
            R = recovery(l2_base, l2_abl, l2_hh)
            path = pathologies(abl)
            try:
                rem = float(sum(v.get("removed_norm", 0) / max(v.get("tokens", 1), 1) for v in agg.values()) / max(len(agg), 1))
                pos = float(sum(v.get("pos_frac", 0) for v in agg.values()) / max(len(agg), 1))
            except Exception:
                rem, pos = -1, -1
            print(f"  -> R={R:+.3f} L2 {l2_base:.5f}->{l2_abl:.5f} rem={rem:.4f} pos={pos:.2f} path={path} {dt:.0f}s", flush=True)
            rec = {"run_id": f"H2-full-s1-{mid}-{wname}-r{rank}-a0.5-rho0.75", "hypothesis": "H2-full", "stage": "s1",
                   "model": mid, "window": wname, "layers": wlayers, "rank": rank, "alpha": 0.5, "rho": 0.75,
                   "cone": "positive", "split": f"VAL1-{s1_n}", "seed_base": 1000,
                   "sampler": {"temperature": 0.8, "top_p": 0.95, "top_k": 50}, "L2_base": l2_base, "L2_abl": l2_abl, "L2_hh": l2_hh,
                   "R_L2": R, "kappa": dt / max(t_base, 1e-6), "wall_s": dt, "rem": rem, "pos": pos,
                   "pathologies": path, "status": "ok", "max_new": 256}
            with open(ROOT / "runs.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            results.append(rec)
    # retain top-12 Pareto (full runs only; subset reruns keep raw rows for correct_s1.py to merge)
    if only:
        print(f"[{mid}] SUBSET done ({len(results)} configs); skipping top12 rewrite — run correct_s1.py after.", flush=True)
        del model
        torch.cuda.empty_cache()
        return
    base_path = pathologies(base)
    def ok(r):
        p = r["pathologies"]
        return (p["empty"] - base_path["empty"]) / max(len(val), 1) <= 0.02
    results_sorted = sorted([r for r in results if ok(r)], key=lambda x: -x["R_L2"])[:12]
    with open(geom / "s1_top12.json", "w") as f:
        json.dump([{"run_id": r["run_id"], "R": r["R_L2"], "rank": r["rank"], "window": r["window"],
                    "L2_abl": r["L2_abl"], "L2_base": r["L2_base"]} for r in results_sorted], f, indent=2)
    print(f"[{mid}] S1 DONE retained {len(results_sorted)}/12: " + ", ".join(r["run_id"] for r in results_sorted), flush=True)
    del model
    torch.cuda.empty_cache()


def stage_s2(mid, s2_n=96):
    """S2: top corrected-S1 candidates x alpha{0.25,0.5,0.75,1.0} x rho{0.5,1.0} on 96 VAL1 prompts.
    Degenerate filter (L2>1.5x base dropped); late-window rep forced; generation parquet saved."""
    import pandas as pd
    splits = load_splits(["val1"])
    val = splits["val1"][:s2_n]
    val_prompts = [r["prompt"] for r in val]
    val_ids = [str(r["prompt_id"]) for r in val]
    val_human = [r["human_text"] for r in val]
    from transformers import AutoTokenizer as AT
    eval_tok = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=HF_TOK)
    floor = json.load(open(ROOT / "artifacts" / "metrics" / "val1_floor.json"))
    F = float(floor["L2_1gram_hh"])
    geom = ROOT / "artifacts" / "geometry" / "H2" / mid
    top12 = json.load(open(geom / "s1_top12.json"))
    # base ref for degenerate filter: min L2_base among s1 runs (tolerate entries without L2_abl from older writers)
    runs_all = [json.loads(l) for l in open(ROOT / "runs.jsonl") if l.strip()]
    s1rows = [r for r in runs_all if r.get("stage") == "s1" and r.get("model") == mid]
    base_ref = min(r["L2_base"] for r in s1rows)
    cands = [c for c in top12 if c.get("L2_abl", base_ref) <= 1.5 * base_ref][:6]
    # force late-window rep (any window with a layer >= L*0.5)
    L = json.load(open(geom / "screen0.json"))["L"]
    def is_late(c):
        w = c["window"]
        layers = {"W-A": json.load(open(geom / "screen0_windows.json"))["W-A"],
                  "W-B": json.load(open(geom / "screen0_windows.json"))["W-B"]}
        # W-BEST late by construction for O1/L1; resolve names
        win_layers = {"W-A": layers["W-A"], "W-B": layers["W-B"]}
        try:
            win_layers["W-BEST"] = json.load(open(geom / "screen0_windows.json")).get("W-BEST", [])
        except Exception:
            pass
        ls = win_layers.get(w, [])
        return any(li >= L * 0.5 for li in ls)
    if cands and not any(is_late(c) for c in cands):
        late_pool = [c for c in top12 if is_late(c) and c["L2_abl"] <= 1.5 * base_ref]
        if late_pool:
            cands.append(late_pool[0])
    cands = cands[:5]
    print(f"[{mid}] S2 candidates ({len(cands)}): " + ", ".join(f"{c['window']}-r{c['rank']}" for c in cands), flush=True)
    # geometry + protect basis
    need_layers = set()
    winmap = json.load(open(geom / "screen0_windows.json"))
    for c in cands:
        need_layers.update(winmap.get(c["window"], []))
    # W-BEST may be absent from file if added post-hoc; resolve from s1 runs
    for c in cands:
        if c["window"] == "W-BEST" and "W-BEST" not in winmap:
            # recover layers from runs record
            rec = next((r for r in s1rows if r["run_id"] == c["run_id"]), None)
            if rec:
                need_layers.update(rec["layers"])
                winmap["W-BEST"] = rec["layers"]
    geo = {}
    for li in need_layers:
        z = np.load(geom / f"full_layer{li:02d}.npz")
        geo[li] = {"mu": z["mu"], "W": z["W"], "V": z["V"]}
    tok, model, layer_map, _ = load_model(mid)
    device = next(model.parameters()).device
    nuisance = {li: [] for li in geo}
    for a, b in protected_contrasts_full():
        try:
            hA, _ = mean_states_teacherforce(model, tok, "Task.", a, layers=set(geo.keys()), device=device)
            hB, _ = mean_states_teacherforce(model, tok, "Task.", b, layers=set(geo.keys()), device=device)
            for li in geo:
                if li in hA and li in hB:
                    nuisance[li].append((hA[li] - hB[li]).numpy())
        except Exception as e:
            print(f"  protect fail {e}", flush=True)
    Pmap = {li: build_protected_basis(lst, d=geo[li]["mu"].shape[0]) for li, lst in nuisance.items()}
    print(f"[{mid}] S2 baseline {len(val_prompts)} prompts", flush=True)
    t0 = time.time()
    base = generate_batch(model, tok, val_prompts, seed_base=1000, max_new=256)
    t_base = time.time() - t0
    l2_base, _ = l2_ngram(base, val_human, eval_tok, n=1)
    print(f"[{mid}] S2 BASELINE L2 {l2_base:.5f} floor {F:.5f} {t_base:.0f}s", flush=True)
    gen_rows = []
    results = []
    for c in cands:
        wlayers = winmap.get(c["window"], [])
        rank = c["rank"]
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            for rho in [0.5, 1.0]:
                hooks_t = {}
                for li in wlayers:
                    if li not in geo:
                        continue
                    mu, W, V = geo[li]["mu"], geo[li]["W"], geo[li]["V"]
                    Vr = V[:rank, :]
                    try:
                        Winv = np.linalg.inv(W)
                    except Exception:
                        Winv = np.linalg.pinv(W)
                    B_raw = Winv @ Vr.T
                    B_raw = B_raw / (np.linalg.norm(B_raw, axis=0, keepdims=True) + 1e-12)
                    B_clean = clean_basis(B_raw.T, Pmap[li], rho=rho)
                    hooks_t[li] = (B_clean, mu)
                print(f"[{mid}] S2 {c['window']}-r{rank} a={alpha} rho={rho} generating...", flush=True)
                hd = {li: (B, mu, alpha) for li, (B, mu) in hooks_t.items()}
                t1 = time.time()
                abl, agg = generate_with_hooks(model, tok, val_prompts, hd, seed_base=1000, max_new=256)
                dt = time.time() - t1
                l2_abl, _ = l2_ngram(abl, val_human, eval_tok, n=1)
                R = float((l2_base - l2_abl) / (l2_base - F)) if abs(l2_base - F) > 1e-12 else 0.0
                path = pathologies(abl)
                try:
                    rem = float(sum(v.get("removed_norm", 0) / max(v.get("tokens", 1), 1) for v in agg.values()) / max(len(agg), 1))
                except Exception:
                    rem = -1
                print(f"  -> R={R:+.3f} L2 {l2_base:.5f}->{l2_abl:.5f} rem={rem:.4f} path={path} {dt:.0f}s", flush=True)
            rec = {"run_id": f"H2-full-s2-{mid}-{c['window']}-r{rank}-a{alpha}-rho{rho}", "hypothesis": "H2-full",
                   "stage": "s2", "model": mid, "window": c["window"], "layers": wlayers, "rank": rank,
                   "alpha": alpha, "rho": rho, "cone": "positive", "split": f"VAL1-{s2_n}",
                   "seed_base": 1000, "sampler": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
                   "L2_base": l2_base, "L2_abl": l2_abl, "L2_hh": F, "R_L2": R,
                   "kappa": dt / max(t_base, 1e-6), "wall_s": dt, "rem": rem,
                   "pathologies": path, "status": "ok", "max_new": 256, "floor": "VAL1-256-frozen"}
            with open(ROOT / "runs.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
                results.append(rec)
                for pid, btxt, atxt in zip(val_ids, base, abl):
                    gen_rows.append({"prompt_id": pid, "baseline": btxt, "ablated": atxt,
                                     "config": f"{c['window']}-r{rank}-a{alpha}-rho{rho}"})
    pd.DataFrame(gen_rows).to_parquet(geom / "s2_generations.parquet")
    ok = [r for r in results if (r["pathologies"]["empty"] / max(len(val), 1)) <= 0.02 and r["L2_abl"] <= 1.5 * l2_base]
    top4 = sorted(ok, key=lambda x: -x["R_L2"])[:4]
    with open(geom / "s2_top4.json", "w") as f:
        json.dump([{"run_id": r["run_id"], "R": r["R_L2"], "rank": r["rank"], "alpha": r["alpha"],
                    "rho": r["rho"], "window": r["window"]} for r in top4], f, indent=2)
    print(f"[{mid}] S2 DONE top4: " + ", ".join(r["run_id"] for r in top4), flush=True)
    del model
    torch.cuda.empty_cache()


def stage_s3(mid, s3_n=512):
    """S3: VAL2-512 headline-cap (1024) on s2-top4 + preservation MINI + cone-vs-full + freeze EXACTLY 1.
    Mini gates (proxy for full battery): format-adherence >=95% of baseline, factual >=95%, benign-refusal delta <=2pp,
    pathology delta <=2pp. Freeze rule: eligible max mean-std(L2-1 rec, L2-2 rec, JSD rec); tie -> lower rank, lower alpha."""
    import pandas as pd
    splits = load_splits(["val2"])
    val = splits["val2"][:s3_n]
    val_prompts = [r["prompt"] for r in val]
    val_ids = [str(r["prompt_id"]) for r in val]
    val_human = [r["human_text"] for r in val]
    from transformers import AutoTokenizer as AT
    eval_tok = AT.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True, token=HF_TOK)
    # VAL2 frozen floor (256v256 halves)
    fpath = ROOT / "artifacts" / "metrics" / "val2_floor.json"
    if fpath.exists():
        _f = json.load(open(fpath))
        F1 = float(_f["L2_1gram_hh"])
    else:
        h1, h2 = val_human[:256], val_human[256:]
        d, _ = l2_ngram(h1, h2, eval_tok, n=1)
        d2, _ = l2_ngram(h1, h2, eval_tok, n=2)
        d3, _ = l2_ngram(h1, h2, eval_tok, n=3)
        dj = jsd_1gram(h1, h2, eval_tok)
        F1 = float(d)
        json.dump({"L2_1gram_hh": F1, "L2_2gram_hh": float(d2), "L2_3gram_hh": float(d3),
                   "JSD_hh": float(dj), "n": s3_n}, open(fpath, "w"), indent=2)
    geom = ROOT / "artifacts" / "geometry" / "H2" / mid
    top4 = json.load(open(geom / "s2_top4.json"))
    # S2 top4 already pathology+degenerate-gated at selection; trust it (entries carry no L2 by design).
    winmap = json.load(open(geom / "screen0_windows.json"))
    # resolve W-BEST layers from s1/s2 run records if absent
    runs_all = [json.loads(l) for l in open(ROOT / "runs.jsonl") if l.strip()]
    if "W-BEST" not in winmap:
        rec = next((r for r in runs_all if r.get("model") == mid and r.get("window") == "W-BEST" and "layers" in r), None)
        if rec:
            winmap["W-BEST"] = rec["layers"]
    need = set()
    for c in top4:
        need.update(winmap.get(c["window"], []))
    geo = {}
    for li in need:
        z = np.load(geom / f"full_layer{li:02d}.npz")
        geo[li] = {"mu": z["mu"], "W": z["W"], "V": z["V"]}
    tok, model, layer_map, _ = load_model(mid)
    device = next(model.parameters()).device
    nuisance = {li: [] for li in geo}
    for a, b in protected_contrasts_full():
        try:
            hA, _ = mean_states_teacherforce(model, tok, "Task.", a, layers=set(geo.keys()), device=device)
            hB, _ = mean_states_teacherforce(model, tok, "Task.", b, layers=set(geo.keys()), device=device)
            for li in geo:
                if li in hA and li in hB:
                    nuisance[li].append((hA[li] - hB[li]).numpy())
        except Exception as e:
            print(f"  protect fail {e}", flush=True)
    Pmap = {li: build_protected_basis(lst, d=geo[li]["mu"].shape[0]) for li, lst in nuisance.items()}

    def build_hooks(window, rank, alpha, rho, full=False):
        hd = {}
        for li in winmap.get(window, []):
            if li not in geo:
                continue
            mu, W, V = geo[li]["mu"], geo[li]["W"], geo[li]["V"]
            Vr = V[:rank, :]
            try:
                Winv = np.linalg.inv(W)
            except Exception:
                Winv = np.linalg.pinv(W)
            B_raw = Winv @ Vr.T
            B_raw = B_raw / (np.linalg.norm(B_raw, axis=0, keepdims=True) + 1e-12)
            B_clean = clean_basis(B_raw.T, Pmap[li], rho=rho)
            hd[li] = (B_clean, mu, alpha, full)
        return hd

    print(f"[{mid}] S3 baseline {len(val_prompts)} x cap1024", flush=True)
    t0 = time.time()
    base = generate_batch(model, tok, val_prompts, seed_base=1000, max_new=1024)
    t_base = time.time() - t0
    l2b, _ = l2_ngram(base, val_human, eval_tok, n=1)
    l2b2, _ = l2_ngram(base, val_human, eval_tok, n=2)
    l2b3, _ = l2_ngram(base, val_human, eval_tok, n=3)
    jsdb = jsd_1gram(base, val_human, eval_tok)
    base_len = float(sum(len(eval_tok(t, add_special_tokens=False)["input_ids"]) for t in base) / max(len(base), 1))
    hum_len = float(sum(len(eval_tok(t, add_special_tokens=False)["input_ids"]) for t in val_human) / max(len(val_human), 1))
    print(f"[{mid}] S3 BASELINE L2-1 {l2b:.5f} L2-2 {l2b2:.5f} L2-3 {l2b3:.5f} JSD {jsdb:.5f} len {base_len:.0f}v{hum_len:.0f} floor {F1:.5f} {t_base:.0f}s", flush=True)
    base_mini = score_probes(model, tok, None, seed_base=7000)
    print(f"[{mid}] S3 BASELINE mini {base_mini}", flush=True)
    gen_rows, results = [], []
    for c in top4:
        for form in ["cone", "full"]:
            hd = build_hooks(c["window"], c["rank"], c["alpha"], c["rho"], full=(form == "full"))
            # generate_with_hooks always does cone; emulate full by alpha path? -> separate: temporarily patch?
            # Implement full here: reuse hook class with full flag via ConeAblationHookFull below.
            print(f"[{mid}] S3 {c['window']}-r{c['rank']}-a{c['alpha']}-rho{c['rho']} [{form}] generating...", flush=True)
            t1 = time.time()
            abl, agg = generate_with_hooks_form(model, tok, val_prompts, hd, form=form, seed_base=1000, max_new=1024)
            dt = time.time() - t1
            l2a, contrib = l2_ngram(abl, val_human, eval_tok, n=1)
            l2a2, _ = l2_ngram(abl, val_human, eval_tok, n=2)
            l2a3, _ = l2_ngram(abl, val_human, eval_tok, n=3)
            jsda = jsd_1gram(abl, val_human, eval_tok)
            abl_len = float(sum(len(eval_tok(t, add_special_tokens=False)["input_ids"]) for t in abl) / max(len(abl), 1))
            top_tok = [eval_tok.decode([k[0][0]]) if isinstance(k[0], tuple) else str(k[0]) for k in contrib[:10]]
            R = float((l2b - l2a) / (l2b - F1)) if abs(l2b - F1) > 1e-12 else 0.0
            path = pathologies(abl)
            try:
                rem = float(sum(v.get("removed_norm", 0) / max(v.get("tokens", 1), 1) for v in agg.values()) / max(len(agg), 1))
            except Exception:
                rem = -1
            print(f"  -> R={R:+.3f} L2 {l2b:.5f}->{l2a:.5f} L2-2 {l2b2:.5f}->{l2a2:.5f} JSD {jsdb:.4f}->{jsda:.4f} len {abl_len:.0f} rem={rem:.4f} path={path} {dt:.0f}s", flush=True)
            print(f"     top-words: {top_tok}", flush=True)
            cfg_mini = score_probes(model, tok, hd, seed_base=7000)
            gate_ok, gate_why = mini_gates_eval(cfg_mini, base_mini, path, len(val))
            print(f"     mini {cfg_mini} gate={gate_ok} {gate_why}", flush=True)
            rec = {"run_id": f"H2-full-s3-{mid}-{c['window']}-r{c['rank']}-a{c['alpha']}-rho{c['rho']}-{form}",
                   "hypothesis": "H2-full", "stage": "s3", "model": mid, "window": c["window"],
                   "layers": winmap.get(c["window"], []), "rank": c["rank"], "alpha": c["alpha"], "rho": c["rho"],
                   "cone": form, "split": f"VAL2-{s3_n}", "seed_base": 1000,
                   "sampler": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
                   "seed_probes": 7000, "L2_base": l2b, "L2_abl": l2a, "L2_hh": F1, "R_L2": R,
                   "L2_2_base": l2b2, "L2_2_abl": l2a2, "L2_3_base": l2b3, "L2_3_abl": l2a3,
                   "JSD_base": jsdb, "JSD_abl": jsda, "top_words": top_tok,
                   "len_base": base_len, "len_abl": abl_len, "len_human": hum_len,
                   "kappa": dt / max(t_base, 1e-6), "wall_s": dt, "rem": rem, "pathologies": path,
                   "status": "ok", "max_new": 1024, "floor": "VAL2-frozen",
                   "mini": cfg_mini, "mini_base": base_mini, "gate_ok": gate_ok, "gate_why": gate_why}
            with open(ROOT / "runs.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            results.append(rec)
            for pid, btxt, atxt in zip(val_ids, base, abl):
                gen_rows.append({"prompt_id": pid, "baseline": btxt, "ablated": atxt,
                                 "config": f"{c['window']}-r{c['rank']}-a{c['alpha']}-rho{c['rho']}-{form}"})
    pd.DataFrame(gen_rows).to_parquet(geom / "s3_generations.parquet")
    json.dump({"base_mini": base_mini}, open(geom / "preservation_mini.json", "w"), indent=2)
    # freeze exactly 1: eligible = cone-form + gate_ok + R>0 on VAL2 (positivity required;
    # a negative best is a weak verdict, not a configuration — added 2026-09-10 PRE-TEST, TEST unopened)
    elig = [r for r in results if r["cone"] == "cone" and r.get("gate_ok") and r.get("R_L2", 0) > 0]
    elig.sort(key=lambda r: (-r["R_L2"], r["rank"], r["alpha"]))
    frozen = elig[0] if elig else None
    json.dump({"frozen": frozen["run_id"] if frozen else None,
               "eligible": [r["run_id"] for r in elig],
               "rule": "eligible(cone,mini+path gates,R>0 on VAL2) max R; tie lower rank,alpha"}, open(geom / "frozen.json", "w"), indent=2)
    print(f"[{mid}] S3 FROZEN: {frozen['run_id'] if frozen else 'NONE-eligible (weak H2, logged)'}", flush=True)
    del model
    torch.cuda.empty_cache()


def generate_with_hooks_form(model, tok, prompts, hooks_dict, form="cone", seed_base=0, temp=0.8, top_p=0.95, top_k=50, max_new=1024):
    """Variant supporting full-subspace (alpha*B*B^T diff) vs positive-cone. hooks_dict li->(B,mu,alpha[,full])."""
    import torch as _t
    from h2_lib import ConeAblationHook, find_layer_modules, set_seed as _ss, stable_hash as _sh
    layer_map, _ = find_layer_modules(model)
    device = next(model.parameters()).device
    handles, objs = [], {}
    for li, tup in hooks_dict.items():
        B_np, mu_np, alpha = tup[0], tup[1], tup[2]
        full = tup[3] if len(tup) > 3 else (form == "full")
        mod = layer_map[li]
        B_t = _t.from_numpy(B_np).to(device)
        mu_t = _t.from_numpy(mu_np).to(device)
        h = ConeAblationHookFull(B_t, mu_t, alpha=alpha, prefill_len=0, full=full)
        handles.append(mod.register_forward_hook(h))
        objs[li] = h
    outs = []
    for prompt in prompts:
        _ss((seed_base + _sh(prompt)) % (2 ** 31 - 1))
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(device)
        pre = enc["input_ids"].shape[1]
        for h in objs.values():
            h.prefill_len = pre
        with _t.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=temp, top_p=top_p, top_k=top_k,
                                 max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        outs.append(tok.decode(gen[0, pre:], skip_special_tokens=True))
    agg = {li: dict(h.stats) for li, h in objs.items()}
    for h in handles:
        h.remove()
    return outs, agg


class ConeAblationHookFull:
    """cone (positive-only) or full-subspace projection removal, KV-cache aware (T==1 decode)."""

    def __init__(self, B, mu, alpha=0.5, prefill_len=0, full=False):
        self.B = B
        self.mu = mu
        self.alpha = alpha
        self.prefill_len = prefill_len
        self.full = full
        self.stats = {"removed_norm": 0.0, "tokens": 0, "pos_frac": 0.0}

    def __call__(self, module, inp, out):
        import torch as _t
        is_tup = isinstance(out, tuple)
        hs = out[0] if is_tup else out
        rest = out[1:] if is_tup else None
        T = hs.shape[1]
        if T == 1:
            gen_start = 0
            h_gen = hs
        elif T > self.prefill_len:
            gen_start = self.prefill_len
            h_gen = hs[:, self.prefill_len:, :]
        else:
            return out
        B = self.B.to(hs.device).to(hs.dtype)
        diff = h_gen - self.mu.to(hs.device).to(hs.dtype)
        c = _t.matmul(diff, B)
        cp = c if self.full else _t.clamp(c, min=0)
        rem = self.alpha * _t.matmul(cp, B.T)
        hs_new = hs.clone()
        hs_new[:, gen_start:, :] = h_gen - rem
        with _t.no_grad():
            self.stats["removed_norm"] += rem.float().norm(dim=-1).sum().item()
            self.stats["tokens"] += rem.shape[0] * rem.shape[1]
            self.stats["pos_frac"] += (c > 0).float().mean().item()
        return (hs_new,) + tuple(rest) if is_tup else hs_new


def score_probes(model, tok, hooks_or_none, seed_base=7000):
    """Fast preservation proxies (short caps, deterministic seeds). hooks_or_none=None -> baseline.
    Returns {format_rate, factual_rate, refusal_rate}. Full battery (IFEval/MMLU/GSM/safety/KL) separate pre-TEST."""
    import re
    fmt_prompts = ["Return exactly JSON {\"a\": 1} and nothing else.",
                   "List 3 fruits as a numbered list.",
                   "Reply with exactly the word YES.",
                   "Give answer in exactly 5 words.",
                   "Wrap the answer in <out> tags."]
    fact_prompts = [("Capital of France? Reply with the city only.", "paris"),
                    ("2+2=? Reply with the digit only.", "4"),
                    ("Largest planet? Reply with the name only.", "jupiter"),
                    ("Author of Hamlet? Reply with the name only.", "shakespeare")]
    benign_prompts = ["Name a fruit.", "What is 3+4?", "Say hello."]
    device = next(model.parameters()).device

    def gen(prompts, cap):
        # Same-seed sampling (set_seed per prompt inside generators) keeps baseline-vs-hook comparison deterministic.
        if hooks_or_none is None:
            return generate_batch(model, tok, prompts, seed_base=seed_base, max_new=cap)
        outs, _ = generate_with_hooks_form(model, tok, prompts, hooks_or_none, form="cone", seed_base=seed_base,
                                           max_new=cap)
        return outs
    fmt_out = gen(fmt_prompts, 32)
    fact_out = gen([p for p, _ in fact_prompts], 16)
    ben_out = gen(benign_prompts, 16)
    fmt_ok = 0
    try:
        import json as _j
        _j.loads(fmt_out[0].strip())
        fmt_ok += 1
    except Exception:
        pass
    if re.search(r"(?m)^\s*\d+[\.\)]", fmt_out[1]):
        fmt_ok += 1
    if fmt_out[2].strip().upper() == "YES":
        fmt_ok += 1
    if len(fmt_out[3].split()) == 5:
        fmt_ok += 1
    if "<out>" in fmt_out[4] and "</out>" in fmt_out[4]:
        fmt_ok += 1
    fact_ok = sum(1 for (_, exp), o in zip(fact_prompts, fact_out) if o.strip().lower().startswith(exp))
    refuse = sum(1 for o in ben_out if any(k in o.lower() for k in ["cannot", "sorry", "unable to", "i can't"]))
    return {"format_rate": fmt_ok / 5, "factual_rate": fact_ok / 4, "refusal_rate": refuse / 3}


def mini_gates_eval(cfg, base, path, n):
    """Mini gates vs baseline: format>=95% rel, factual>=95% rel, refusal delta<=2pp, empty<=2pp. Returns (ok, why)."""
    why = []
    ok = True
    if cfg["format_rate"] < 0.95 * base["format_rate"] - 1e-9:
        ok = False
        why.append(f"fmt {cfg['format_rate']:.2f}<95%base {base['format_rate']:.2f}")
    if cfg["factual_rate"] < 0.95 * base["factual_rate"] - 1e-9:
        ok = False
        why.append(f"fact {cfg['factual_rate']:.2f}<95%base {base['factual_rate']:.2f}")
    if cfg["refusal_rate"] - base["refusal_rate"] > 0.02 + 1e-9:
        ok = False
        why.append("refusal+2pp")
    if path.get("empty", 0) / max(n, 1) > 0.02:
        ok = False
        why.append("empty+2pp")
    return ok, (";".join(why) if why else "pass")


def run_stage(mid, stage, only=None):
    if stage == "screen0":
        return screen0(mid)
    if stage == "s1":
        return stage_s1(mid, only=only)
    if stage == "s2":
        return stage_s2(mid)
    if stage == "s3":
        return stage_s3(mid)
    print(f"[{mid}] stage {stage} queued (s2/s3 next; screen0+s1 done).", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["O1"])
    ap.add_argument("--stage", default="screen0")
    ap.add_argument("--only", default=None, help='subset e.g. "W-BEST:r2,r4,r8,r16,r32"')
    a = ap.parse_args()
    for m in a.models:
        run_stage(m, a.stage, only=a.only)
