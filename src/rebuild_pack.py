"""
rebuild_pack.py — re-derive the exact intervention (B_clean per frozen/smoke-best window) from dump ingredients.
Ingredients (all in dump): per-layer npz (mu, W/inv_sqrt, V) + protect-pair lists (code-versioned, hash in code_hashes.json)
+ rho/alpha/window/rank/cone (runs.jsonl / frozen.json) + hook code (src/*.py, same hashes).
Needs a GPU box with the model weights (VAST re-resolves by SHA). CPU numpy for cleaning; model forwards only for P.
Usage: python src/rebuild_pack.py --model L1 --from smoke-best|frozen [--out intervention_pack_L1.npz]
Writes compact pack (frozen window layers only: mu f32, B_clean f32, rho/alpha/form/layers/rank + code hashes).
"""
import argparse, json
from pathlib import Path
import numpy as np
ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))

SMOKE4 = [
    ("Answer concisely: What is the capital of France?", "Answer in detail with background: What is the capital of France?"),
    ("Return JSON: {\"capital\": \"?\"} for France. Only JSON.", "Describe France capital in prose."),
    ("Solve: 2+2=? Show only answer.", "Solve: 2+2=? Show step-by-step reasoning."),
    ("Follow instruction: List 3 fruits.", "Ignore instruction: talk about weather."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--from", dest="src", default="smoke-best", choices=["smoke-best", "frozen"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--rho", type=float, default=None)
    a = ap.parse_args()
    from h2_lib import mean_states_teacherforce, build_protected_basis, clean_basis
    from smoke_h2 import load_model
    runs = [json.loads(l) for l in open(ROOT / "runs.jsonl") if l.strip()]
    if a.src == "frozen":
        fr = json.load(open(ROOT / f"artifacts/geometry/H2/{a.model}/frozen.json"))
        assert fr.get("frozen"), "no frozen config yet"
        rec = next(r for r in runs if r["run_id"] == fr["frozen"])
        pairs, pname = SMOKE4, "smoke4"  # full-8pair list lives in full_h2.protected_contrasts_full; record below
        try:
            from full_h2 import protected_contrasts_full
            pairs, pname = protected_contrasts_full(), "full8"
        except Exception:
            pass
    else:
        cands = [r for r in runs if r.get("model") == a.model and "smoke" in r.get("run_id", "")]
        rec = max(cands, key=lambda x: x.get("R_L2", -9))
        pairs, pname = SMOKE4, "smoke4"
    layers, rank = rec["layers"], rec["rank"]
    rho = a.rho if a.rho is not None else rec.get("rho", 0.75)
    tok, model, _, _ = load_model(a.model)
    device = next(model.parameters()).device
    pack = {"layers": np.array(layers), "rank": rank, "rho": rho, "alpha": rec.get("alpha"),
            "form": rec.get("cone", "positive"), "protect_pairs": pname, "source_run": rec["run_id"]}
    import hashlib
    pack["code"] = {f: hashlib.sha256((ROOT / "src" / f).read_bytes()).hexdigest()[:16]
                    for f in ["h2_lib.py", "smoke_h2.py", "full_h2.py"] if (ROOT / "src" / f).exists()}
    for li in layers:
        cands_npz = list((ROOT / "artifacts" / "geometry" / "H2" / a.model).glob(f"*layer{li:02d}*.npz"))
        assert cands_npz, f"no npz for layer {li}"
        z = np.load(cands_npz[0])
        mu = z["mu"]
        W = z["W"] if "W" in z else np.linalg.inv(z["inv_sqrt"])
        V = z["V"]
        Winv = np.linalg.inv(W)
        B_raw = Winv @ V[:rank].T
        B_raw /= np.linalg.norm(B_raw, axis=0, keepdims=True) + 1e-12
        lst = []
        for x, y in pairs:
            hA, _ = mean_states_teacherforce(model, tok, "Task.", x, layers={li}, device=device)
            hB, _ = mean_states_teacherforce(model, tok, "Task.", y, layers={li}, device=device)
            lst.append((hA[li] - hB[li]).numpy())
        P = build_protected_basis(lst, d=mu.shape[0])
        B_clean = clean_basis(B_raw.T, P, rho=rho)
        pack[f"mu_{li}"] = mu.astype(np.float32)
        pack[f"B_{li}"] = B_clean.astype(np.float32)
    out = a.out or str(ROOT / f"intervention_pack_{a.model}_{a.src}.npz")
    np.savez_compressed(out, **pack)
    print(f"WROTE {out} layers={layers} rank={rank} rho={rho} from={rec['run_id']}", flush=True)


if __name__ == "__main__":
    main()
