"""
diagnose_hook.py — why is O1 W-BEST [8-11] dead (rem=0) with full CAL2048+8pair geometry
when smoke W2 [8-11] fired with CAL512+4pair? 2x2 probe on VAL prompts (needs GPU, ~10 min):
  mu in {smoke-CAL512, full-CAL2048} x protect in {P4-smoke-pairs, P8-full-pairs}, rank 4, layer 8.
Reports mean c, pos-rate per variant using smoke-V vs full-V native bases.
"""
import json
from pathlib import Path
import numpy as np
import torch
ROOT = Path("/root/AblationWriting")
import sys
sys.path.insert(0, str(ROOT / "src"))
from h2_lib import load_splits, mean_states_teacherforce, build_protected_basis, clean_basis
from smoke_h2 import load_model
TOK = (ROOT / ".hf_token").read_text().strip()

SMOKE4 = [
    ("Answer concisely: What is the capital of France?", "Answer in detail with background: What is the capital of France?"),
    ("Return JSON: {\"capital\": \"?\"} for France. Only JSON.", "Describe France capital in prose."),
    ("Solve: 2+2=? Show only answer.", "Solve: 2+2=? Show step-by-step reasoning."),
    ("Follow instruction: List 3 fruits.", "Ignore instruction: talk about weather."),
]
from full_h2 import protected_contrasts_full as FULL8

mid = "O1"
li = 8
splits = load_splits(["val1"])
val = splits["val1"][:8]
tok, model, layer_map, _ = load_model(mid)
device = next(model.parameters()).device
# geometry
sm = np.load(ROOT / f"artifacts/geometry/H2/{mid}/layer{li:02d}_smoke.npz")
fu = np.load(ROOT / f"artifacts/geometry/H2/{mid}/full_layer{li:02d}.npz")
mu_s, Ws = sm["mu"], np.linalg.inv(sm["inv_sqrt"]) if "inv_sqrt" in sm else None
# smoke npz keys? saved mu, inv_sqrt, S, V, H_mean
print("smoke keys:", list(sm.keys()), flush=True)
print("full keys:", list(fu.keys()), flush=True)
mu_f, Wf, Vf = fu["mu"], fu["W"], fu["V"]
Vs = sm["V"]
print(f"mu shift ||mu_f-mu_s||={np.linalg.norm(mu_f - mu_s):.4f} mu_s_norm={np.linalg.norm(mu_s):.3f} mu_f_norm={np.linalg.norm(mu_f):.3f}", flush=True)
# native top dirs
Winv_s = np.linalg.inv(sm["inv_sqrt"])
Winv_f = np.linalg.inv(Wf)
b_s = Winv_s @ Vs[0]
b_f = Winv_f @ Vf[0]
b_s /= np.linalg.norm(b_s)
b_f /= np.linalg.norm(b_f)
print(f"cos(native top1 smoke, full)={float(b_s @ b_f):.4f}", flush=True)
# protect bases
def P_for(pairs):
    lst = []
    for a, b in pairs:
        hA, _ = mean_states_teacherforce(model, tok, "Task.", a, layers={li}, device=device)
        hB, _ = mean_states_teacherforce(model, tok, "Task.", b, layers={li}, device=device)
        lst.append((hA[li] - hB[li]).numpy())
    return build_protected_basis(lst, d=mu_f.shape[0])
P4 = P_for(SMOKE4)
P8 = P_for(FULL8())
print(f"P4 k={P4.shape[1]} P8 k={P8.shape[1]}", flush=True)
# probe VAL baseline states: use teacher-forced human text as proxy states? Better: mean VAL-human states
for mu, mn in [(mu_s, "mu512"), (mu_f, "mu2048")]:
    for P, pn in [(P4, "P4"), (P8, "P8")]:
        for V, W_, vn in [(Vs, sm["inv_sqrt"], "Vsmoke"), (Vf, Wf, "Vfull")]:
            Winv = np.linalg.inv(W_)
            B_raw = Winv @ V[:4].T
            B_raw /= np.linalg.norm(B_raw, axis=0, keepdims=True) + 1e-12
            B_cl = clean_basis(B_raw.T, P, rho=0.75)
            # project VAL human means
            pos_rates, means = [], []
            for r in val:
                hH, _ = mean_states_teacherforce(model, tok, r["prompt"], r["human_text"], layers={li}, device=device)
                d = (hH[li].numpy() - mu)
                c = d @ B_cl
                pos_rates.append(float((c > 0).mean()))
                means.append(float(c.mean()))
            print(f"{mn}x{pn}x{vn}: mean_c={np.mean(means):+.4f} posrate={np.mean(pos_rates):.3f} raw_norm_ratio={np.linalg.norm(B_raw - 0.75 * (P @ (P.T @ B_raw)))/np.linalg.norm(B_raw):.3f}", flush=True)
print("DIAG DONE", flush=True)
