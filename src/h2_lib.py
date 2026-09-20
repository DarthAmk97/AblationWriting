"""
h2_lib.py — shared H2 utilities (PRE-TEST frozen logic)
- sampler, chat templating, teacher-forcing means, whitening/SVD, protected cleaning, hooks, metrics
Operates ONLY in /root/AblationWriting
"""
import os, json, hashlib, random
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/root/AblationWriting")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)

def load_splits(names):
    out = {}
    for n in names:
        p = ROOT / "data" / "splits" / f"{n}.jsonl"
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        out[n] = rows
    return out

def build_messages(prompt, completion=None):
    msgs = [{"role": "user", "content": prompt}]
    if completion is not None:
        msgs.append({"role": "assistant", "content": completion})
    return msgs

def tokenize_teacherforce(tokenizer, prompt, completion, device):
    """
    Returns input_ids, attention_mask, response_mask (1 for completion tokens).
    Uses chat template. Finds prompt boundary by tokenizing prompt-only then full.
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=2048)
    input_ids = torch.tensor([full["input_ids"]], device=device)
    attn = torch.tensor([full["attention_mask"]], device=device)
    # response mask: tokens after prompt_ids (allow 5-token tolerance for template quirks)
    # find longest prefix match
    full_ids = full["input_ids"]
    # locate split: assume first len(prompt_ids) tokens are prompt (clipped to full length)
    split = min(len(prompt_ids), len(full_ids))
    # refine: search for prompt suffix alignment? Keep simple.
    resp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    resp_mask[0, split:] = True
    # remove template/control tokens? We keep all response tokens; control tokens at end (eos) excluded
    # exclude last token if eos
    if full_ids and full_ids[-1] == tokenizer.eos_token_id:
        resp_mask[0, -1] = False
    return input_ids, attn, resp_mask

@torch.no_grad()
def mean_states_teacherforce(model, tokenizer, prompt, completion, layers=None, device="cuda"):
    """
    Returns dict layer_idx -> mean vector (d,) over response tokens, plus n_tokens.
    Uses output_hidden_states. FP32 for stats, model in BF16.
    """
    model.eval()
    input_ids, attn, resp_mask = tokenize_teacherforce(tokenizer, prompt, completion, device)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple (embed + each layer)
    # hs[0] embed, hs[1] layer0, etc.
    res = {}
    # resp positions
    pos = resp_mask[0].nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return res, 0
    for li in range(len(hs)-1):
        if layers is not None and li not in layers:
            continue
        h = hs[li+1][0, pos, :].float()  # (T,d) fp32
        # uniform mean
        m = h.mean(dim=0)  # (d,)
        res[li] = m.cpu()
    return res, len(pos)

def estimate_whitening(H_human: np.ndarray, ridge=1e-3):
    """
    H_human: (N,d) human means per layer.
    Returns mu (d,), Sigma_reg (d,d), inv_sqrt (d,d) via eigh, cond number.
    Deterministic ridge rule: lambda = ridge * trace(Sigma)/d  (ridge=1e-3 default)
    """
    mu = H_human.mean(axis=0)
    X = H_human - mu
    N, d = X.shape
    Sigma = (X.T @ X) / max(N-1, 1)
    lam = ridge * (np.trace(Sigma) / max(d, 1) + 1e-12)
    Sigma_reg = Sigma + lam * np.eye(d)
    # eigh
    vals, vecs = np.linalg.eigh(Sigma_reg)
    vals = np.maximum(vals, 1e-12)
    cond = float(vals.max() / vals.min())
    inv_sqrt = (vecs * (1.0/np.sqrt(vals))) @ vecs.T
    return mu, Sigma_reg, inv_sqrt, cond, lam

def discover_basis(D: np.ndarray, rank: int):
    """
    D: (M,d) stacked whitened displacements. Returns U,S,Vt (Vt rank x d), explained energy.
    """
    # truncated SVD
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    # top rank
    Vr = Vt[:rank, :]  # (r,d) — rows are directions in whitened space
    # explained energy
    energy = (S**2) / np.sum(S**2 + 1e-12)
    cum = np.cumsum(energy)
    eff_rank = int(np.sum(cum < 0.95) + 1)
    return {"U": U[:, :rank], "S": S, "V": Vr, "energy": energy[:rank], "cum": cum, "eff_rank": eff_rank}

def build_protected_basis(nuisance_list, d, ridge=1e-3):
    """
    nuisance_list: list of np arrays (d,) contrast vectors (already normalized per-task externally).
    Returns P (d,k) orthonormal via QR/SVD after ridge-whiten? For smoke, simple QR on stacked.
    """
    if not nuisance_list:
        return np.zeros((d, 0))
    M = np.stack(nuisance_list, axis=0)  # (K,d)
    # normalize each to unit norm so high-norm tasks don't dominate
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    M = M / norms
    # SVD then QR equivalent: orthonormal basis via SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    # keep components with S > 1e-6
    keep = S > 1e-6
    P = Vt[keep, :].T  # (d,k)
    # re-orthonormalize via QR
    if P.shape[1] > 0:
        Q, _ = np.linalg.qr(P)
        return Q
    return P

def clean_basis(B_raw_whitened_T, P_whitened, rho=0.75):
    """
    B_raw: (d,r) in whitened space? Actually protocol: orth[(I - rho P P^T) B_raw]
    Both in same (whitened or native?) space. We do in whitened space then map back.
    B_raw_whitened_T: (r,d) rows directions; convert to (d,r) cols.
    P_whitened: (d,k)
    Returns B_clean (d,r) orthonormal.
    """
    B = B_raw_whitened_T.T  # (d,r)
    if P_whitened.shape[1] > 0:
        proj = P_whitened @ (P_whitened.T @ B)
        B = B - rho * proj
    # orth via QR
    Q, _ = np.linalg.qr(B)
    return Q  # (d,r)

# ---- hooks ----
class ConeAblationHook:
    def __init__(self, B, mu, alpha=0.5, prefill_len=0):
        """
        B: (d,r) torch tensor on device, orthonormal, in native (unwhitened) space? We store native.
        mu: (d,) human mean native.
        """
        self.B = B
        self.mu = mu
        self.alpha = alpha
        self.prefill_len = prefill_len
        self.stats = {"removed_norm": 0.0, "tokens": 0, "pos_frac": 0.0}

    def __call__(self, module, inp, out):
        # out: tensor or tuple
        if isinstance(out, tuple):
            hs = out[0]
            rest = out[1:]
            is_tuple = True
        else:
            hs = out
            rest = None
            is_tuple = False
        # hs: (B,T,d)
        # KV-cache handling: during generate with use_cache, prefill is full prompt (T==prefill_len),
        # decode steps are single tokens (T==1) which are ALWAYS generated -> intervene.
        # Without cache, prefill+generated are concatenated (T>prefill_len) -> intervene tail only.
        T = hs.shape[1]
        if T == 1:
            # cached decode: this single token is generated
            h_gen = hs  # (B,1,d)
            gen_start = 0
        elif T > self.prefill_len:
            h_gen = hs[:, self.prefill_len:, :]  # (B,Tg,d)
            gen_start = self.prefill_len
        else:
            # prefill only, do not touch (protocol Sec 5 locus = generated only)
            return out
        # c = B^T (h - mu)
        diff = h_gen - self.mu.to(hs.device).to(hs.dtype)
        # B to same dtype/device
        B = self.B.to(hs.device).to(hs.dtype)  # (d,r)
        c = torch.matmul(diff, B)  # (B,Tg,r)
        c_pos = torch.clamp(c, min=0)
        # removed = alpha * B @ c_pos^T
        rem = self.alpha * torch.matmul(c_pos, B.T)  # (B,Tg,d)
        h_gen_new = h_gen - rem
        hs_new = hs.clone()
        hs_new[:, gen_start:, :] = h_gen_new
        # stats
        with torch.no_grad():
            self.stats["removed_norm"] += rem.float().norm(dim=-1).sum().item()
            self.stats["tokens"] += rem.shape[0]*rem.shape[1]
            self.stats["pos_frac"] += (c > 0).float().mean().item()
        if is_tuple:
            return (hs_new,) + tuple(rest)
        return hs_new

def find_layer_modules(model):
    """
    Returns dict idx -> module for residual blocks, handling multimodal.
    Tries: model.language_model.layers, model.model.layers, model.layers, transformer.h, etc.
    """
    candidates = [
        "language_model.layers",
        "model.language_model.layers",
        "model.layers",
        "transformer.h",
        "model.model.layers",
        "layers",
    ]
    # direct attribute walk
    for path in candidates:
        try:
            obj = model
            for p in path.split("."):
                obj = getattr(obj, p)
            # obj should be ModuleList
            if hasattr(obj, "__len__"):
                n = len(obj)
                if n >= 8:  # plausible
                    return {i: obj[i] for i in range(n)}, path
        except:
            continue
    # fallback: search for ModuleList containing blocks with self_attn
    found = {}
    for name, mod in model.named_modules():
        # look for ModuleList
        from torch import nn
        if isinstance(mod, nn.ModuleList):
            try:
                if len(mod) >= 8:
                    # check first element has self_attn or attn
                    ch = [n for n,_ in mod[0].named_children()]
                    if any("attn" in c.lower() for c in ch):
                        # ensure all similar?
                        return {i: mod[i] for i in range(len(mod))}, name
            except:
                continue
    return {}, "NOTFOUND"

# ---- metrics ----
def l2_ngram(p_texts, q_texts, tokenizer, n=1):
    """
    Fixed-tokenizer n-gram L2. Builds normalized frequency vectors over union vocab of n-grams in batch.
    Returns L2 distance + top contributors.
    p_texts, q_texts: lists of str
    """
    from collections import Counter
    def ngrams(text, n):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < n:
            return []
        return [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    cp, cq = Counter(), Counter()
    for t in p_texts:
        cp.update(ngrams(t, n))
    for t in q_texts:
        cq.update(ngrams(t, n))
    # normalize
    tp, tq = sum(cp.values()), sum(cq.values())
    if tp == 0 or tq == 0:
        return float("nan"), []
    keys = set(cp.keys()) | set(cq.keys())
    # vectorize in sorted order for determinism
    keys = sorted(keys)
    s = 0.0
    contribs = []
    for k in keys:
        pv = cp.get(k, 0)/tp
        qv = cq.get(k, 0)/tq
        d = (pv-qv)**2
        s += d
        contribs.append((k, d, pv, qv))
    contribs.sort(key=lambda x: -x[1])
    return float(np.sqrt(s)), contribs[:20]

def jsd_1gram(p_texts, q_texts, tokenizer, eps=1e-8):
    from collections import Counter
    def toks(t):
        return tokenizer(t, add_special_tokens=False)["input_ids"]
    cp, cq = Counter(), Counter()
    for t in p_texts:
        cp.update(toks(t))
    for t in q_texts:
        cq.update(toks(t))
    vocab = sorted(set(cp.keys()) | set(cq.keys()))
    tp, tq = sum(cp.values()), sum(cq.values())
    if tp == 0 or tq == 0:
        return float("nan")
    import math
    jsd = 0.0
    for k in vocab:
        p = cp.get(k, 0)/tp
        q = cq.get(k, 0)/tq
        m = 0.5*(p+q)
        if p > 0:
            jsd += 0.5*p*math.log((p+eps)/(m+eps))
        if q > 0:
            jsd += 0.5*q*math.log((q+eps)/(m+eps))
    return float(jsd)

def recovery(d_bh, d_ah, d_hh):
    denom = (d_bh - d_hh)
    if abs(denom) < 1e-12:
        return 0.0
    return float((d_bh - d_ah) / denom)

def pathologies(texts):
    stats = {"empty": 0, "rep_loop": 0, "total": len(texts)}
    for t in texts:
        if not t.strip():
            stats["empty"] += 1
        # repetition loop: same 20-gram repeated? simple: if any 10-token phrase repeats 5+ times consecutively? approximate by checking max char n-gram repeat
        # cheap: if len>50 and (len(set(t.split())) < 5): rep
        toks = t.split()
        if len(toks) > 50 and len(set(toks[:100])) < 8:
            stats["rep_loop"] += 1
    return stats
