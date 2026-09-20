---
license: other
base_model: meta-llama/Llama-3.2-1B-Instruct
pipeline_tag: text-generation
library_name: transformers
language:
- en
tags:
- ablationwriting
- h2
- concept-cone-ablation
- inference-time-hook
- mechanistic-interpretability
- llama
datasets:
- szyszy/GEN
model-index:
- name: AblationWriting-H2-L1-cone
  results:
  - task:
      type: text-generation
      name: TEST MMD recovery, cone
    dataset:
      name: TEST-JMQ 400 plus GEN human refs
      type: szyszy/GEN
    metrics:
    - name: R_MMD core
      type: recovery
      value: 0.108
    - name: R_MMD v5
      type: recovery
      value: 0.092
  - task:
      type: text-generation
      name: Blind quality JMQ, refusal-filtered
    dataset:
      name: TEST-JMQ 400 pairs
      type: szyszy/GEN
    metrics:
    - name: ablated win share
      type: preference
      value: 0.3427
---

# Llama-3.2-1B that writes simpler (H2 safety-cone ablation)

Take a 1B Llama, find the four activation directions tied to its safety-refusal voice, and shave them off as it writes. Word choice moves toward human level. Nothing is retrained and no weights change. Pipeline codename L1: internal shorthand from the eight-model sweep, safe to ignore.

Words we keep using: rank is how many directions get cut, here 4 out of 2048. Alpha is cut strength, here 0.75, meaning 75 percent of the positive projection comes off. MMD is distance between two piles of text; smaller against human is better. Recovery is the fraction of the baseline-to-human gap closed. JMQ is a blind taste test judged by another model where 0.50 is a coin flip.

## Abstract

TEST MMD recovery +0.092 against a dose-matched random control at -0.036, so direction matters, not just dose. Lexical deletion ties the cone here at +0.062. Blind quality preference goes the wrong way at 0.343. No GGUF by design, see Usage.

Collection: [AblationWriting](https://huggingface.co/collections/amkkk/ablationwriting-6a9e1c709941b6e3fd1c1dea). Full dump: [AblationWriting-H2-dump](https://huggingface.co/amkkk/AblationWriting-H2-dump). Code and paper: [DarthAmk97/AblationWriting](https://github.com/DarthAmk97/AblationWriting). Siblings: O1-cone, Q08-cone, Q20-cone, G2-cone in the same collection.

> [!IMPORTANT]
> This is not a standalone checkpoint and not a GGUF. You still need the parent weights, and the cone is a runtime hook, not baked weights. A static bake cannot express the positive-part clamp or the generated-tokens-only locus, so any baked file would be a different intervention. Load the parent, register the hook, generate.

> [!NOTE]
> Parent: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Refs: [Panickssery et al, arXiv:2404.13076](https://arxiv.org/abs/2404.13076), [Joad et al, arXiv:2602.02132](https://arxiv.org/abs/2602.02132). Embedder: [nvidia/llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b). Human refs: [szyszy/GEN](https://huggingface.co/datasets/szyszy/GEN).

## Method

Hook: h minus alpha times B times the positive part of B-transpose times (h minus mu), generated tokens only, BF16. Basis B is orthonormal native-space, rebuilt from full_layer npz files plus frozen.json rank and rho. LEX arm is a text-level deletion rule, K token ids at rate q, hash-decided per position, with LEX-MATCH outputs in this repo for comparison.

| Parameter | Value |
|---|---|
| base model | meta-llama/Llama-3.2-1B-Instruct, rev 92131767 |
| frozen run | H2-full-s3-L1-W-BEST-r4-a0.75-rho0.5-cone |
| window, rank, alpha, rho | BEST, 4, 0.75, 0.5 |
| layers | 8, 9, 10, 11 |
| sampler | temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024 |

## Files

| Path | What |
|---|---|
| basis/full_layer*.npz | frozen cone basis per layer |
| frozen.json | freeze record |
| test_generations.parquet | core TEST baseline and cone texts |
| controls_generations.parquet | v5 BASE, CONE, RAND, LEX plus HUMAN, 4048 rows |
| control_metrics.json | per-model metrics with CIs and Holm |
| mmd_metrics.json | frozen core MMD |

## Usage, cone arm, exact

Needs CUDA plus a git clone of the repo above for src/. First basis build takes minutes because the nuisance calibration runs forward passes; generation after that is fast.

```python
import sys
sys.path.insert(0, 'AblationWriting/src')
from pathlib import Path
import json
import torch
from h2_controls import load_target, build_control_bases, GeneratedTokenHook, set_seed, stable_prompt_seed

tok, model, layer_map = load_target('L1')  # pinned base and revision, BF16
model.eval()
geom = Path('AblationWriting-H2-L1-cone')  # this repo
record = json.load(open(geom / 'frozen.json'))
source = {'record': record, 'geom': geom, 'model': 'L1'}
bases = build_control_bases(model, tok, source, layer_map)
alpha = float(record['alpha'])
dev = next(model.parameters()).device
hooks, handles = {}, []
for layer in [8, 9, 10, 11]:
    hook = GeneratedTokenHook(bases[layer]['frozen'], bases[layer]['mu'], alpha, dev)
    hooks[layer] = hook
    handles.append(layer_map[layer].register_forward_hook(hook))

prompt = 'Explain how tides work.'
set_seed(stable_prompt_seed(prompt))
rendered = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
enc = {k: v.to(dev) for k, v in tok(rendered, return_tensors='pt').items()}
n = enc['input_ids'].shape[1]
for hook in hooks.values():
    hook.reset(n)
with torch.inference_mode():
    out = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95, top_k=50,
                         repetition_penalty=1.0, max_new_tokens=1024,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0, n:], skip_special_tokens=True))
for handle in handles:
    handle.remove()
```

Same seeds as controls_plan.json reproduce our 400 TEST outputs exactly.

## Usage, lexical arm, exact

The rival that ties the cone here. Exact text rule, needs only the eval tokenizer:

```python
import hashlib, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17', trust_remote_code=True)
pol = json.load(open('controls_lexical_policy.json'))
sel, q = set(pol['selected_token_ids']), pol['q']
MODEL, PROMPT_ID = 'L1', 'test_0001'  # same id plus same baseline text gives the same output, always
ids = tok(baseline_text, add_special_tokens=False)['input_ids']
kept = []
for i, t in enumerate(ids):
    payload = ('H2-MRSC|LEX|' + MODEL + '|' + PROMPT_ID + '|' + str(i)).encode()
    drop = t in sel and int(hashlib.sha256(payload).hexdigest(), 16) < q * 2**256
    if not drop:
        kept.append(t)
print(tok.decode(kept))
```

Our LEX-MATCH outputs for all 400 prompts sit in controls_generations.parquet, compare directly.

## Related artifacts

- Full data dump: [AblationWriting-H2-dump](https://huggingface.co/amkkk/AblationWriting-H2-dump)
- Code, paper, claims: [DarthAmk97/AblationWriting](https://github.com/DarthAmk97/AblationWriting)
- Siblings: [O1-cone](https://huggingface.co/amkkk/AblationWriting-H2-O1-cone), [Q08-cone](https://huggingface.co/amkkk/AblationWriting-H2-Q08-cone), [Q20-cone](https://huggingface.co/amkkk/AblationWriting-H2-Q20-cone), [G2-cone](https://huggingface.co/amkkk/AblationWriting-H2-G2-cone)
- Collection: [AblationWriting](https://huggingface.co/collections/amkkk/ablationwriting-6a9e1c709941b6e3fd1c1dea)

## Evaluation

| Arm | TEST MMD recovery | 95 percent CI | Holm p vs cone |
|---|---|---|---|
| ANCHOR-CONE | +0.092 | +0.059 to +0.126 | 1.0 |
| RAND-RANK-DOSE | -0.036 | -0.062 to -0.011 | 0.0005 |
| LEX-MATCH | +0.062 | +0.054 to +0.072 | 0.22, tie |
| Core TEST MMD | +0.108 | point | n/a |
| Blind JMQ | 0.343 ablated, worse | p = 0.000261 | n/a |

## Limitations

- Linear bake impossible: clamp plus generated-only locus cannot fuse into weights, so no GGUF.
- LEX unmatched on L1 with ratio 1.96, outside the 0.8 to 1.25 band.
- Quality preference favors baseline; the cone humanizes distributions, not taste.
- One deterministic random draw; a second seed would strengthen the rank claim.
- English GEN refs only.

## Scope

Suitable for: reproducing the cone arm exactly, measuring placement of cone vs random vs lexical, starting a bake with measured divergence. Not suitable for: claiming quality gains, running from this repo without the parent, treating any future GGUF as the measured intervention.

## Provenance and SHAs

- Base rev 9213176726f574b556790deb65791e0c5aa438b6. Frozen run H2-full-s3-L1-W-BEST-r4-a0.75-rho0.5-cone.
- v5 stats marker 8a9efe30. Dump commit dd9e18b0. Protocol CONTROL_PROTOCOL.md with v5-patch1.
- Sampler temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024, BF16.

## License

Other, matching the private research use. GEN human refs are CC-BY-NC. You still need a legal copy of the parent to run anything.

## Acknowledgements and citation

Llama team for the parent. Panickssery et al for the steering-vector literature. Joad et al for refusal multidirectionality. NVIDIA for the embedder. GEN providers for human references.

```
@misc{ablationwriting-h2-l1-cone,
  title = {AblationWriting H2 frozen cone: Llama-3.2-1B-Instruct},
  author = {Abdullah Mujeeb Khawaja},
  year = {2026},
  howpublished = {https://huggingface.co/amkkk/AblationWriting-H2-L1-cone}
}
```

## Field notes, JMQ breakdown

| Domain | Ablated | Baseline | Note |
|---|---|---|---|
| arxiv | 16 | 24 | outgunned |
| reddit | 11 | 29 | |
| story | 17 | 23 | |
| wikihow | 8 | 12 | n = 20, refusals ate half the cell |
| wikipedia | 9 | 29 | |

22 of the 24 bilateral-refusal exclusions on the whole panel live here. The cone overlaps the refusal direction and retention sits at 0.61, see the L1 exhibit in the paper methods. Refusal rides along, it does not drive. Case-level color: exports/jmq_case_probe.md.
