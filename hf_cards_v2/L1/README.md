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

# Cone-ablated Llama-3.2-1B for writing, H2 frozen run

## Abstract

Llama-3.2-1B-Instruct with a frozen rank-4 safety cone hooked into layers 8 to 11. TEST MMD recovery +0.092 against a dose-matched random control at -0.036, so direction matters, not just dose. Lexical deletion ties the cone here at +0.062. Blind quality preference goes the wrong way at 0.343. No GGUF by design, see Usage.

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

## Usage

Not AutoModelForCausalLM.from_pretrained on this repo alone. Load the parent, rebuild each layer basis, register the hook, generate.

```text
base = meta-llama/Llama-3.2-1B-Instruct, rev 92131767
model = load(base, torch_dtype=bfloat16, device_map=auto).eval()
for layer in [8, 9, 10, 11]:
    B, mu = build_control_bases(npz(layer), frozen.json)  # src/h2_controls.py
    hook = GeneratedTokenHook(B, mu, alpha=0.75)           # generated tokens only
    register(layer, hook)
generate(temp=0.8, top_p=0.95, top_k=50)  # seeds in controls_plan.json
```

Needs transformers plus the two functions named above, both in the GitHub repo. LEX rule: read K, q, token ids from controls_lexical_policy.json and apply to baseline outputs.

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
