---
license: other
base_model: Qwen/Qwen3.5-2B
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
- qwen
datasets:
- szyszy/GEN
model-index:
- name: AblationWriting-H2-Q20-cone
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
      value: 0.24
    - name: R_MMD v5
      type: recovery
      value: 0.411
  - task:
      type: text-generation
      name: Blind quality JMQ, refusal-filtered
    dataset:
      name: TEST-JMQ 400 pairs
      type: szyszy/GEN
    metrics:
    - name: ablated win share
      type: preference
      value: 0.3625
---

# Cone-ablated Qwen3.5-2B for writing, H2 frozen run, cone with a rival

## Abstract

Qwen3.5-2B with a frozen rank-32 gentle-push cone hooked into layers 11 to 15. Strong cone at TEST MMD +0.411 against random at -0.091, but lexical deletion ties it at +0.436 with p = 0.22. Blind quality preference goes the wrong way at 0.363. No GGUF by design, see Usage. Both the cone and the lexical rule ship in this repo, pick your fighter.

Collection: [AblationWriting](https://huggingface.co/collections/amkkk/ablationwriting-6a9e1c709941b6e3fd1c1dea). Full dump: [AblationWriting-H2-dump](https://huggingface.co/amkkk/AblationWriting-H2-dump). Code and paper: [DarthAmk97/AblationWriting](https://github.com/DarthAmk97/AblationWriting). Siblings: L1-cone, O1-cone, Q08-cone, G2-cone in the same collection.

> [!IMPORTANT]
> This is not a standalone checkpoint and not a GGUF. You still need the parent weights. The cone is a runtime hook, the lexical arm is an exact text rule, and both are in this repo. A static cone bake cannot express the positive-part clamp or the generated-tokens-only locus, so any baked file would be a different intervention. Load the parent, register the hook, generate. Or apply the lexical rule.

> [!NOTE]
> Parent: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). Refs: [Panickssery et al, arXiv:2404.13076](https://arxiv.org/abs/2404.13076), [Joad et al, arXiv:2602.02132](https://arxiv.org/abs/2602.02132). Embedder: [nvidia/llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b). Human refs: [szyszy/GEN](https://huggingface.co/datasets/szyszy/GEN).

## Method

Hook: h minus alpha times B times the positive part of B-transpose times (h minus mu), generated tokens only, BF16. Basis B is orthonormal native-space, rebuilt from full_layer npz files plus frozen.json rank and rho. Gentle strength wins here where force failed on the family twin. LEX arm is a text-level deletion rule, K token ids at rate q, hash-decided per position, with LEX-MATCH outputs in this repo for comparison.

| Parameter | Value |
|---|---|
| base model | Qwen/Qwen3.5-2B, rev 15852e8c |
| frozen run | H2-full-s3-Q20-W-B-r32-a0.25-rho1.0-cone |
| window, rank, alpha, rho | B, 32, 0.25, 1.0 |
| layers | 11, 12, 13, 14, 15 |
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

Not AutoModelForCausalLM.from_pretrained on this repo alone. Two working arms, both reproducible.

```text
base = Qwen/Qwen3.5-2B, rev 15852e8c
model = load(base, torch_dtype=bfloat16, device_map=auto).eval()
for layer in [11, 12, 13, 14, 15]:
    B, mu = build_control_bases(npz(layer), frozen.json)  # src/h2_controls.py
    hook = GeneratedTokenHook(B, mu, alpha=0.25)           # generated tokens only
    register(layer, hook)
generate(temp=0.8, top_p=0.95, top_k=50)  # seeds in controls_plan.json
```

Lexical arm: read K, q, token ids from controls_lexical_policy.json and delete matching occurrences from baseline outputs. Needs transformers plus the two functions named above, both in the GitHub repo.

## Related artifacts

- Full data dump: [AblationWriting-H2-dump](https://huggingface.co/amkkk/AblationWriting-H2-dump)
- Code, paper, claims: [DarthAmk97/AblationWriting](https://github.com/DarthAmk97/AblationWriting)
- Siblings: [L1-cone](https://huggingface.co/amkkk/AblationWriting-H2-L1-cone), [O1-cone](https://huggingface.co/amkkk/AblationWriting-H2-O1-cone), [Q08-cone](https://huggingface.co/amkkk/AblationWriting-H2-Q08-cone), [G2-cone](https://huggingface.co/amkkk/AblationWriting-H2-G2-cone)
- Collection: [AblationWriting](https://huggingface.co/collections/amkkk/ablationwriting-6a9e1c709941b6e3fd1c1dea)

## Evaluation

| Arm | TEST MMD recovery | 95 percent CI | Holm p vs cone |
|---|---|---|---|
| ANCHOR-CONE | +0.411 | +0.370 to +0.457 | 1.0 |
| RAND-RANK-DOSE | -0.091 | -0.120 to -0.064 | 0.0005 |
| LEX-MATCH | +0.436 | +0.406 to +0.469 | 0.22, tie |
| Core TEST MMD | +0.240 | point | n/a |
| Blind JMQ | 0.363 ablated, worse | p = 0.000825 | n/a |

## Limitations

- Linear bake impossible: clamp plus generated-only locus cannot fuse into weights, so no GGUF.
- Lexical ties the cone here, so geometric uniqueness is not established on this model.
- Quality preference favors baseline; the cone humanizes distributions, not taste.
- One deterministic random draw; a second seed would strengthen the rank claim.
- English GEN refs only.

## Scope

Suitable for: reproducing either arm, comparing geometric vs lexical humanization at matched dose. Not suitable for: claiming quality gains, running from this repo without the parent, treating any future GGUF as the measured intervention.

## Provenance and SHAs

- Base rev 15852e8c16360a2fea060d615a32b45270f8a8fc. Frozen run H2-full-s3-Q20-W-B-r32-a0.25-rho1.0-cone.
- v5 stats marker 8a9efe30. Dump commit dd9e18b0. Protocol CONTROL_PROTOCOL.md with v5-patch1.
- Sampler temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024, BF16.

## License

Other, matching the private research use. GEN human refs are CC-BY-NC. You still need a legal copy of the parent to run anything.

## Acknowledgements and citation

Qwen team for the parent. Panickssery et al for the steering-vector literature. Joad et al for refusal multidirectionality. NVIDIA for the embedder. GEN providers for human references.

```
@misc{ablationwriting-h2-q20-cone,
  title = {AblationWriting H2 frozen cone: Qwen3.5-2B},
  author = {Abdullah Mujeeb Khawaja},
  year = {2026},
  howpublished = {https://huggingface.co/amkkk/AblationWriting-H2-Q20-cone}
}
```
