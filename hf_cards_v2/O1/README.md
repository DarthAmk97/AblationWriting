---
license: other
base_model: allenai/OLMo-2-0425-1B-Instruct
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
- olmo
datasets:
- szyszy/GEN
model-index:
- name: AblationWriting-H2-O1-cone
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
      value: 0.214
    - name: R_MMD v5
      type: recovery
      value: 0.332
  - task:
      type: text-generation
      name: Blind quality JMQ, refusal-filtered
    dataset:
      name: TEST-JMQ 400 pairs
      type: szyszy/GEN
    metrics:
    - name: ablated win share
      type: preference
      value: 0.37
---

# Cone-ablated OLMo-2-0425-1B for writing, H2 frozen run

## Abstract

OLMo-2-0425-1B-Instruct with a frozen rank-2 safety cone hooked into layers 8 to 11. The strongest cone win on the panel: TEST MMD recovery +0.332 against dose-matched random at -0.045, with lexical deletion trailing at +0.051. Blind quality preference goes the wrong way at 0.370. No GGUF by design, see Usage.

Collection: [AblationWriting](https://huggingface.co/collections/amkkk/ablationwriting-6a9e1c709941b6e3fd1c1dea). Full dump: [AblationWriting-H2-dump](https://huggingface.co/amkkk/AblationWriting-H2-dump). Code and paper: [DarthAmk97/AblationWriting](https://github.com/DarthAmk97/AblationWriting). Siblings: L1-cone, Q08-cone, Q20-cone, G2-cone in the same collection.

> [!IMPORTANT]
> This is not a standalone checkpoint and not a GGUF. You still need the parent weights, and the cone is a runtime hook, not baked weights. A static bake cannot express the positive-part clamp or the generated-tokens-only locus, so any baked file would be a different intervention. Load the parent, register the hook, generate.

> [!NOTE]
> Parent: [allenai/OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct). Refs: [Panickssery et al, arXiv:2404.13076](https://arxiv.org/abs/2404.13076), [Joad et al, arXiv:2602.02132](https://arxiv.org/abs/2602.02132). Embedder: [nvidia/llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b). Human refs: [szyszy/GEN](https://huggingface.co/datasets/szyszy/GEN).

## Method

Hook: h minus alpha times B times the positive part of B-transpose times (h minus mu), generated tokens only, BF16. Basis B is orthonormal native-space, rebuilt from full_layer npz files plus frozen.json rank and rho. LEX arm is a text-level deletion rule, K token ids at rate q, hash-decided per position, with LEX-MATCH outputs in this repo for comparison.

| Parameter | Value |
|---|---|
| base model | allenai/OLMo-2-0425-1B-Instruct, rev 48d788ec |
| frozen run | H2-full-s3-O1-W-BEST-r2-a0.75-rho0.5-cone |
| window, rank, alpha, rho | BEST, 2, 0.75, 0.5 |
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
base = allenai/OLMo-2-0425-1B-Instruct, rev 48d788ec
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
- Siblings: [L1-cone](https://huggingface.co/amkkk/AblationWriting-H2-L1-cone), [Q08-cone](https://huggingface.co/amkkk/AblationWriting-H2-Q08-cone), [Q20-cone](https://huggingface.co/amkkk/AblationWriting-H2-Q20-cone), [G2-cone](https://huggingface.co/amkkk/AblationWriting-H2-G2-cone)
- Collection: [AblationWriting](https://huggingface.co/collections/amkkk/ablationwriting-6a9e1c709941b6e3fd1c1dea)

## Evaluation

| Arm | TEST MMD recovery | 95 percent CI | Holm p vs cone |
|---|---|---|---|
| ANCHOR-CONE | +0.332 | +0.306 to +0.361 | 1.0 |
| RAND-RANK-DOSE | -0.045 | -0.068 to -0.023 | 0.0005 |
| LEX-MATCH | +0.051 | +0.048 to +0.055 | 0.0005 |
| Core TEST MMD | +0.214 | point | n/a |
| Blind JMQ | 0.370 ablated, worse | p = 0.00174 | n/a |

## Limitations

- Linear bake impossible: clamp plus generated-only locus cannot fuse into weights, so no GGUF.
- Weak frozen model overall, taken to TEST to fail honestly; the cone win here is relative, not absolute quality.
- Quality preference favors baseline; the cone humanizes distributions, not taste.
- One deterministic random draw; a second seed would strengthen the rank claim.
- English GEN refs only.

## Scope

Suitable for: reproducing the cone arm exactly, measuring placement of cone vs random vs lexical. Not suitable for: claiming quality gains, running from this repo without the parent, treating any future GGUF as the measured intervention.

## Provenance and SHAs

- Base rev 48d788eca847d4d7548f375ad03d3c9312f6139e. Frozen run H2-full-s3-O1-W-BEST-r2-a0.75-rho0.5-cone.
- v5 stats marker 8a9efe30. Dump commit dd9e18b0. Protocol CONTROL_PROTOCOL.md with v5-patch1.
- Sampler temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024, BF16.

## License

Other, matching the private research use. GEN human refs are CC-BY-NC. You still need a legal copy of the parent to run anything.

## Acknowledgements and citation

AllenAI for the parent. Panickssery et al for the steering-vector literature. Joad et al for refusal multidirectionality. NVIDIA for the embedder. GEN providers for human references.

```
@misc{ablationwriting-h2-o1-cone,
  title = {AblationWriting H2 frozen cone: OLMo-2-0425-1B-Instruct},
  author = {Abdullah Mujeeb Khawaja},
  year = {2026},
  howpublished = {https://huggingface.co/amkkk/AblationWriting-H2-O1-cone}
}
```
