---
base_model:
- meta-llama/Llama-3.2-1B-Instruct
pipeline_tag: text-generation
library_name: transformers
license: other
tags:
- ablationwriting
- h2
- concept-cone-ablation
- inference-time-hook
- mechanistic-interpretability
- llama
---

# H2 frozen cone: Llama-3.2-1B-Instruct, run H2-full-s3-L1-W-BEST-r4-a0.75-rho0.5-cone

Inference-time safety-cone ablation. No weights changed, no training. Hook the residual stream at the listed layers with the frozen basis in basis/ and the run config in frozen.json.

## Settings

| Parameter | Value |
|---|---|
| base model | meta-llama/Llama-3.2-1B-Instruct (92131767) |
| frozen run | H2-full-s3-L1-W-BEST-r4-a0.75-rho0.5-cone |
| window / rank / alpha / rho | BEST / 4 / 0.75 / 0.5 |
| layers | 8, 9, 10, 11 |
| sampler | temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024 |
| hook | generated-token residual stream only, BF16 |

## Results, frozen held-out TEST

| Metric | Value |
|---|---|
| TEST MMD recovery, core | +0.108 |
| TEST MMD recovery, v5 contemporaneous | +0.092, 95 percent CI [+0.059, +0.126] |
| Dose-matched random MMD | -0.036, Holm p = 0.0005 vs cone |
| Lexical-match MMD | +0.062, LEX unmatched on L1, ratio 1.96, tie p = 0.22 |
| Blind quality JMQ, refusal-filtered | 0.3427 ablated, worse than baseline, p = 0.000261 |

## Falsification read

Dose-matched random ablation moves away from human while the cone moves toward it. Direction matters, not just dose. Lexical deletion ties the cone here. Quality preference does not improve.

## Files

| File | What |
|---|---|
| basis/full_layer*.npz | frozen cone basis per layer |
| frozen.json | freeze record |
| test_generations.parquet | core TEST baseline and cone texts |
| controls_generations.parquet | v5 BASE, CONE, RAND, LEX plus HUMAN, 4048 rows |
| control_metrics.json | per-model metrics with CIs and Holm |
| mmd_metrics.json | frozen core MMD |

## Repro

Full bundle: amkkk/AblationWriting-H2-dump at dd9e18b0. Regenerate with src/control_stats.py and src/h2_controls.py from the dump. Protocol: CONTROL_PROTOCOL.md, v5-patch1.

## License and privacy

Private. GEN human references are CC-BY-NC. No model weights in this repo.
