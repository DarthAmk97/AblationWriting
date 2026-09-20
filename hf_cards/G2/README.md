---
base_model:
- google/gemma-4-E2B-it
pipeline_tag: text-generation
library_name: transformers
license: other
tags:
- ablationwriting
- h2
- concept-cone-ablation
- inference-time-hook
- mechanistic-interpretability
- gemma
---

# H2 frozen cone: Gemma-4-E2B-it, run H2-full-s3-G2-W-A-r8-a0.5-rho1.0-cone

Inference-time safety-cone ablation. No weights changed, no training. Hook the residual stream at the listed layers with the frozen basis in basis/ and the run config in frozen.json.

## Settings

| Parameter | Value |
|---|---|
| base model | google/gemma-4-E2B-it (3e22461f) |
| frozen run | H2-full-s3-G2-W-A-r8-a0.5-rho1.0-cone |
| window / rank / alpha / rho | A / 8 / 0.5 / 1.0 |
| layers | 11, 12, 13, 14, 15, 16 |
| sampler | temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024 |
| hook | generated-token residual stream only, BF16 |

## Results, frozen held-out TEST

| Metric | Value |
|---|---|
| TEST MMD recovery, core | +0.134 |
| TEST MMD recovery, v5 contemporaneous | +0.193, 95 percent CI [+0.180, +0.206] |
| Dose-matched random MMD | -0.067, Holm p = 0.0005 vs cone |
| Lexical-match MMD | +0.028, LEX matched, ratio 1.056 |
| Blind quality JMQ, refusal-filtered | 0.4900 ablated, parity with baseline, p = 1 |

## Falsification read

The cone wins and lexical deletion barely registers. Random ablation hurts. Quality preference is parity, not improvement.

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
