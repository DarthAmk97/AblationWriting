---
base_model:
- allenai/OLMo-2-0425-1B-Instruct
pipeline_tag: text-generation
library_name: transformers
license: other
tags:
- ablationwriting
- h2
- concept-cone-ablation
- inference-time-hook
- mechanistic-interpretability
- olmo
---

# H2 frozen cone: OLMo-2-0425-1B-Instruct, run H2-full-s3-O1-W-BEST-r2-a0.75-rho0.5-cone

Inference-time safety-cone ablation. No weights changed, no training. Hook the residual stream at the listed layers with the frozen basis in basis/ and the run config in frozen.json.

## Settings

| Parameter | Value |
|---|---|
| base model | allenai/OLMo-2-0425-1B-Instruct (48d788ec) |
| frozen run | H2-full-s3-O1-W-BEST-r2-a0.75-rho0.5-cone |
| window / rank / alpha / rho | BEST / 2 / 0.75 / 0.5 |
| layers | 8, 9, 10, 11 |
| sampler | temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024 |
| hook | generated-token residual stream only, BF16 |

## Results, frozen held-out TEST

| Metric | Value |
|---|---|
| TEST MMD recovery, core | +0.214 |
| TEST MMD recovery, v5 contemporaneous | +0.332, 95 percent CI [+0.306, +0.361] |
| Dose-matched random MMD | -0.045, Holm p = 0.0005 vs cone |
| Lexical-match MMD | +0.051, LEX matched, ratio 0.932 |
| Blind quality JMQ, refusal-filtered | 0.3700 ablated, worse than baseline, p = 0.00174 |

## Falsification read

The cone clearly wins here. Dose-matched random hurts, lexical deletion trails far behind. Quality preference does not improve.

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
