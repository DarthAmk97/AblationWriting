---
base_model:
- Qwen/Qwen3.5-0.8B
pipeline_tag: text-generation
library_name: transformers
license: other
tags:
- ablationwriting
- h2
- concept-cone-ablation
- inference-time-hook
- mechanistic-interpretability
- qwen
---

# H2 frozen cone: Qwen3.5-0.8B, run H2-full-s3-Q08-W-A-r8-a1.0-rho0.5-cone

Inference-time safety-cone ablation. No weights changed, no training. Hook the residual stream at the listed layers with the frozen basis in basis/ and the run config in frozen.json.

## Settings

| Parameter | Value |
|---|---|
| base model | Qwen/Qwen3.5-0.8B (2fc06364) |
| frozen run | H2-full-s3-Q08-W-A-r8-a1.0-rho0.5-cone |
| window / rank / alpha / rho | A / 8 / 1.0 / 0.5 |
| layers | 18, 19, 20, 21, 22 |
| sampler | temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024 |
| hook | generated-token residual stream only, BF16 |

## Results, frozen held-out TEST

| Metric | Value |
|---|---|
| TEST MMD recovery, core | +0.022 |
| TEST MMD recovery, v5 contemporaneous | +0.020, 95 percent CI [-0.024, +0.060], crosses zero |
| Dose-matched random MMD | -0.134, Holm p = 0.0005 vs cone |
| Lexical-match MMD | +0.182, LEX matched and significantly better, p = 0.0005 |
| Blind quality JMQ, refusal-filtered | 0.4722 ablated, parity with baseline, p = 1 |

## Falsification read

The honest null of the panel. The cone effect on MMD is indistinguishable from zero here, random ablation hurts, and plain lexical deletion wins outright. Quality preference is parity.

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
