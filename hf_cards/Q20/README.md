---
base_model:
- Qwen/Qwen3.5-2B
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

# H2 frozen cone: Qwen3.5-2B, run H2-full-s3-Q20-W-B-r32-a0.25-rho1.0-cone

Inference-time safety-cone ablation. No weights changed, no training. Hook the residual stream at the listed layers with the frozen basis in basis/ and the run config in frozen.json.

## Settings

| Parameter | Value |
|---|---|
| base model | Qwen/Qwen3.5-2B (15852e8c) |
| frozen run | H2-full-s3-Q20-W-B-r32-a0.25-rho1.0-cone |
| window / rank / alpha / rho | B / 32 / 0.25 / 1.0 |
| layers | 11, 12, 13, 14, 15 |
| sampler | temp 0.8, top-p 0.95, top-k 50, seed per prompt, cap 1024 |
| hook | generated-token residual stream only, BF16 |

## Results, frozen held-out TEST

| Metric | Value |
|---|---|
| TEST MMD recovery, core | +0.240 |
| TEST MMD recovery, v5 contemporaneous | +0.411, 95 percent CI [+0.370, +0.457] |
| Dose-matched random MMD | -0.091, Holm p = 0.0005 vs cone |
| Lexical-match MMD | +0.436, LEX matched, statistical tie, p = 0.22 |
| Blind quality JMQ, refusal-filtered | 0.3625 ablated, worse than baseline, p = 0.000825 |

## Falsification read

Strong cone with a worthy rival. Random ablation hurts, so direction matters, but lexical deletion ties the cone at the top. Quality preference does not improve.

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
