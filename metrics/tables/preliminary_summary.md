# Preliminary H2 findings

Generated from the frozen VAL2/TEST tables, verified MMD artifacts, preservation artifacts, and `metrics/jmq_bootstrap.parquet`. Controls remain pending.

| Model | VAL2 R L2-1 | TEST R L2-1 | TEST R MMD | TEST JSD uplift | Refusal-filtered JMQ H2-vs-baseline (95% CI) | Excluded | Holm p | Harm-refusal retention | Preliminary read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| meta-llama/Llama-3.2-1B-Instruct | +0.135 | +0.065 | +0.108 | +7.4% | 0.3427 [0.2753, 0.4157] | 22 | 0.000261 | 61.1% | Small token recovery and modest embedding recovery; judge quality significantly worse |
| allenai/OLMo-2-0425-1B-Instruct | +0.015 | -0.000 | +0.214 | +15.1% | 0.3700 [0.3050, 0.4350] | 0 | 0.00174 | 93.3% | Unigram L2 flat but embedding recovery positive; judge quality significantly worse |
| Qwen/Qwen3.5-0.8B | +0.245 | +0.249 | +0.022 | +6.4% | 0.4722 [0.4040, 0.5429] | 2 | 1.0 | 81.8% | Strong token recovery but negligible embedding recovery; judge quality at parity |
| Qwen/Qwen3.5-2B | +0.012 | -0.036 | +0.240 | +10.0% | 0.3625 [0.2975, 0.4300] | 0 | 0.000825 | 93.3% | Token L2 worsens while embedding recovery is strongest; judge quality significantly worse |
| google/gemma-4-E2B-it | +0.011 | -0.009 | +0.134 | +11.1% | 0.4900 [0.4250, 0.5600] | 0 | 1.0 | Token L2 is flat/slightly worse while embedding recovery is positive; judge quality at parity |

## Current synthesis

- Held-out L2-1 supports heterogeneous recovery: strong for Qwen-0.8B, small for Llama, and flat or negative for the other three frozen models.
- TEST MMD recovery is positive for all five models, but it sharply disagrees with token metrics for Qwen-0.8B and Qwen-2B. These are point estimates until uncertainty and controls are complete.
- TEST JSD improves for all five models, supporting broad movement while leaving the affected distributional features model-dependent.
- JMQ does not support a quality-improvement claim. It is retained as a secondary stress test and qualitative audit, not treated as an oracle.
- Instruction following and capability are preserved under the frozen pass rule, but harmful-refusal retention falls for four models, showing that the writing cone is not behaviorally isolated.
- The named controls are still required before the final representation/selectivity claim can be closed.
