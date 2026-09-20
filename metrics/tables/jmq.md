<!-- generated 2026-09-16T22:35:04.671329+00:00 by src/jmq_stats.py -->
## Ablated versus baseline

Pairs where both candidates refused are excluded before estimation and inference. Score gives ties half credit; difference is ablated minus baseline.

| Model | Included | Both refused | Ablated wins | Baseline wins | Ties | Ablated score (95% CI) | Baseline score | Difference | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| meta-llama/Llama-3.2-1B-Instruct | 178 | 22 | 61 | 117 | 0 | 0.3427 [0.2753, 0.4157] | 0.6573 | -31.5pp | 0.000261 |
| allenai/OLMo-2-0425-1B-Instruct | 200 | 0 | 74 | 126 | 0 | 0.3700 [0.3050, 0.4350] | 0.6300 | -26.0pp | 0.00174 |
| Qwen/Qwen3.5-0.8B | 198 | 2 | 93 | 104 | 1 | 0.4722 [0.4040, 0.5429] | 0.5278 | -5.6pp | 1 |
| Qwen/Qwen3.5-2B | 200 | 0 | 72 | 127 | 1 | 0.3625 [0.2975, 0.4300] | 0.6375 | -27.5pp | 0.000825 |
| google/gemma-4-E2B-it | 200 | 0 | 98 | 102 | 0 | 0.4900 [0.4250, 0.5600] | 0.5100 | -2.0pp | 1 |

## Human-reference audit

| Model | H2 vs human JMQ_H (95% CI) | Baseline vs human JMQ_H (95% CI) |
|---|---:|---:|
| meta-llama/Llama-3.2-1B-Instruct | 0.7475 [0.6550, 0.8450] | 0.8025 [0.7050, 0.8975] |
| allenai/OLMo-2-0425-1B-Instruct | 0.9400 [0.8400, 1.0400] | 0.9700 [0.8700, 1.0700] |
| Qwen/Qwen3.5-0.8B | 0.7425 [0.6500, 0.8400] | 0.7625 [0.6675, 0.8575] |
| Qwen/Qwen3.5-2B | 0.9625 [0.8674, 1.0600] | 1.0100 [0.9150, 1.1050] |
| google/gemma-4-E2B-it | 1.1950 [1.1000, 1.2900] | 1.2550 [1.1600, 1.3500] |
