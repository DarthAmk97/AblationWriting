# Joint TEST distribution and JMQ results

Positive distance reduction means the ablated distribution moved toward the human corpus. R is human-gap recovery using order-matched frozen VAL2 human-human floors. MMD uses an RBF bandwidth selected from VAL2 human+baseline embeddings only and reused unchanged on TEST.

## Distributional movement

| Model | R L2-1 | L2-1 baseline→ablated | R L2-2 | L2-2 baseline→ablated | R L2-3 | L2-3 baseline→ablated | R MMD | MMD baseline→ablated | JSD baseline→ablated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| meta-llama/Llama-3.2-1B-Instruct | +0.065 | 0.03033→0.02889 (+4.8%) | -0.041 | 0.01175→0.01203 (-2.4%) | +0.052 | 0.00552→0.00544 (+1.5%) | +0.108 | 0.1072→0.0986 (+8.0%) | 0.0732→0.0678 (+7.4%) |
| allenai/OLMo-2-0425-1B-Instruct | -0.000 | 0.02859→0.02860 (-0.0%) | +0.091 | 0.01099→0.01045 (+4.9%) | +0.478 | 0.00497→0.00448 (+9.9%) | +0.214 | 0.1094→0.0919 (+16.0%) | 0.0816→0.0693 (+15.1%) |
| Qwen/Qwen3.5-0.8B | +0.249 | 0.03324→0.02697 (+18.9%) | +0.097 | 0.01149→0.01086 (+5.5%) | +0.258 | 0.00499→0.00472 (+5.4%) | +0.022 | 0.0858→0.0845 (+1.5%) | 0.0806→0.0755 (+6.4%) |
| Qwen/Qwen3.5-2B | -0.036 | 0.03041→0.03120 (-2.6%) | -0.097 | 0.01070→0.01126 (-5.2%) | -0.156 | 0.00460→0.00470 (-2.2%) | +0.240 | 0.0892→0.0744 (+16.6%) | 0.0768→0.0691 (+10.0%) |
| google/gemma-4-E2B-it | -0.009 | 0.02894→0.02913 (-0.6%) | -0.071 | 0.01271→0.01325 (-4.3%) | -0.086 | 0.00698→0.00724 (-3.7%) | +0.134 | 0.1287→0.1151 (+10.5%) | 0.1070→0.0950 (+11.1%) |

## Refusal-filtered quality and preservation

| Model | JMQ included/excluded | Ablated score | Baseline score | A−B | Holm p | IFEval retention | Capability retention | Harm-refusal retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| meta-llama/Llama-3.2-1B-Instruct | 178/22 | 0.3427 | 0.6573 | -31.5pp | 0.000261 | 95.9% | 100.5% | 61.1% |
| allenai/OLMo-2-0425-1B-Instruct | 200/0 | 0.3700 | 0.6300 | -26.0pp | 0.00174 | 98.8% | 103.3% | 93.3% |
| Qwen/Qwen3.5-0.8B | 198/2 | 0.4722 | 0.5278 | -5.6pp | 1 | 99.3% | 109.0% | 81.8% |
| Qwen/Qwen3.5-2B | 200/0 | 0.3625 | 0.6375 | -27.5pp | 0.000825 | 105.2% | 105.0% | 93.3% |
| google/gemma-4-E2B-it | 200/0 | 0.4900 | 0.5100 | -2.0pp | 1 | 103.4% | 106.9% | 100.0% |
