# H2 vs. baseline on held-out TEST

| Model / condition | MMD ↓ | Direct JMQ ↑ | Token L2-1 ↓ |
|---|---:|---:|---:|
| Llama-1B · Baseline | 0.1072 | **0.657** | 0.03033 |
| Llama-1B · H2 | **0.0986** | 0.343 | **0.02889** |
| OLMo-1B · Baseline | 0.1094 | **0.630** | **0.02859** |
| OLMo-1B · H2 | **0.0919** | 0.370 | 0.02860 |
| Qwen-0.8B · Baseline | 0.0858 | **0.528** | 0.03324 |
| Qwen-0.8B · H2 | **0.0845** | 0.472 | **0.02697** |
| Qwen-2B · Baseline | 0.0892 | **0.637** | **0.03041** |
| Qwen-2B · H2 | **0.0744** | 0.362 | 0.03120 |
| Gemma-2B · Baseline | 0.1287 | **0.510** | **0.02894** |
| Gemma-2B · H2 | **0.1151** | 0.490 | 0.02913 |

MMD and Token L2-1 are distances to the human TEST distribution, so lower is closer. Direct JMQ is the refusal-filtered H2-vs-baseline preference score, so higher is preferred; the two condition scores are complementary within each model and are not cross-model ratings. Bold marks the numerically better value in each same-model pair. Holm-significant JMQ losses occur for Llama-1B, OLMo-1B, and Qwen-2B. MMD recovery percentages remain in the companion table.
