# H2 distributional shift on held-out TEST

| Model | MMD ↓ | R MMD ↑ | L2-1 ↓ | R L2-1 ↑ | L2-2 ↓ | R L2-2 ↑ | L2-3 ↓ | R L2-3 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-1B | 0.1072 → 0.0986<br>-8.0% · closer | +10.8% | 0.03033 → 0.02889<br>-4.8% · closer | +6.5% | 0.01175 → 0.01203<br>+2.4% · farther | -4.1% | 0.00552 → 0.00544<br>-1.5% · closer | +5.2% |
| OLMo-1B | 0.1094 → 0.0919<br>-16.0% · closer | +21.4% | 0.02859 → 0.02860<br>+0.02% · farther | -0.02% | 0.01099 → 0.01045<br>-4.9% · closer | +9.1% | 0.00497 → 0.00448<br>-9.9% · closer | +47.8% |
| Qwen-0.8B | 0.0858 → 0.0845<br>-1.5% · closer | +2.2% | 0.03324 → 0.02697<br>-18.9% · closer | +24.9% | 0.01149 → 0.01086<br>-5.5% · closer | +9.7% | 0.00499 → 0.00472<br>-5.4% · closer | +25.8% |
| Qwen-2B | 0.0892 → 0.0744<br>-16.6% · closer | +24.0% | 0.03041 → 0.03120<br>+2.6% · farther | -3.6% | 0.01070 → 0.01126<br>+5.2% · farther | -9.7% | 0.00460 → 0.00470<br>+2.2% · farther | -15.6% |
| Gemma-2B | 0.1287 → 0.1151<br>-10.5% · closer | +13.4% | 0.02894 → 0.02913<br>+0.6% · farther | -0.9% | 0.01271 → 0.01325<br>+4.3% · farther | -7.1% | 0.00698 → 0.00724<br>+3.7% · farther | -8.6% |

All four metrics are distances to the human TEST distribution, so lower is closer. The signed raw change is `100 × (H2 − baseline) / baseline`: **negative is better** because the distance fell. Human-gap recovery is `R = (baseline − H2) / (baseline − human floor)`: **positive is better** because H2 closed part of the baseline-to-human gap. Raw change and R can differ in magnitude because R accounts for the nonzero human-human floor.
