# H2-MRSC-400-v5 controls (RAND-only; contemporaneous TEST anchors; archived VAL2 secondary)

TEST comparisons use same-runtime contemporaneous ANCHOR-BASE/ANCHOR-CONE with a total-dose matched RAND arm (PC1 dropped as dose-infeasible; see protocol). VAL2 FULL-SYM remains archived-original-runtime and is never pooled with v5 TEST. Distances use the pinned Qwen tokenizer and NVIDIA embedder. Recovery is relative to the split's human-human floor; positive values move toward human. CIs use 10,000 paired prompt-attribution bootstraps. Holm families are arm x metric across the five frozen models. *n/a = degenerate recovery denominator (split baseline already at the human floor, v5-patch1); raw distances are still reported and the block stays visible.

| Model | Split | Arm | R L2-1 | R L2-2 | R L2-3 | R JSD | R MMD | Holm p vs cone (MMD) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| G2 | TEST_JMQ | ANCHOR-BASE | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.0005 |
| G2 | TEST_JMQ | ANCHOR-CONE | -0.006 | -0.130 | -0.221 | +0.324 | +0.193 | 1 |
| G2 | TEST_JMQ | LEX-MATCH | +0.079 | +0.172 | +0.150 | +0.010 | +0.028 | 0.0005 |
| G2 | TEST_JMQ | RAND-RANK-DOSE | +0.045 | -0.247 | -0.744 | -0.265 | -0.067 | 0.0005 |
| L1 | TEST_JMQ | ANCHOR-BASE | +0.000 | +0.000 | +0.000 | n/a* | +0.000 | 0.0005 |
| L1 | TEST_JMQ | ANCHOR-CONE | +0.052 | -0.091 | +0.008 | n/a* | +0.092 | 1 |
| L1 | TEST_JMQ | LEX-MATCH | +0.074 | +0.173 | +0.253 | n/a* | +0.062 | 0.22 |
| L1 | TEST_JMQ | RAND-RANK-DOSE | -0.087 | -0.222 | -0.345 | n/a* | -0.036 | 0.0005 |
| O1 | TEST_JMQ | ANCHOR-BASE | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.0005 |
| O1 | TEST_JMQ | ANCHOR-CONE | +0.021 | +0.141 | +0.605 | +1.098 | +0.332 | 1 |
| O1 | TEST_JMQ | LEX-MATCH | +0.045 | +0.028 | +0.026 | +0.006 | +0.051 | 0.0005 |
| O1 | TEST_JMQ | RAND-RANK-DOSE | -0.141 | -0.019 | -0.312 | -1.033 | -0.045 | 0.0005 |
| Q08 | TEST_JMQ | ANCHOR-BASE | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.356 |
| Q08 | TEST_JMQ | ANCHOR-CONE | +0.279 | +0.140 | +0.203 | +0.263 | +0.020 | 1 |
| Q08 | TEST_JMQ | LEX-MATCH | +0.183 | +0.332 | +0.880 | -0.270 | +0.182 | 0.0005 |
| Q08 | TEST_JMQ | RAND-RANK-DOSE | -0.749 | -1.068 | -1.540 | -1.815 | -0.134 | 0.0005 |
| Q20 | TEST_JMQ | ANCHOR-BASE | +0.000 | +0.000 | +0.000 | n/a* | +0.000 | 0.0005 |
| Q20 | TEST_JMQ | ANCHOR-CONE | -0.114 | -0.170 | -0.254 | n/a* | +0.411 | 1 |
| Q20 | TEST_JMQ | LEX-MATCH | +0.091 | +0.214 | +1.077 | n/a* | +0.436 | 0.22 |
| Q20 | TEST_JMQ | RAND-RANK-DOSE | -0.182 | -0.040 | -0.066 | n/a* | -0.091 | 0.0005 |
| G2 | VAL2 | FULL-SYM | -0.090 | -0.312 | -0.139 | +0.250 | +0.217 | 0.0005 |
| G2 | VAL2 | VAL2-BASE | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.0005 |
| G2 | VAL2 | VAL2-CONE | +0.011 | -0.076 | -0.088 | +0.237 | +0.163 | 1 |
| L1 | VAL2 | FULL-SYM | +0.064 | -0.176 | -0.198 | +0.530 | +0.261 | 0.0005 |
| L1 | VAL2 | VAL2-BASE | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.0005 |
| L1 | VAL2 | VAL2-CONE | +0.135 | -0.007 | +0.113 | +0.524 | +0.167 | 1 |
| O1 | VAL2 | FULL-SYM | -0.076 | -0.008 | +0.391 | +0.639 | +0.317 | 0.0009 |
| O1 | VAL2 | VAL2-BASE | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.0005 |
| O1 | VAL2 | VAL2-CONE | +0.015 | +0.072 | +0.251 | +0.581 | +0.295 | 1 |
| Q08 | VAL2 | FULL-SYM | -0.471 | -0.885 | -1.339 | -0.540 | +0.398 | 0.0005 |
| Q08 | VAL2 | VAL2-BASE | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.143 |
| Q08 | VAL2 | VAL2-CONE | +0.245 | +0.059 | -0.151 | +0.228 | +0.024 | 1 |
| Q20 | VAL2 | FULL-SYM | -0.214 | -0.310 | -0.672 | +0.008 | +0.514 | 0.0005 |
| Q20 | VAL2 | VAL2-BASE | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.0005 |
| Q20 | VAL2 | VAL2-CONE | +0.012 | -0.034 | -0.071 | +0.565 | +0.421 | 1 |
