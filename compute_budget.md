# compute_budget.md — planning estimates + measured throughput (Sec 21, Sec 25)
# Currency: full-model generated tokens. T_H2,m ≈ sum_s G_m(N_s) q_s kappa + T_geometry + T_metrics

## Planning estimate (protocol, 1x RTX 3090 24GB, BF16, no quant for headline)
| Stage | Est GPU-h |
|---|---:|
| Baseline generation 8 models | 4–7 |
| CAL teacher-forcing + geometry | 3–6 |
| H2 successive-halving search | 12–20 |
| Frozen TEST generation | 5–8 |
| Preservation battery | 3–5 |
| MMD embedding + token metrics | 2–4 |
| StoryScope generation | 4–10 |
| H2 total | 30–70 GPU-h (~1.5–3 days wall) |

## Smoke (Q08+O1, 512 CAL, 128 VAL, ranks {2,4,8}, 2 windows, alpha {0.5,1.0}, rho 0.75, no StoryScope)
- Budget: 3–6 GPU-h. Purpose: validate hooks/storage/effect direction, NOT hypothesis decision.
- Residency: Q08+O1 together (0.8–1B class) where VRAM permits; Gemma E2B/E4B one at a time.

## Measured (to fill after 100-prompt smoke)
- G_m(N): TBD s for avg output tokens over N prompts, no intervention
- kappa_H2(m,r,w): TBD (target ~1.0; if >1.20 profile/vectorize)
- I_s = eliminated configs / GPU-h per stage: TBD
- E_compute = recovery / GPU-h, E_intervention = recovery / mean removed norm: TBD

## Precision
- BF16/FP16 per model stable path; no quantization for headline. Quantized replication only after BF16 complete.

## H2-MRSC-400-v5 controls — frozen 2026-09-17 (supersedes blocked v1–v4; v4 L1 generations adopted)
| Work | GPU-h |
|---|---:|
| Contemporaneous ANCHOR-BASE + ANCHOR-CONE, five models × (400 TEST-JMQ + 96 VAL1) | ~14–18 |
| RAND-RANK-DOSE, five frozen models × 400 TEST-JMQ | ~12.9 |
| 64-prompt dose calibration + 8-prompt same-runtime equivalence gates | ~1.4–2.0 |
| New control embeddings | ~2–4 |
| Nominal total | 45–51 |
| Hard cap | 72 |

PC1-DOSE dropped as dose-infeasible (v2 per-layer alpha 25.15, v3 total alpha 16.83 on L1, cap 4.0). Archived VAL2 full-subspace remains reused; lexical control is CPU-only. Target models run one at a time; no control overlaps another GPU worker.

## Parallelism priority
1. batch prompts within one model; 2. vectorize/interleave configs when hooks permit; 3. concurrent small-model workers to fill idle VRAM/CPU.
