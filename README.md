# AblationWriting: does cutting out the safety cone make models write like humans?

Short answer: the words get more human. The judge does not care. Long answer below, with receipts.

Five frozen safety cones on five small instruct models. 514-run mirror, 5,000 blind pairs, dose-matched falsification controls sealed 2026-09-20. Training-free, inference-time hooks only.

## The verdict in one table

| Claim | Verdict |
|---|---|
| Cone exists and does something causal | mixed |
| Cone moves text toward human distribution | mixed, MMD yes 5 of 5, taste no |
| Writing quality improves | REJECTED, L1, O1, Q20 significantly worse, rest parity |
| It is just the dose, any direction works | REJECTED, random hurts 5 of 5, p = 0.0005 |
| A word-deletion rule explains it | mixed, wins Q08, ties Q20 |

Details and CIs in claims_ledger.md. Nothing here is overstated; the ledger would not allow it.

## The cast, human names first

Pipeline codenames are internal shorthand from the eight-model sweep. Use the left column.

| Model | Codename | Cone | One-line read |
|---|---|---|---|
| Llama-3.2-1B-Instruct | L1 | rank 4, layers 8-11 | refusal overlap, lexical ties |
| OLMo-2-0425-1B-Instruct | O1 | rank 2, layers 8-11 | arxiv specialist, strongest cone win |
| Qwen3.5-0.8B | Q08 | rank 8, layers 18-22 | the honest null, lexical wins |
| Qwen3.5-2B | Q20 | rank 32, layers 11-15 | big cone, lexical ties at the top |
| Gemma-4-E2B-it | G2 | rank 8, layers 11-16 | cleanest cone win |

## The taste test, 5,000 blind pairs

The judge preferred baseline almost everywhere. Ablated win share, refusal-filtered: L1 0.343, O1 0.370, Q08 0.472, Q20 0.363, G2 0.490. Zero improvements, three significant losses.

Best cells on the whole panel:

| Cell | Score | Note |
|---|---|---|
| O1 arxiv | 27-13 | the one heroic cell, carries pooled arxiv to 102-98 |
| G2 wikipedia | 25-15 | the only wiki win on the panel |
| Q08 reddit | 23-17 | casual internet, Q08 territory |
| Q08 wikihow | 22-17 | same story, how-to land |

Worst cells:

| Cell | Score | Note |
|---|---|---|
| Q20 wikipedia | 5-34 | the judge has seen enough |
| O1 wikihow | 8-32 | the specialist tax, arxiv or nothing |
| L1 wikipedia | 9-29 | plus L1 lost half its wikihow cell to refusals, n = 20 |

Full grid with Wilson CIs: metrics/tables/jmq_domain.md.

## The staircase, words get simpler

Mean word length and long-word rate, every domain, no exceptions:

| Domain | Human | Ablated | Baseline |
|---|---|---|---|
| arxiv | 5.08 | 5.41 | 5.56 |
| reddit | 4.65 | 4.87 | 4.98 |
| story | 4.12 | 4.43 | 4.53 |
| wikihow | 4.42 | 4.72 | 4.86 |
| wikipedia | 5.09 | 5.26 | 5.44 |

Human writes shortest, baseline longest, ablated always in the middle. Ablation literally dumbs the diction down toward human level. Whether that counts as good writing is above our pay grade. The judge says no. Per-cell rates: metrics/tables/style_bigword.md.

Why the judge says no, in its own words: format decides 45 percent of pairs. When it talks fidelity it favors ablation 127 to 100. When it talks grammar it favors baseline 72 to 16. Truer, but uglier. Full themes: metrics/tables/jmq_rationale_themes.md.

## The showdown, cone vs random vs lexical, TEST MMD

| Model | Cone | Random, dose-matched | Lexical | Read |
|---|---|---|---|---|
| G2 | +0.193 | -0.067 | +0.028 | clean cone win |
| L1 | +0.092 | -0.036 | +0.062 | cone wins, lexical ties |
| O1 | +0.332 | -0.045 | +0.051 | strongest win on the panel |
| Q08 | +0.020 | -0.134 | +0.182 | cone null, lexical wins outright |
| Q20 | +0.411 | -0.091 | +0.436 | cone strong, lexical ties at the top |

Random hurts everywhere with p = 0.0005, so dose alone is dead. Lexical deletion is the rival that will not leave: it wins one, ties one. Full table with CIs: metrics/tables/controls.md.

## The blooper reel, v1 to v5

Four dead epochs before the one that lived. Each failure is a finding with a body count.

| Epoch | Cause of death | Finding |
|---|---|---|
| v1 | exact bit-replay across NVIDIA drivers, 8 of 8 mismatches before hooks ran | runtime identity crisis, not science |
| v2-v3 | PC1 needed alpha 25.15 against a cap of 4.0, carrying 1/17th of the cone | the top direction alone cannot explain the cone, multidirectionality evidence |
| v4 | died merging spreadsheets, lex-merge defect, deterministic code bug | progress survived and was adopted, bug documented |
| v5-patch1 | baseline sat ON the human floor for JSD, division by almost zero | flagged NaNs, kept visible, suite completed |

## The fine print that actually matters

- Refusals: 24 pairs excluded, 22 from L1. Refusal overlap rides along with the cone, it does not drive it. Retention exhibit 0.61.
- The judge prefers G2 outputs to actual human writing. The judge is a taste machine, not a human. Appendix H.
- Truncation at 4000 chars punishes whoever gets cut. Neither-cut pairs: ablated 0.420.
- One deterministic random draw across 24 layer bases. A second seed would be nice. It is not funded.

## Repo map, what is here and why

| Path | Why it exists |
|---|---|
| src/ | the entire pipeline, nothing trimmed: geometry sweeps, TEST, judging, MMD, controls, verification, watchdogs |
| metrics/tables/ | all 11 rendered results tables, the human-readable face of the results |
| metrics/*.parquet | the same results machine-readable, with CIs and Holm p-values |
| claims_ledger.md | every paper claim mapped to evidence and a verdict, the adjudication record |
| CONTROL_PROTOCOL.md | the frozen v5 control spec, including v5-patch1 and the decision rule |
| failures.md, lab_notebook.md, state_ledger.md | provenance: what failed, what was tried, what it cost |
| figures/ | PNG, SVG, and the data parquets behind each figure |
| exports/ | reviewer packs: prompt-level review and the 150-pair case probe |
| paper/ | section drafts, appendix, references; main manuscript compiles from these |
| manifest.yaml, DUMP_SPEC.md | frozen revisions, splits, and sampler, the identity of the study |
| judgments/ | 5,000 blind pairwise judgments in four shards plus the refusal audit |
| artifacts/geometry/H2/ | per-model small markers and metrics; bulk embeddings and bases live on HF |

## Data, here vs HF

Code, paper, tables, and small results live here. Bulk data, about 12 GB, lives in the private HF dump amkkk/AblationWriting-H2-dump under the AblationWriting collection with per-folder data cards. Per-model repos hold each frozen cone basis, generations, metrics, and a model card:

- amkkk/AblationWriting-H2-L1-cone, Llama-3.2-1B-Instruct
- amkkk/AblationWriting-H2-O1-cone, OLMo-2-0425-1B-Instruct
- amkkk/AblationWriting-H2-Q08-cone, Qwen3.5-0.8B
- amkkk/AblationWriting-H2-Q20-cone, Qwen3.5-2B
- amkkk/AblationWriting-H2-G2-cone, Gemma-4-E2B-it

No GGUFs by design: the cone hook removes only the positive projection on generated tokens, which no static weight edit can express exactly. Each model repo documents this with the numbers.

## Usage

Exact cone, transformers plus 30 lines: rebuild each layer basis with src/h2_controls.py build_control_bases from full_layer npz files plus frozen.json, apply src/h2_controls.py GeneratedTokenHook at temp 0.8, top-p 0.95, top-k 50. Same seeds as controls_plan.json reproduce our 400 TEST outputs exactly.

Analysis only, no GPU: every generation, metric, and judgment is a parquet. Pandas reproduces every table.

Lexical control: the exact deletion rule lives in controls_lexical_policy.json, K token ids at rate q, hash-decided per position, with our LEX-MATCH outputs beside it for comparison.

## Repro map

combined_test_jmq.md from combined_test_jmq.parquet via src/combined_results.py and src/test_headline.py. paired_mmd_jmq_l2.md and grouped_distribution_shift.md likewise. jmq.md, jmq_domain.md, jmq_truncation.md, style_bigword.md, jmq_rationale_themes.md from their parquets via src/jmq_stats.py. mmd.md from per-model mmd_metrics.json via src/mmd_score.py. controls.md from control_metrics.parquet via src/control_stats.py. Prompt review via src/export_jmq_review.py. A one-command verifier is tracked as repro/, in progress.

## Links

- HF collection: https://huggingface.co/collections/amkkk/ablationwriting-6a9e1c709941b6e3fd1c1dea
- Data dump: https://huggingface.co/amkkk/AblationWriting-H2-dump
- L1-cone: https://huggingface.co/amkkk/AblationWriting-H2-L1-cone
- O1-cone: https://huggingface.co/amkkk/AblationWriting-H2-O1-cone
- Q08-cone: https://huggingface.co/amkkk/AblationWriting-H2-Q08-cone
- Q20-cone: https://huggingface.co/amkkk/AblationWriting-H2-Q20-cone
- G2-cone: https://huggingface.co/amkkk/AblationWriting-H2-G2-cone

## References

- Panickssery et al, arXiv:2404.13076
- Joad et al, arXiv:2602.02132, EMNLP 2026

## License and privacy

Private research repo. GEN human references are CC-BY-NC. No model weights here. Pinned env: Python 3.12.14, Torch 2.11.0+cu128, Transformers 5.16.1.
